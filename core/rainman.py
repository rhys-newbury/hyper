"""Drifting loss for the exact hyperspherical setting."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Iterable, Literal

import torch
import torch.nn.functional as F

from core.drifting_loss import (  # noqa: F401
    FeatureSet,
    _ensure_nlc,
    feature_sets_ab_from_feature_map,
    feature_sets_from_feature_map,
    feature_sets_from_encoder_input,
    extract_feature_sets,
    flatten_latents_as_feature_set,
    compute_uncond_weight,
    DriftForm,
    Coupling,
    SinkhornMarginal,
    DistMetric,
    cost_matrix,
    _row_stochastic_from_logits,
    _partial_two_sided_from_logits,
    _sinkhorn_from_logits,
    _apply_uncond_bias_,
    _has_nonpositive_uncond_weight,
    _mask_self_neg_dist_,
    sample_power_law_omega,
)

_EPS = 1e-7
_SCALE_MAX = 1.0 / _EPS


# ---------------------------------------------------------------------------
# Riemannian primitives
# ---------------------------------------------------------------------------

def _pairwise_geodesic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise geodesic distances for unit vectors.

    x: [..., N, C], y: [..., M, C]
    returns: [..., N, M] in [0, pi]
    """
    cos = torch.matmul(x, y.transpose(-2, -1))
    return torch.acos(cos.clamp(-1.0 + _EPS, 1.0 - _EPS))


def _weighted_log_map(
    P: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Riemannian barycentric projection:
        sum_j P[i,j] log_{x_i}(y_j)

    log_x(y) = theta / sin(theta) * (y - cos(theta) x)

    P: [..., N, M]
    x: [..., N, C]
    y: [..., M, C]
    returns: [..., N, C]
    """
    cos = torch.matmul(x, y.transpose(-2, -1))
    cos = cos.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos)

    scale = (theta / torch.sin(theta).clamp_min(_EPS)).clamp_max(_SCALE_MAX)

    direction = (
        y.unsqueeze(-3)
        - cos.unsqueeze(-1) * x.unsqueeze(-2)
    )

    log_vecs = scale.unsqueeze(-1) * direction
    return (P.unsqueeze(-1) * log_vecs).sum(dim=-2)


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map:
        exp_x(v) = cos(||v||) x + sin(||v||) v / ||v||

    x: [..., C]
    v: [..., C]
    returns: [..., C]
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cos(v_norm) * x + torch.sin(v_norm) * (v / v_norm)


def _vmf_marginal(
    x: torch.Tensor,
    kappa: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    vMF-weighted empirical marginal for unit vectors.

    x: [..., N, C]
    returns: [..., N]

    kappa = 0 recovers uniform marginals.
    Larger kappa concentrates mass near the spherical batch mean direction.
    """
    n = x.shape[-2]

    if kappa <= 0:
        return x.new_full(x.shape[:-1], 1.0 / n)

    mu = x.mean(dim=-2)                         # [..., C]
    mu_norm = mu.norm(dim=-1, keepdim=True)     # [..., 1]

    valid = mu_norm.squeeze(-1) > eps
    mu = mu / mu_norm.clamp_min(eps)

    log_w = float(kappa) * (x * mu.unsqueeze(-2)).sum(dim=-1)  # [..., N]
    w = torch.softmax(log_w, dim=-1)

    uniform = x.new_full(w.shape, 1.0 / n)
    return torch.where(valid.unsqueeze(-1), w, uniform)


def _col_marginal(
    ny: int,
    nuncond: int,
    w: torch.Tensor | None,
    sinkhorn_marginal: SinkhornMarginal,
    ref: torch.Tensor,
) -> torch.Tensor:
    """Build Sinkhorn column marginal, handling CFG weighting."""
    if sinkhorn_marginal == "weighted_cols" and nuncond > 0 and w is not None:
        col_w = ref.new_ones(ny)
        col_w[-nuncond:] = w.clamp(min=1e-6).to(ref.device, ref.dtype)
        return col_w / col_w.sum()
    return ref.new_full((ny,), 1.0 / ny)


def _apply_weighted_uncond_to_batched_col_marginal(
    c_m: torch.Tensor,
    nuncond: int,
    w: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply CFG unconditional weighting to a batched column marginal.

    c_m: [..., ny]
    w: [nuncond] or broadcastable to last nuncond entries
    """
    if nuncond <= 0 or w is None:
        return c_m

    c_m = c_m.clone()
    c_m[..., -nuncond:] *= w.clamp(min=1e-6).to(c_m.device, c_m.dtype)
    return c_m / c_m.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

def drifting_loss_hyperspherical(
    x_feat: torch.Tensor,
    y_pos_feat: torch.Tensor,
    y_uncond_feat: torch.Tensor,
    *,
    omega: torch.Tensor,
    temps: Iterable[float],
    impl: Literal["logspace", "kernel"] = "logspace",
    vanilla: bool = False,
    drift_form: DriftForm = "alg2_joint",
    coupling: Coupling = "partial_two_sided",
    sinkhorn_iters: int = 20,
    sinkhorn_marginal: SinkhornMarginal = "none",
    sinkhorn_agg_kernel: bool = False,
    mask_self_neg: bool = True,
    dist_metric: str = "geodesic",
    normalize_drift_theta: bool = False,
    drift_tau_scale: bool = False,
    drift_unit_vec: bool = False,
    use_expmap_target: bool = False,
    vmf_marginals: bool = True,
    vmf_kappa: float = 1.0,
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Drifting loss for features on S^{C-1}.

    Key differences from the Euclidean version:
    - Distances are geodesic, bounded in [0, pi].
    - No S_j feature scaling.
    - No sqrt(C) temperature scaling.
    - Drift field is computed via log map in tangent space.
    - Loss target is x + v by default, or exp_x(v) if use_expmap_target=True.

    Args:
        vmf_marginals:
            If True, Sinkhorn row and column marginals are weighted by a
            von Mises--Fisher density around each layer's spherical batch mean.
            Only affects coupling='sinkhorn'.

        vmf_kappa:
            vMF concentration parameter. vmf_kappa=0 recovers uniform marginals.
            Suggested ablation: {0, 0.5, 1, 2, 5, 10}.
    """
    amp_off = (
        torch.autocast(device_type="cuda", enabled=False)
        if x_feat.device.type == "cuda"
        else nullcontext()
    )

    def _impl(x_feat, y_pos_feat, y_uncond_feat):
        x_feat = _ensure_nlc(x_feat, "x_feat")
        y_pos_feat = _ensure_nlc(y_pos_feat, "y_pos_feat")
        y_uncond_feat = _ensure_nlc(y_uncond_feat, "y_uncond_feat")

        nneg, l, c = x_feat.shape
        npos = y_pos_feat.shape[0]
        nuncond = y_uncond_feat.shape[0]

        if y_pos_feat.shape[1:] != (l, c) or y_uncond_feat.shape[1:] != (l, c):
            raise ValueError("Feature shapes must match across x/pos/uncond")
        if coupling == "sinkhorn" and impl != "logspace":
            raise ValueError("coupling='sinkhorn' requires impl='logspace'")
        if sinkhorn_marginal != "none" and coupling != "sinkhorn":
            raise ValueError("sinkhorn_marginal only used with coupling='sinkhorn'")
        x = x_feat.permute(1, 0, 2).contiguous().detach().float()
        yp = y_pos_feat.permute(1, 0, 2).contiguous().detach().float()

        yuc = (
            y_uncond_feat.permute(1, 0, 2).contiguous().detach().float()
            if nuncond > 0
            else None
        )

        with torch.no_grad():
            omega_s = omega.detach().to(x.device, torch.float32).reshape(())

            if nuncond > 0:
                yn = torch.cat([x, yuc], dim=1)
                w = compute_uncond_weight(
                    omega_s,
                    nneg=nneg,
                    nuncond=nuncond,
                ).to(x.device, torch.float32)

                if (
                    coupling == "sinkhorn"
                    and sinkhorn_marginal == "none"
                    and _has_nonpositive_uncond_weight(w)
                ):
                    raise ValueError(
                        "coupling='sinkhorn' + sinkhorn_marginal='none' + w<=0 "
                        "is infeasible. Use sinkhorn_marginal='weighted_cols'."
                    )
            else:
                yn = x
                w = None

            dp = _pairwise_geodesic(x, yp)
            dn = _pairwise_geodesic(x, yn)
            # Normalize to [0,1] so tau is geometry-aware (tau=1 = maximally entropic).
            dp = dp / math.pi
            dn = dn / math.pi
            
            if stats is not None:
                stats["drift_c"] = float(c)
                stats["drift_mean_dist_pos"] = float(dp.mean())
                stats["drift_mean_dist_neg"] = float(dn.mean())
                stats["drift_vmf_enabled"] = float(bool(vmf_marginals))
                stats["drift_vmf_kappa"] = float(vmf_kappa)

            if mask_self_neg and coupling != "sinkhorn":
                _mask_self_neg_dist_(dn)

            w = w.to(x.device, dp.dtype) if w is not None else None

            v_agg = torch.zeros((l, nneg, c), device=x.device, dtype=x.dtype)

            for rho in temps:
                tau = float(rho)
                kappa_t = min(1.0 / tau, 10.0) if vmf_marginals else 0.0
                if drift_form == "alg2_joint":
                    lp = -dp / tau
                    ln = -dn / tau

                    if coupling != "sinkhorn" or sinkhorn_marginal == "none":
                        _apply_uncond_bias_(ln, nuncond=nuncond, uncond_weight=w)

                    logits = torch.cat([lp, ln], dim=-1)

                    if coupling == "partial_two_sided":
                        log_r = torch.logsumexp(logits, dim=-1, keepdim=True)
                        log_c_ = torch.logsumexp(logits, dim=-2, keepdim=True)
                        log_a = logits - 0.5 * (log_r + log_c_)
                        log_a = log_a.masked_fill(torch.isneginf(logits), -math.inf)
                        A = torch.exp(log_a)

                    elif coupling == "row":
                        A = _row_stochastic_from_logits(logits)

                    elif coupling == "sinkhorn":
                        nx, ny = nneg, int(logits.shape[-1])

                        if vmf_marginals:
                            r = _vmf_marginal(x, kappa_t)  # [L, Nneg]

                            y_joint = torch.cat([yp, yn], dim=1)
                            c_m = _vmf_marginal(y_joint, kappa_t)  # [L, ny]

                            if sinkhorn_marginal == "weighted_cols":
                                c_m = _apply_weighted_uncond_to_batched_col_marginal(
                                    c_m,
                                    nuncond,
                                    w,
                                )
                        else:
                            r = logits.new_full((nx,), 1.0 / nx)
                            c_m = _col_marginal(
                                ny,
                                nuncond,
                                w,
                                sinkhorn_marginal,
                                logits,
                            )

                        A = _sinkhorn_from_logits(
                            logits,
                            r=r,
                            c=c_m,
                            iters=int(sinkhorn_iters),
                        )

                        if stats is not None:
                            tag = str(rho).replace(".", "p")
                            stats[f"drift_sinkhorn_row_mae_{tag}"] = float(
                                (A.sum(-1) - r).abs().mean()
                            )
                            stats[f"drift_sinkhorn_col_mae_{tag}"] = float(
                                (A.sum(-2) - c_m).abs().mean()
                            )

                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    Ap = A[..., :npos]
                    An = A[..., npos:]

                    Wp = Ap * An.sum(-1, keepdim=True)
                    Wn = An * Ap.sum(-1, keepdim=True)

                    v_raw = (
                        _weighted_log_map(Wp, x, yp)
                        - _weighted_log_map(Wn, x, yn)
                    )

                elif drift_form == "split":
                    lp = kappa_t * torch.matmul(x, yp.transpose(-2, -1))  # pure dot product, no acos

                    if coupling == "row":
                        Pp = _row_stochastic_from_logits(lp)

                    elif coupling == "partial_two_sided":
                        Ap = _partial_two_sided_from_logits(lp, impl=impl)
                        Pp = Ap / Ap.sum(-1, keepdim=True).clamp_min(1e-12)

                    elif coupling == "sinkhorn":
                        if vmf_marginals:
                            r = _vmf_marginal(x, kappa_t)
                            c_m = _vmf_marginal(yp, kappa_t)
                        else:
                            r = lp.new_full((nneg,), 1.0 / nneg)
                            c_m = lp.new_full((npos,), 1.0 / npos)

                        Pi = _sinkhorn_from_logits(
                            lp,
                            r=r,
                            c=c_m,
                            iters=int(sinkhorn_iters),
                        )
                        Pp = Pi / Pi.sum(-1, keepdim=True).clamp_min(1e-12)

                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    drift_pos = _weighted_log_map(Pp, x, yp)

                    # ln = -dn / tau
                    ln = kappa_t * torch.matmul(x, yn.transpose(-2, -1))

                    if coupling != "sinkhorn" or sinkhorn_marginal == "none":
                        _apply_uncond_bias_(ln, nuncond=nuncond, uncond_weight=w)

                    if coupling == "row":
                        Pn = _row_stochastic_from_logits(ln)

                    elif coupling == "partial_two_sided":
                        An = _partial_two_sided_from_logits(ln, impl=impl)
                        Pn = An / An.sum(-1, keepdim=True).clamp_min(1e-12)

                    elif coupling == "sinkhorn":
                        ny_n = int(ln.shape[-1])

                        if vmf_marginals:
                            r_n = _vmf_marginal(x, kappa_t)
                            c_m = _vmf_marginal(yn, kappa_t)

                            if sinkhorn_marginal == "weighted_cols":
                                c_m = _apply_weighted_uncond_to_batched_col_marginal(
                                    c_m,
                                    nuncond,
                                    w,
                                )
                        else:
                            r_n = ln.new_full((nneg,), 1.0 / nneg)
                            c_m = _col_marginal(
                                ny_n,
                                nuncond,
                                w,
                                sinkhorn_marginal,
                                ln,
                            )

                        Pi_n = _sinkhorn_from_logits(
                            ln,
                            r=r_n,
                            c=c_m,
                            iters=int(sinkhorn_iters),
                        )
                        Pn = Pi_n / Pi_n.sum(-1, keepdim=True).clamp_min(1e-12)

                        if sinkhorn_marginal == "post_guidance" and nuncond > 0:
                            Pn[..., -nuncond:] *= omega_s
                            Pn = Pn / Pn.sum(-1, keepdim=True).clamp_min(1e-12)

                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    drift_neg = _weighted_log_map(Pn, x, yn)
                    v_raw = drift_pos - drift_neg

                else:
                    raise ValueError(f"Unknown drift_form: {drift_form}")

                theta_norm = (v_raw * v_raw).mean().clamp_min(1e-12).sqrt()

                if stats is not None:
                    stats[f"drift_theta_{str(rho).replace('.', 'p')}"] = float(
                        theta_norm
                    )

                v_agg = v_agg + (
                    v_raw / theta_norm if normalize_drift_theta else v_raw
                )

            # Enforce tangency numerically.
            v_agg = v_agg - (v_agg * x).sum(dim=-1, keepdim=True) * x

            v_agg_nlc = v_agg.permute(1, 0, 2).contiguous()

            if stats is not None:
                stats["drift_v_rms"] = float(
                    (v_agg_nlc * v_agg_nlc).mean().clamp_min(0).sqrt()
                )
                stats["drift_v_nonfinite_count"] = float(
                    (~torch.isfinite(v_agg_nlc)).sum()
                )
                stats["drift_v_nan_count"] = float(torch.isnan(v_agg_nlc).sum())

        x_feat_f = x_feat.float()

        if use_expmap_target:
            target = exp_map(x_feat_f.detach(), v_agg_nlc).detach()
        else:
            target = F.normalize(x_feat_f.detach() + v_agg_nlc, dim=-1).detach()

        cos = (x_feat_f * target.detach()).sum(dim=-1).clamp(-1 + _EPS, 1 - _EPS)
        theta_sq = torch.acos(cos).pow(2)
        return theta_sq.mean()

    with amp_off:
        return _impl(x_feat, y_pos_feat, y_uncond_feat)


# Drop-in alias
drifting_loss_for_feature_set = drifting_loss_hyperspherical