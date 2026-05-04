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
# Clamp for theta/sin(theta) — prevents antipodal pairs from blowing up v_raw.
# pi / _EPS would be ~3e7; we use a much tighter practical bound.
_SCALE_MAX = 1.0 / _EPS


# ---------------------------------------------------------------------------
# Riemannian primitives
# ---------------------------------------------------------------------------

def _pairwise_geodesic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise geodesic distances for unit vectors.

    x: [..., N, C],  y: [..., M, C]
    Returns [..., N, M] in [0, pi].
    """
    cos = torch.matmul(x, y.transpose(-2, -1))
    return torch.acos(cos.clamp(-1.0 + _EPS, 1.0 - _EPS))


def _weighted_log_map(
    P: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Riemannian barycentric projection: sum_j P[i,j] * log_{x_i}(y_j).

    log_x(y) = (theta / sin(theta)) * (y - cos(theta)*x)

    Numerically: scale = theta/sin(theta) is clamped to [0, _SCALE_MAX]
    to handle near-antipodal pairs (theta -> pi, sin(theta) -> 0).

    P: [..., N, M]
    x: [..., N, C]  unit vectors (base points)
    y: [..., M, C]  unit vectors (target points)
    Returns [..., N, C]  tangent vectors at x.
    """
    cos   = torch.matmul(x, y.transpose(-2, -1))           # [..., N, M]
    cos   = cos.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos)                                 # [..., N, M]

    # FIX: clamp from above to handle antipodal pairs (theta -> pi)
    scale = (theta / torch.sin(theta).clamp_min(_EPS)).clamp_max(_SCALE_MAX)

    direction = (
        y.unsqueeze(-3)                                     # [...,  1, M, C]
        - cos.unsqueeze(-1) * x.unsqueeze(-2)               # [..., N, M, C]
    )                                                       # [..., N, M, C]

    log_vecs = scale.unsqueeze(-1) * direction              # [..., N, M, C]
    return (P.unsqueeze(-1) * log_vecs).sum(dim=-2)         # [..., N, C]


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map: exp_x(v) = cos(||v||)*x + sin(||v||)*v_hat.

    x: [..., C]  unit vector
    v: [..., C]  tangent vector at x
    Returns [..., C]  point on S^{C-1}.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cos(v_norm) * x + torch.sin(v_norm) * (v / v_norm)


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
    normalize_drift_theta: bool = True,
    drift_tau_scale: bool = False,
    drift_unit_vec: bool = False,
    use_expmap_target: bool = False,
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Drifting loss for features on S^{C-1}.

    Key differences from the Euclidean version:
    - Distances are geodesic (arccos), bounded in [0, pi].
    - No S_j feature scaling (geodesic distances are scale-invariant).
    - No sqrt(C) temperature scaling (distances don't grow with C).
    - Drift field computed via log map (tangent space).
    - Loss target: x + v_agg (ambient, like the original) by default,
      or exp_x(v_agg) (on-sphere) when use_expmap_target=True.

    Args:
        use_expmap_target: if True, map the drift back onto the sphere via
            exp_map before computing MSE. The gradient signal is slightly
            compressed but the target is geometrically exact. Default False
            (ambient target, matching the original loss convention).

    All other args are API-compatible with drifting_loss_for_feature_set.
    vanilla / dist_metric / drift_tau_scale / drift_unit_vec are accepted
    but have no effect (documented no-ops for this variant).
    """
    amp_off = (
        torch.autocast(device_type="cuda", enabled=False)
        if x_feat.device.type == "cuda"
        else nullcontext()
    )

    def _impl(x_feat, y_pos_feat, y_uncond_feat):
        x_feat        = _ensure_nlc(x_feat,        "x_feat")
        y_pos_feat    = _ensure_nlc(y_pos_feat,    "y_pos_feat")
        y_uncond_feat = _ensure_nlc(y_uncond_feat, "y_uncond_feat")

        nneg, l, c = x_feat.shape
        npos    = y_pos_feat.shape[0]
        nuncond = y_uncond_feat.shape[0]

        if y_pos_feat.shape[1:] != (l, c) or y_uncond_feat.shape[1:] != (l, c):
            raise ValueError("Feature shapes must match across x/pos/uncond")
        if coupling == "sinkhorn" and impl != "logspace":
            raise ValueError("coupling='sinkhorn' requires impl='logspace'")
        if sinkhorn_marginal != "none" and coupling != "sinkhorn":
            raise ValueError("sinkhorn_marginal only used with coupling='sinkhorn'")

        # Permute to [L, N, C] for batched distance computation
        x  = x_feat.permute(1, 0, 2).contiguous().detach().float()
        yp = y_pos_feat.permute(1, 0, 2).contiguous().detach().float()
        yuc = (y_uncond_feat.permute(1, 0, 2).contiguous().detach().float()
               if nuncond > 0 else None)

        with torch.no_grad():
            omega_s = omega.detach().to(x.device, torch.float32).reshape(())

            if nuncond > 0:
                yn = torch.cat([x, yuc], dim=1)    # [L, Nneg+Nunc, C]
                w  = compute_uncond_weight(omega_s, nneg=nneg, nuncond=nuncond).to(
                         x.device, torch.float32)
                if (coupling == "sinkhorn" and sinkhorn_marginal == "none"
                        and _has_nonpositive_uncond_weight(w)):
                    raise ValueError(
                        "coupling='sinkhorn' + sinkhorn_marginal='none' + w<=0 "
                        "is infeasible. Use sinkhorn_marginal='weighted_cols'."
                    )
            else:
                yn = x
                w  = None

            # Geodesic distance matrices — no S_j scaling, no sqrt(C) on temps
            dp = _pairwise_geodesic(x, yp)    # [L, Nneg, Npos]
            dn = _pairwise_geodesic(x, yn)    # [L, Nneg, Nneg(+Nunc)]

            if stats is not None:
                stats["drift_c"] = float(c)
                stats["drift_mean_dist_pos"] = float(dp.mean())
                stats["drift_mean_dist_neg"] = float(dn.mean())

            if mask_self_neg and coupling != "sinkhorn":
                _mask_self_neg_dist_(dn)

            w = w.to(x.device, dp.dtype) if w is not None else None

            v_agg = torch.zeros((l, nneg, c), device=x.device, dtype=x.dtype)

            for rho in temps:
                # FIX: use raw rho — no sqrt(C) scaling on S^{C-1}
                tau = float(rho)

                if drift_form == "alg2_joint":
                    lp = -dp / tau
                    ln = -dn / tau
                    if coupling != "sinkhorn" or sinkhorn_marginal == "none":
                        _apply_uncond_bias_(ln, nuncond=nuncond, uncond_weight=w)
                    logits = torch.cat([lp, ln], dim=-1)

                    if coupling == "partial_two_sided":
                        log_r  = torch.logsumexp(logits, dim=-1, keepdim=True)
                        log_c_ = torch.logsumexp(logits, dim=-2, keepdim=True)
                        log_a  = logits - 0.5 * (log_r + log_c_)
                        log_a  = log_a.masked_fill(torch.isneginf(logits), -math.inf)
                        A = torch.exp(log_a)
                    elif coupling == "row":
                        A = _row_stochastic_from_logits(logits)
                    elif coupling == "sinkhorn":
                        nx, ny = nneg, int(logits.shape[-1])
                        r   = logits.new_full((nx,), 1.0 / nx)
                        c_m = _col_marginal(ny, nuncond, w, sinkhorn_marginal, logits)
                        A   = _sinkhorn_from_logits(logits, r=r, c=c_m,
                                                    iters=int(sinkhorn_iters))
                        if stats is not None:
                            tag = str(rho).replace(".", "p")
                            stats[f"drift_sinkhorn_row_mae_{tag}"] = float(
                                (A.sum(-1) - r).abs().mean())
                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    Ap = A[..., :npos];  An = A[..., npos:]
                    Wp = Ap * An.sum(-1, keepdim=True)
                    Wn = An * Ap.sum(-1, keepdim=True)
                    v_raw = (_weighted_log_map(Wp, x, yp)
                           - _weighted_log_map(Wn, x, yn))

                elif drift_form == "split":
                    lp = -dp / tau
                    if coupling == "row":
                        Pp = _row_stochastic_from_logits(lp)
                    elif coupling == "partial_two_sided":
                        Ap = _partial_two_sided_from_logits(lp, impl=impl)
                        Pp = Ap / Ap.sum(-1, keepdim=True).clamp_min(1e-12)
                    elif coupling == "sinkhorn":
                        r   = lp.new_full((nneg,), 1.0 / nneg)
                        c_m = lp.new_full((npos,),  1.0 / npos)
                        Pi  = _sinkhorn_from_logits(lp, r=r, c=c_m,
                                                    iters=int(sinkhorn_iters))
                        Pp  = Pi / Pi.sum(-1, keepdim=True).clamp_min(1e-12)
                    else:
                        raise ValueError(f"Unknown coupling: {coupling}")

                    drift_pos = _weighted_log_map(Pp, x, yp)

                    ln = -dn / tau
                    if coupling != "sinkhorn" or sinkhorn_marginal == "none":
                        _apply_uncond_bias_(ln, nuncond=nuncond, uncond_weight=w)

                    if coupling == "row":
                        Pn = _row_stochastic_from_logits(ln)
                    elif coupling == "partial_two_sided":
                        An = _partial_two_sided_from_logits(ln, impl=impl)
                        Pn = An / An.sum(-1, keepdim=True).clamp_min(1e-12)
                    elif coupling == "sinkhorn":
                        ny_n = int(ln.shape[-1])
                        r_n  = ln.new_full((nneg,),   1.0 / nneg)
                        c_m  = _col_marginal(ny_n, nuncond, w, sinkhorn_marginal, ln)
                        Pi_n = _sinkhorn_from_logits(ln, r=r_n, c=c_m,
                                                     iters=int(sinkhorn_iters))
                        Pn   = Pi_n / Pi_n.sum(-1, keepdim=True).clamp_min(1e-12)
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
                    stats[f"drift_theta_{str(rho).replace('.', 'p')}"] = float(theta_norm)
                v_agg = v_agg + (v_raw / theta_norm if normalize_drift_theta else v_raw)

            v_agg_nlc = v_agg.permute(1, 0, 2).contiguous()   # [Nneg, L, C]

            if stats is not None:
                stats["drift_v_rms"] = float(
                    (v_agg_nlc * v_agg_nlc).mean().clamp_min(0).sqrt())
                stats["drift_v_nonfinite_count"] = float(
                    (~torch.isfinite(v_agg_nlc)).sum())
                stats["drift_v_nan_count"] = float(
                    torch.isnan(v_agg_nlc).sum())

        # FIX: loss target — ambient convention (matches original Euclidean loss)
        # target = sg(x + v),  loss = MSE(x_feat, target)
        # Gradient flows through x_feat (from generator), same as the original.
        x_feat_f = x_feat.float()
        if use_expmap_target:
            # Geometrically exact: map drift onto sphere via exp_map.
            # Slightly compressed gradient magnitude but correct on-sphere target.
            target = exp_map(x_feat_f.detach(), v_agg_nlc).detach()
        else:
            # First order taylor expansion.
            target = (x_feat_f.detach() + v_agg_nlc).detach()

        return F.mse_loss(x_feat_f, target, reduction="mean")

    with amp_off:
        return _impl(x_feat, y_pos_feat, y_uncond_feat)


# Drop-in alias
drifting_loss_for_feature_set = drifting_loss_hyperspherical