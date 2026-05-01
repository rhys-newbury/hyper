"""
Drifting loss for the exact hyperspherical setting.

Assumes all feature vectors already lie on S^{C-1} — i.e. the encoder
hard-normalises outputs (``F.normalize``) and the generator ends with
``F.normalize``.  Under this assumption several Euclidean heuristics
become unnecessary and are removed:

Removed vs. the Euclidean version
----------------------------------
* **Internal re-normalisation** — inputs are already unit vectors.
* **Feature-scale normalisation S_j** (Eq. 20-21) — geodesic distances
  are bounded in [0, π] regardless of batch or dimension; no mean-rescaling
  is needed to keep the kernel in a useful regime.
* **Temperature dimension-scaling** ``ρ̃ = ρ·√C`` (Eq. 22) — that factor
  compensated for Euclidean distances growing as √C in high dimensions.
  Geodesic distances on S^{C-1} are scale-invariant, so raw ρ is used.
* **``dist_metric`` choice** — there is exactly one Riemannian distance on
  a sphere (the great-circle / geodesic distance); the option is a no-op.

Kept / adapted
--------------
* All coupling modes: ``row``, ``partial_two_sided``, ``sinkhorn``.
* All drift forms: ``alg2_joint``, ``split``.
* CFG unconditional weighting (Appendix A.7).
* Drift normalisation θ_j (Eq. 24-25), applied to tangent vectors.
* Stop-gradient / AMP policy (float32, no autocast).

Riemannian primitives
----------------------
* **Geodesic distance**: d(x, y) = arccos(x·y)   ∈ [0, π]
* **Log map** (tangent displacement):
      log_x(y) = θ · (y − cosθ·x) / sinθ,   θ = arccos(x·y)
      stable limit as θ → 0: log_x(y) ≈ (y − x)
* **Exp map** (move along tangent vector):
      exp_x(v) = cos(‖v‖)·x + sin(‖v‖)·v/‖v‖
* **Loss**: MSE(x, sg(exp_x(Ṽ_agg)))

Tensor shapes
-------------
All inputs use [N, L, C]:
  N : samples,  L : feature vectors per sample,  C : feature dimension.
"""

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


# ---------------------------------------------------------------------------
# Riemannian primitives  (inputs assumed to be unit vectors)
# ---------------------------------------------------------------------------

def _pairwise_geodesic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise geodesic distances for unit vectors.

    Args:
        x: [..., N, C]
        y: [..., M, C]
    Returns:
        [..., N, M]  in [0, π]
    """
    cos = torch.matmul(x, y.transpose(-2, -1))
    return torch.acos(cos.clamp(-1.0 + _EPS, 1.0 - _EPS))


def _weighted_log_map(
    P: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Riemannian barycentric projection: Σ_j P[i,j] · log_{x_i}(y_j).

    log_x(y) = (θ / sinθ) · (y − cosθ·x),   θ = arccos(x·y)

    The scale factor θ/sinθ → 1 as θ → 0, so no special-case branching
    is needed — we just guard sinθ from below.

    Args:
        P: [..., N, M]  coupling weights
        x: [..., N, C]  base points (unit vectors)
        y: [..., M, C]  target points (unit vectors)
    Returns:
        [..., N, C]  weighted tangent vectors
    """
    cos   = torch.matmul(x, y.transpose(-2, -1))          # [..., N, M]
    cos   = cos.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos)                                # [..., N, M]
    scale = theta / torch.sin(theta).clamp_min(_EPS)       # [..., N, M]  → 1 as θ→0

    # direction[i,j,:] = y_j - cos[i,j] * x_i
    direction = (
        y.unsqueeze(-3)                                    # [...,  1, M, C]
        - cos.unsqueeze(-1) * x.unsqueeze(-2)              # [..., N, M, C]
    )

    log_vecs = scale.unsqueeze(-1) * direction             # [..., N, M, C]
    return (P.unsqueeze(-1) * log_vecs).sum(dim=-2)        # [..., N, C]


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on S^{C-1}: exp_x(v) = cos(‖v‖)·x + sin(‖v‖)·v̂.

    Args:
        x: [..., C]  base point (unit vector)
        v: [..., C]  tangent vector at x
    Returns:
        [..., C]  point on S^{C-1}
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
    vanilla: bool = False,        # no-op: no S_j scaling on S^{C-1}
    drift_form: DriftForm = "alg2_joint",
    coupling: Coupling = "partial_two_sided",
    sinkhorn_iters: int = 20,
    sinkhorn_marginal: SinkhornMarginal = "none",
    sinkhorn_agg_kernel: bool = False,
    mask_self_neg: bool = True,
    dist_metric: str = "geodesic",  # no-op: always geodesic
    normalize_drift_theta: bool = True,
    drift_tau_scale: bool = False,  # no-op: no τ dimension-scaling on S^{C-1}
    drift_unit_vec: bool = False,   # no-op: log-map already gives geodesic direction
    stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Drifting loss for features exactly on S^{C-1}.

    Inputs must be L2-normalised unit vectors. The generator must end with
    ``F.normalize`` and the encoder must hard-normalise (e.g. ConvAE.encode).

    The drift field is computed in the tangent space via the log map and the
    loss target is placed back on the sphere via the exp map:

        loss = MSE(x, sg(exp_x(Ṽ_agg)))

    All coupling / drift-form / CFG options are supported.
    ``vanilla``, ``dist_metric``, ``drift_tau_scale``, ``drift_unit_vec``
    are accepted for API compatibility but have no effect.
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

        # [L, N, C] — batch over spatial locations
        x  = x_feat.permute(1, 0, 2).contiguous().detach().float()       # [L, Nneg, C]
        yp = y_pos_feat.permute(1, 0, 2).contiguous().detach().float()    # [L, Npos, C]
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
                        "coupling='sinkhorn' + sinkhorn_marginal='none' + w≤0 "
                        "is infeasible. Use sinkhorn_marginal='weighted_cols'."
                    )
            else:
                yn = x
                w  = None

            # Geodesic distance matrices — bounded in [0,π], no S_j scaling
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
                # Use rho directly — no √C scaling needed on S^{C-1}
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

                # Drift normalisation θ_j (Eq. 24-25)
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

        # Loss: MSE(x, sg(exp_x(Ṽ)))
        # x_feat has grad (from generator); target is stop-grad point on sphere.
        target = exp_map(x_feat.detach().float(), v_agg_nlc).detach()
        return F.mse_loss(x_feat.float(), target, reduction="mean")

    with amp_off:
        return _impl(x_feat, y_pos_feat, y_uncond_feat)


# Drop-in alias
drifting_loss_for_feature_set = drifting_loss_hyperspherical
