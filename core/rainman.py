"""Hyperspherical split-drift loss with Sinkhorn coupling."""

from __future__ import annotations

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
    _sinkhorn_from_logits,
    sample_power_law_omega,
)

_EPS = 1e-7
_SCALE_MAX = 1.0 / _EPS
_DEFAULT_SINKHORN_ITERS = 5


def _weighted_log_map(
    P: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return sum_j P[i,j] log_{x_i}(y_j) on the unit sphere."""
    cos = torch.matmul(x, y.transpose(-2, -1)).clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos)
    scale = (theta / torch.sin(theta).clamp_min(_EPS)).clamp_max(_SCALE_MAX)
    direction = y.unsqueeze(-3) - cos.unsqueeze(-1) * x.unsqueeze(-2)
    return (P.unsqueeze(-1) * scale.unsqueeze(-1) * direction).sum(dim=-2)


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exponential map on the unit sphere."""
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cos(v_norm) * x + torch.sin(v_norm) * (v / v_norm)


def _vmf_marginal(
    x: torch.Tensor,
    kappa: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """vMF-weighted empirical marginal around the spherical batch mean."""
    n = x.shape[-2]
    if kappa <= 0:
        return x.new_full(x.shape[:-1], 1.0 / n)

    mu = x.mean(dim=-2)
    mu_norm = mu.norm(dim=-1, keepdim=True)
    valid = mu_norm.squeeze(-1) > eps
    mu = mu / mu_norm.clamp_min(eps)

    weights = torch.softmax(float(kappa) * (x * mu.unsqueeze(-2)).sum(dim=-1), dim=-1)
    uniform = x.new_full(weights.shape, 1.0 / n)
    return torch.where(valid.unsqueeze(-1), weights, uniform)


def _apply_weighted_uncond_cols(
    c: torch.Tensor,
    nuncond: int,
    w: torch.Tensor | None,
) -> torch.Tensor:
    """Apply CFG unconditional weights to the last nuncond column marginals."""
    if nuncond <= 0 or w is None:
        return c

    c = c.clone()
    c[..., -nuncond:] *= w.clamp(min=1e-6).to(c.device, c.dtype)
    return c / c.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _sinkhorn_row_plan(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    iters: int,
    vmf_marginals: bool,
    kappa: float,
    nuncond: int = 0,
    uncond_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sinkhorn transport plan normalized row-wise for barycentric projection."""
    nx = logits.shape[-2]
    ny = logits.shape[-1]

    if vmf_marginals:
        r = _vmf_marginal(x, kappa)
        c = _vmf_marginal(y, kappa)
    else:
        r = logits.new_full((nx,), 1.0 / nx)
        c = logits.new_full((ny,), 1.0 / ny)

    c = _apply_weighted_uncond_cols(c, nuncond, uncond_weight)
    pi = _sinkhorn_from_logits(logits, r=r, c=c, iters=iters)
    return pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)

def _geodesic_logits(x, y, tau):
    cos = torch.matmul(x, y.transpose(-2, -1)).clamp(-1 + _EPS, 1 - _EPS)
    theta = torch.acos(cos)
    return -(theta * theta) / tau

def _signed_frechet_barycenter(
    x,
    y_pos,
    y_neg,
    p_pos,
    p_neg,
    *,
    neg_scale=1.0,
    steps=4,
    step_size=0.5,
):
    z = x

    for _ in range(steps):
        grad_pos = _weighted_log_map(p_pos, z, y_pos)
        grad_neg = _weighted_log_map(p_neg, z, y_neg)

        u = grad_pos - neg_scale * grad_neg

        # tangent cleanup
        u = u - (u * z).sum(dim=-1, keepdim=True) * z

        z = exp_map(z, step_size * u)
        z = F.normalize(z, dim=-1)

    return z

def _log_map_point(x, y):
    cos = (x * y).sum(dim=-1, keepdim=True)
    cos = cos.clamp(-1.0 + _EPS, 1.0 - _EPS)

    theta = torch.acos(cos)

    direction = y - cos * x

    scale = theta / torch.sin(theta).clamp_min(_EPS)

    return scale * direction

def drifting_loss_hyperspherical(
    x_feat: torch.Tensor,
    y_pos_feat: torch.Tensor,
    y_uncond_feat: torch.Tensor,
    *,
    omega: torch.Tensor,
    temps: Iterable[float],
    coupling: Literal["sinkhorn"] = "sinkhorn",
    sinkhorn_iters: int = _DEFAULT_SINKHORN_ITERS,
    sinkhorn_marginal: Literal["weighted_cols"] = "weighted_cols",
    normalize_drift_theta: bool = False,
    use_expmap_target: bool = False,
    vmf_marginals: bool = True,
    stats: dict[str, float] | None = None,
    **kwargs
) -> torch.Tensor:
    """
    Hyperspherical split-drift loss.

    """
    if coupling != "sinkhorn":
        raise ValueError("Only coupling='sinkhorn' is supported.")
    if sinkhorn_marginal != "weighted_cols":
        raise ValueError("Only sinkhorn_marginal='weighted_cols' is supported.")

    amp_off = (
        torch.autocast(device_type="cuda", enabled=False)
        if x_feat.device.type == "cuda"
        else nullcontext()
    )

    with amp_off:
        x_feat = _ensure_nlc(x_feat, "x_feat")
        y_pos_feat = _ensure_nlc(y_pos_feat, "y_pos_feat")
        y_uncond_feat = _ensure_nlc(y_uncond_feat, "y_uncond_feat")

        nneg, layers, channels = x_feat.shape
        npos = y_pos_feat.shape[0]
        nuncond = y_uncond_feat.shape[0]

        if y_pos_feat.shape[1:] != (layers, channels):
            # print(y_pos_feat.shape, layers, channels)
            # input()
            raise ValueError("y_pos_feat shape must match x_feat after batch dim.")
        if y_uncond_feat.shape[1:] != (layers, channels):
            raise ValueError("y_uncond_feat shape must match x_feat after batch dim.")

        x = x_feat.permute(1, 0, 2).contiguous().detach().float()
        yp = y_pos_feat.permute(1, 0, 2).contiguous().detach().float()
        yuc = y_uncond_feat.permute(1, 0, 2).contiguous().detach().float()

        with torch.no_grad():
            omega_s = omega.detach().to(x.device, torch.float32).reshape(())
            yn = torch.cat([x, yuc], dim=1) if nuncond > 0 else x
            uncond_weight = (
                compute_uncond_weight(omega_s, nneg=nneg, nuncond=nuncond)
                .to(x.device, torch.float32)
                if nuncond > 0
                else None
            )
            # indices for masking self-distances on the diagonal
            idx = torch.arange(nneg, device=x.device)

            # denominators for mean geodesic distance (same logic as Euclidean s)
            denom_pos = float(layers * nneg * npos)
            denom_neg = float(layers * nneg * nneg) - float(layers * nneg)  # exclude diagonal self-distances

            v_agg = torch.zeros((layers, nneg, channels), device=x.device, dtype=x.dtype)

            cos_pos = torch.matmul(x, yp.transpose(-2, -1)).clamp(-1 + _EPS, 1 - _EPS)
            geo_pos = torch.acos(cos_pos)               # [L, Nneg, Npos]

            cos_neg = torch.matmul(x, yn.transpose(-2, -1)).clamp(-1 + _EPS, 1 - _EPS)
            geo_neg = torch.acos(cos_neg)               # [L, Nneg, Nneg+Nuncond]

            # mask self
            geo_neg_for_s = geo_neg.clone()
            geo_neg_for_s[..., idx, idx] = 0.0

            mean_geo = (geo_pos.sum() + geo_neg_for_s.sum()) / (denom_pos + denom_neg)

            # normalize tau by mean geodesic distance, same logic as Euclidean s
            s_geo = mean_geo.clamp_min(1e-6)

            for rho in temps:
                tau = float(rho) * s_geo
                kappa = min(1.0 / tau, 10.0) if vmf_marginals else 0.0

                lp = _geodesic_logits(x, yp, tau)
                ln = _geodesic_logits(x, yn, tau)

                # print("logit std across cols per row:", lp.std(dim=-1).mean().item())  # near 0 = all equidistant
                # print("logit std across rows per col:", lp.std(dim=-2).mean().item())

                K = torch.exp(lp)
                # print("kernel min:", K.min().item(), "max:", K.max().item(), "any zero:", (K == 0).any().item())

                # What does the distance matrix look like?
                cos_mat = torch.matmul(x[0], yp[0].T)  # one layer, [Nneg, Npos]
                # print("cosine sim stats:", cos_mat.mean().item(), cos_mat.std().item(), cos_mat.min().item(), cos_mat.max().item())

                p_pos = _sinkhorn_row_plan(
                    lp,
                    x,
                    yp,
                    iters=int(sinkhorn_iters),
                    vmf_marginals=vmf_marginals,
                    kappa=kappa,
                )
                p_neg = _sinkhorn_row_plan(
                    ln,
                    x,
                    yn,
                    iters=int(sinkhorn_iters),
                    vmf_marginals=vmf_marginals,
                    kappa=kappa,
                    nuncond=nuncond,
                    uncond_weight=uncond_weight,
                )

                # # 1. Are features normalized?
                # print("x norms:  ", x.norm(dim=-1).mean().item(), x.norm(dim=-1).min().item(), x.norm(dim=-1).max().item())
                # print("yp norms: ", yp.norm(dim=-1).mean().item(), yp.norm(dim=-1).min().item(), yp.norm(dim=-1).max().item())

                # 2. Are logits sane (not all -inf or NaN)?
                lp = _geodesic_logits(x, yp, tau=0.05)
                # print("logit_pos range:", lp.min().item(), lp.max().item(), "nan?", lp.isnan().any().item())

                # 3. Are plans degenerate (near-permutation = collapsed, near-uniform = dead)?
                # A healthy plan has moderate entropy, not 0 or log(N)
                p = p_pos  # after _sinkhorn_row_plan
                row_entropy = -(p * (p + 1e-12).log()).sum(dim=-1)
                # print("p_pos row entropy: mean", row_entropy.mean().item(), "min", row_entropy.min().item())
                # for N=256 cols, log(256)~5.5 is uniform, 0 is collapsed

                # 4. Does v_raw have per-sample variance?
                v = _weighted_log_map(p_pos, x, yp) - _weighted_log_map(p_neg, x, yn)

                v_pos = _weighted_log_map(p_pos, x, yp)
                v_neg = _weighted_log_map(p_neg, x, yn)

                # print("p_pos has nan:", p_pos.isnan().any().item())
                # print("p_neg has nan:", p_neg.isnan().any().item())
                # print("v_pos has nan:", v_pos.isnan().any().item())
                # print("v_neg has nan:", v_neg.isnan().any().item())

                # Check inside _weighted_log_map manually
                cos = torch.matmul(x, yp.transpose(-2, -1)).clamp(-1.0 + _EPS, 1.0 - _EPS)
                theta = torch.acos(cos)
                sin_theta = torch.sin(theta)
                # print("sin_theta min:", sin_theta.min().item())  # near 0 = acos near 0 or pi = problematic
                scale = (theta / sin_theta.clamp_min(_EPS)).clamp_max(_SCALE_MAX)
                # print("scale max:", scale.max().item(), "any nan:", scale.isnan().any().item())
                

                # input()
                # v_raw = _weighted_log_map(p_pos, x, yp) - _weighted_log_map(p_neg, x, yn)
                # z_star = _signed_frechet_barycenter(
                #     x,
                #     yp,
                #     yn,
                #     p_pos,
                #     p_neg,
                # )

                # v_raw = _log_map_point(x, z_star)
                v_raw = _weighted_log_map(p_pos, x, yp) - _weighted_log_map(p_neg, x, yn)                
                theta_norm = (v_raw * v_raw).mean().clamp_min(1e-12).sqrt()

                # print("v_raw shape:", v_raw.shape)
                # print("v_raw per-sample norms:", v_raw.norm(dim=-1).squeeze())  # [nneg] after squeeze
                # print("v_raw std across batch:", v_raw.std(dim=1).mean().item())  # reduce over N not L
                # print("v_raw per-sample norms:", v_raw.norm(dim=-1).squeeze())
                # print("v_raw mean:", v_raw.mean().item())
                # print("x shape entering loss:", x_feat.shape)  # should be [64, L, C]
                # print("x_feat sample:", x_feat[0])  # what are these 7 values?
                # print("y_pos sample:", y_pos_feat[0])
                if stats is not None:
                    tag = str(rho).replace(".", "p")
                    stats[f"drift_theta_{tag}"] = float(theta_norm)

                v_agg = v_agg + (v_raw / theta_norm if normalize_drift_theta else v_raw)

            # Enforce tangency numerically.
            v_agg = v_agg - (v_agg * x).sum(dim=-1, keepdim=True) * x
            v_agg_nlc = v_agg.permute(1, 0, 2).contiguous()

            if stats is not None:
                stats["drift_c"] = float(channels)
                stats["drift_v_rms"] = float((v_agg_nlc * v_agg_nlc).mean().sqrt())
                stats["drift_v_nonfinite_count"] = float((~torch.isfinite(v_agg_nlc)).sum())
                stats["drift_v_nan_count"] = float(torch.isnan(v_agg_nlc).sum())
                stats["drift_vmf_enabled"] = float(bool(vmf_marginals))

        x_float = x_feat.float()
        target = (
            exp_map(x_float.detach(), v_agg_nlc).detach()
            if use_expmap_target
            else F.normalize(x_float.detach() + v_agg_nlc, dim=-1).detach()
        )

        # target is fixed / detached, but x_float must remain differentiable
        target = target.detach()

        v_target = _log_map_point(x_float, target)
        return (v_target * v_target).sum(dim=-1).mean()


# Drop-in alias
drifting_loss_for_feature_set = drifting_loss_hyperspherical
