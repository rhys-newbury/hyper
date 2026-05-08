"""Compare sphere projection methods on real MNIST AE latents.

For each projection method, encodes the MNIST train/test latents and
reports geodesic distance statistics, kernel discriminability, and
invertibility error — the three things that matter for drifting loss quality.

Usage:
    python compare_projections.py \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --device cuda:0 \
        [--save-best quantile]   # optionally re-encode and save with best method

Output:
    Prints a ranked table. Optionally overwrites {train,test}_latents.npy
    with the best projection.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Lazy import of ConvAE — works whether run as script or from repo root
# ---------------------------------------------------------------------------
def _load_ae(ckpt_path: str, device: torch.device):
    try:
        from mnist.models import ConvAE
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(ckpt_path)))
        from mnist.models import ConvAE  # type: ignore

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    latent_dim = ckpt["latent_dim"]
    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"AE loaded: latent_dim={latent_dim}, test_loss={ckpt['test_loss']:.6f}")
    return ae, latent_dim


# ---------------------------------------------------------------------------
# Projection methods  (all torch, operate on float32 tensors)
# Each method is a pair:  project(w) -> s,   inverse(s) -> w
# w: (N, D)  standardized latents
# s: (N, D+1) or (N, D) unit vectors depending on method
# ---------------------------------------------------------------------------

_EPS = 1e-7

@dataclass
class ProjectionMethod:
    name: str
    project: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]
    out_dim_delta: int = 1   # +1 for methods that embed R^D -> S^D (D+1 output)
                              #  0 for methods that embed R^D -> S^(D-1) (D output, lossy)
    fit_data: torch.Tensor | None = field(default=None, repr=False)  # for quantile method


# --- 1. Stereographic (current baseline) ---
def _stereo_project(w: torch.Tensor) -> torch.Tensor:
    sq = (w ** 2).sum(dim=-1, keepdim=True)
    return torch.cat([2.0 * w / (1.0 + sq), (sq - 1.0) / (1.0 + sq)], dim=-1)

def _stereo_inverse(s: torch.Tensor) -> torch.Tensor:
    s_body = s[..., :-1]
    s_last = s[..., -1:].clamp(max=1.0 - _EPS)
    return s_body / (1.0 - s_last)

# --- 2. Norm-preserving via arctan ---
# Encodes direction + norm via polar angle theta = arctan(||w||) in [0, pi/2)
# Bijection R^D -> upper hemisphere of S^D
def _normpreserving_project(w: torch.Tensor) -> torch.Tensor:
    r = w.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    direction = w / r
    theta = torch.atan(r)                          # (N, 1) in [0, pi/2)
    return torch.cat([torch.sin(theta) * direction, torch.cos(theta)], dim=-1)

def _normpreserving_inverse(s: torch.Tensor) -> torch.Tensor:
    s_body = s[..., :-1]
    s_last = s[..., -1:].clamp(_EPS, 1.0 - _EPS)
    theta = torch.acos(s_last)                     # (N, 1) in [0, pi/2)
    r = torch.tan(theta)
    direction = F.normalize(s_body, dim=-1)
    return r * direction

# --- 3. Sigmoid radial: theta = pi * r / (1 + r) in [0, pi) ---
# More aggressive compression for large norms (like ALAE W-space norm ~22)
def _sigmoid_project(w: torch.Tensor) -> torch.Tensor:
    r = w.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    direction = w / r
    theta = torch.pi * r / (1.0 + r)              # (N, 1) in [0, pi)
    return torch.cat([torch.sin(theta) * direction, torch.cos(theta)], dim=-1)

def _sigmoid_inverse(s: torch.Tensor) -> torch.Tensor:
    s_body = s[..., :-1]
    s_last = s[..., -1:].clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(s_last)                     # (N, 1) in [0, pi)
    r = theta / (torch.pi - theta).clamp_min(_EPS)
    direction = F.normalize(s_body, dim=-1)
    return r * direction

# --- 4. Log-radial: theta = 2 * arctan(log(1+r) / pi) ---
# Designed for log-normally distributed norms
def _logradial_project(w: torch.Tensor) -> torch.Tensor:
    r = w.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    direction = w / r
    theta = 2.0 * torch.atan(torch.log1p(r) / torch.pi)
    return torch.cat([torch.sin(theta) * direction, torch.cos(theta)], dim=-1)

def _logradial_inverse(s: torch.Tensor) -> torch.Tensor:
    s_body = s[..., :-1]
    s_last = s[..., -1:].clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(s_last)
    log1pr = torch.tan(theta / 2.0) * torch.pi
    r = torch.expm1(log1pr.clamp_min(0))
    direction = F.normalize(s_body, dim=-1)
    return r * direction

# --- 5. Quantile-normalized radial ---
# Maps row norms through their empirical CDF -> uniform -> arccos(1-2u)
# Gives cos(theta) ~ Uniform[-1, 1], which maximises geodesic distance variance.
# Requires fitting on training data; stored in fit_data.
class QuantileProjection:
    """Fit on training norms, then project/inverse via empirical CDF."""

    def __init__(self, sorted_train_norms: torch.Tensor):
        self.sorted_norms = sorted_train_norms  # (N,)

    def project(self, w: torch.Tensor) -> torch.Tensor:
        r = w.norm(dim=-1)                          # (N,)
        direction = w / r.unsqueeze(-1).clamp_min(_EPS)
        # Empirical CDF: interp r into [0, 1]
        u = self._ecdf(r)
        u = u.clamp(_EPS, 1.0 - _EPS)
        theta = torch.acos(1.0 - 2.0 * u)          # cos(theta) ~ Uniform[-1,1]
        return torch.cat([
            torch.sin(theta).unsqueeze(-1) * direction,
            torch.cos(theta).unsqueeze(-1),
        ], dim=-1)

    def inverse(self, s: torch.Tensor) -> torch.Tensor:
        s_body = s[..., :-1]
        s_last = s[..., -1:].clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta = torch.acos(s_last).squeeze(-1)
        u = (1.0 - torch.cos(theta)) / 2.0
        u = u.clamp(0.0, 1.0)
        r = self._icdf(u)
        direction = F.normalize(s_body, dim=-1)
        return r.unsqueeze(-1) * direction

    def _ecdf(self, r: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate r through sorted training norms -> [0, 1]."""
        sn = self.sorted_norms.to(r.device)
        n = len(sn)
        u_grid = torch.linspace(0.0, 1.0, n, device=r.device)
        # torch.searchsorted for vectorised lookup
        idx = torch.searchsorted(sn, r.contiguous()).clamp(1, n - 1)
        lo = sn[idx - 1];  hi = sn[idx]
        t = ((r - lo) / (hi - lo).clamp_min(_EPS)).clamp(0.0, 1.0)
        return u_grid[idx - 1] * (1 - t) + u_grid[idx] * t

    def _icdf(self, u: torch.Tensor) -> torch.Tensor:
        sn = self.sorted_norms.to(u.device)
        n = len(sn)
        u_grid = torch.linspace(0.0, 1.0, n, device=u.device)
        idx = torch.searchsorted(u_grid.contiguous(), u.contiguous()).clamp(1, n - 1)
        lo_u = u_grid[idx - 1];  hi_u = u_grid[idx]
        t = ((u - lo_u) / (hi_u - lo_u).clamp_min(_EPS)).clamp(0.0, 1.0)
        return sn[idx - 1] * (1 - t) + sn[idx] * t


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_projection(
    s: torch.Tensor,
    w_orig: torch.Tensor,
    inverse_fn: Callable,
    n_sample: int = 500,
    taus: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0),
) -> dict:
    """
    Returns a dict of metrics:
      - geodesic_{mean,std,min,max,cv}: pairwise geodesic distance stats
      - roundtrip_max_err: max |inverse(project(w)) - w|
      - kernel_cv_{tau}: coeff of variation of Gibbs kernel exp(-d/tau)
                         higher = more discriminative coupling at that tau
      - tau_recommended: tau closest to std of geodesic distances (good starting point)
    """
    # Unit norm check
    norms = s.norm(dim=-1)
    assert (norms - 1.0).abs().max() < 1e-3, f"Not unit norm: {norms.mean():.4f}"

    # Pairwise geodesic distances on subsample
    sub = s[:n_sample]
    cos = torch.mm(sub, sub.T).clamp(-1.0 + _EPS, 1.0 - _EPS)
    mask = ~torch.eye(n_sample, dtype=torch.bool, device=sub.device)
    d = torch.acos(cos)[mask]

    # Roundtrip
    w_rec = inverse_fn(s)
    roundtrip = (w_rec - w_orig).abs().max().item()

    # Kernel discriminability at each tau
    kernel_cv = {}
    for tau in taus:
        k = torch.exp(-d / tau)
        kernel_cv[tau] = (k.std() / k.mean().clamp_min(_EPS)).item()

    # Recommended tau ~ std of distances (kernel most sensitive here)
    tau_rec = d.std().item()

    return {
        "geo_mean":  d.mean().item(),
        "geo_std":   d.std().item(),
        "geo_min":   d.min().item(),
        "geo_max":   d.max().item(),
        "geo_cv":    (d.std() / d.mean()).item(),
        "roundtrip": roundtrip,
        "kernel_cv": kernel_cv,
        "tau_recommended": tau_rec,
    }


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_split(ae, loader, device):
    all_z, all_y = [], []
    for imgs, labels in loader:
        z = ae.encode(imgs.to(device))
        all_z.append(z.cpu())
        all_y.append(labels)
    return torch.cat(all_z), torch.cat(all_y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-ckpt", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-sample", type=int, default=500,
                   help="Subsample size for pairwise geodesic computation")
    p.add_argument("--save-best", type=str, default=None,
                   choices=["stereo", "normpreserving", "sigmoid",
                            "logradial", "quantile"],
                   help="Re-save {train,test}_latents.npy with this projection")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ae, latent_dim = _load_ae(args.ae_ckpt, device)

    # --- Encode raw latents ---
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(args.data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(args.data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, 512, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  512, shuffle=False, num_workers=4, pin_memory=True)

    print("Encoding train split...")
    train_z, train_y = encode_split(ae, train_loader, device)
    print("Encoding test split...")
    test_z, test_y   = encode_split(ae, test_loader,  device)

    # --- Standardize (fit on train) ---
    mu  = train_z.mean(dim=0)
    std = train_z.std(dim=0).clamp_min(1e-6)
    train_w = (train_z - mu) / std
    test_w  = (test_z  - mu) / std

    print(f"\nWhitened train norms: mean={train_w.norm(dim=-1).mean():.3f}  "
          f"std={train_w.norm(dim=-1).std():.3f}")

    # --- Build projection methods ---
    # Quantile needs fitting on train norms first
    train_norms = train_w.norm(dim=-1)
    sorted_train_norms, _ = train_norms.sort()
    qp = QuantileProjection(sorted_train_norms)

    methods: list[tuple[str, Callable, Callable]] = [
        ("stereo",          _stereo_project,         _stereo_inverse),
        ("normpreserving",  _normpreserving_project,  _normpreserving_inverse),
        ("sigmoid",         _sigmoid_project,         _sigmoid_inverse),
        ("logradial",       _logradial_project,       _logradial_inverse),
        ("quantile",        qp.project,               qp.inverse),
    ]

    # --- Evaluate each method ---
    print("\n" + "=" * 90)
    print(f"{'Method':<20} {'geo_mean':>9} {'geo_std':>9} {'geo_cv':>8} "
          f"{'roundtrip':>12} {'tau_rec':>9}  kernel_cv@tau")
    print(f"{'':20} {'':9} {'':9} {'':8} {'':12} {'':9}  "
          + "  ".join(f"{t}" for t in [0.05, 0.1, 0.2, 0.5]))
    print("-" * 90)

    all_results = {}
    all_projected: dict[str, torch.Tensor] = {}

    taus = (0.05, 0.1, 0.2, 0.5, 1.0)

    for key, proj_fn, inv_fn in methods:
        s_train = proj_fn(train_w)
        metrics = eval_projection(
            s_train, train_w, inv_fn,
            n_sample=min(args.n_sample, len(train_w)),
            taus=taus,
        )
        all_results[key] = metrics
        all_projected[key] = s_train

        kv = "  ".join(f"{metrics['kernel_cv'][t]:.4f}" for t in [0.05, 0.1, 0.2, 0.5])
        print(
            f"{key:<20} "
            f"{metrics['geo_mean']:>9.4f} "
            f"{metrics['geo_std']:>9.4f} "
            f"{metrics['geo_cv']:>8.4f} "
            f"{metrics['roundtrip']:>12.2e} "
            f"{metrics['tau_recommended']:>9.4f}  "
            f"{kv}"
        )

    print("=" * 90)

    # --- Rank by geo_cv (higher = better spread = better coupling) ---
    ranked = sorted(all_results.items(), key=lambda x: x[1]["geo_cv"], reverse=True)
    print("\nRanking by geodesic CV (higher = better kernel discriminability):")
    for i, (key, m) in enumerate(ranked):
        star = " ← BEST" if i == 0 else ""
        print(f"  {i+1}. {key:<20} cv={m['geo_cv']:.4f}  "
              f"tau_recommended={m['tau_recommended']:.4f}{star}")

    best_key = ranked[0][0]
    best = all_results[best_key]
    print(f"\nBest method: {best_key}")
    print(f"  Geodesic distances: mean={best['geo_mean']:.4f}  "
          f"std={best['geo_std']:.4f}  range=[{best['geo_min']:.4f}, {best['geo_max']:.4f}]")
    print(f"  Recommended --temps value: {best['tau_recommended']:.4f}")
    print(f"  Roundtrip error: {best['roundtrip']:.2e}")

    # --- Optionally save ---
    save_key = args.save_best or best_key
    if args.save_best is not None or True:  # always print the save command
        print(f"\nTo re-encode with '{save_key}', re-run with:  --save-best {save_key}")

    if args.save_best is not None:
        proj_fn = dict(methods)[save_key][0] if save_key != "quantile" else qp.project
        inv_fn  = dict(methods)[save_key][1] if save_key != "quantile" else qp.inverse

        s_train = dict(zip([k for k,_,_ in methods],
                           [proj_fn(train_w) for proj_fn, _, _ in
                            [(p, i, None) for _, p, i in methods]]))[save_key]
        # rebuild cleanly
        proj_map = {k: (p, i) for k, p, i in methods}
        proj_fn, inv_fn = proj_map[save_key]
        s_train = proj_fn(train_w)
        s_test  = proj_fn(test_w)

        # Sanity
        err = (inv_fn(s_train) - train_w).abs().max().item()
        print(f"Roundtrip error on train: {err:.2e}")
        assert s_train.norm(dim=-1).sub(1).abs().max() < 1e-3

        out_dir = os.path.dirname(args.ae_ckpt)
        np.save(os.path.join(out_dir, "train_latents.npy"),
                s_train.numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "test_latents.npy"),
                s_test.numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "train_labels.npy"), train_y.numpy())
        np.save(os.path.join(out_dir, "test_labels.npy"),  test_y.numpy())
        np.save(os.path.join(out_dir, "latent_mean.npy"),  mu.numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "latent_std.npy"),   std.numpy().astype(np.float32))

        # Save quantile stats if needed for inversion at decode time
        if save_key == "quantile":
            np.save(os.path.join(out_dir, "quantile_sorted_norms.npy"),
                    sorted_train_norms.numpy().astype(np.float32))
            print("Saved quantile_sorted_norms.npy (needed for decode inversion)")

        print(f"Saved {save_key} latents -> {out_dir}/")
        print(f"  train: {s_train.shape}  test: {s_test.shape}")
        print(f"  Sphere dim: {s_train.shape[1]}  (latent_dim + 1 = {latent_dim} + 1)")


if __name__ == "__main__":
    main()