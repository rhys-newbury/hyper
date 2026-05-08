"""Encode MNIST train/test sets into 6-dim latents using a trained ConvAE,
then project onto the unit sphere S^6 in R^7 via stereographic projection.

Saves per-split:
    {train,test}_latents.npy   -- sphere latents (N, D+1), float32, ||s||=1
    {train,test}_labels.npy    -- integer class labels (N,)

And once (computed from train split only):
    latent_mean.npy            -- (D,) per-dim mean of raw latents
    latent_std.npy             -- (D,) per-dim std  of raw latents

To invert back to raw latent space:
    w = stereo_inverse(s) * std + mean

Usage:
    python -m mnist.encode_latents \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt --device cuda:2
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist.models import ConvAE


# ---------------------------------------------------------------------------
# Stereographic projection
# ---------------------------------------------------------------------------

def stereo_project(w: torch.Tensor) -> torch.Tensor:
    """Stereographic projection R^d -> S^d (unit sphere in R^(d+1))."""
    sq = (w ** 2).sum(dim=-1, keepdim=True)   
    s_body = 2.0 * w / (1.0 + sq)             
    s_last = (sq - 1.0) / (1.0 + sq)          
    return torch.cat([s_body, s_last], dim=-1) 


def stereo_inverse(s: torch.Tensor) -> torch.Tensor:
    """Inverse stereographic projection S^d -> R^d.

    Args:
        s: (N, d+1) points on the unit sphere.

    Returns:
        w: (N, d) standardized latents.
    """
    s_body = s[..., :-1]
    s_last = s[..., -1:].clamp(max=1.0 - 1e-6)  # guard against north pole
    return s_body / (1.0 - s_last)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode MNIST -> sphere latents.")
    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default="./data")
    return p.parse_args()


@torch.no_grad()
def encode_split(
    ae: ConvAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a full split and return (latents [N, D], labels [N])."""
    all_z, all_y = [], []
    for imgs, labels in loader:
        z = ae.encode(imgs.to(device))
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # Load AE.
    ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    latent_dim = ckpt["latent_dim"]
    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    print(f"Loaded AE: latent_dim={latent_dim}, epoch={ckpt['epoch']}, test_loss={ckpt['test_loss']:.6f}")

    # Data.
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=args.data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(root=args.data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Encode raw latents.
    train_z, train_y = encode_split(ae, train_loader, device)
    test_z,  test_y  = encode_split(ae, test_loader,  device)
    print(f"Train: {train_z.shape}, Test: {test_z.shape}")
    print(f"Latent stats (train): mean={train_z.mean():.4f}, std={train_z.std():.4f}, "
          f"min={train_z.min():.4f}, max={train_z.max():.4f}")

    # ------------------------------------------------------------------
    # Standardize (fit on train only) then project to unit sphere.
    # ------------------------------------------------------------------
    train_t = torch.from_numpy(train_z)
    test_t  = torch.from_numpy(test_z)

    mu  = train_t.mean(dim=0)               # (D,)
    std = train_t.std(dim=0).clamp_min(1e-6)# (D,)

    train_norm = (train_t - mu) / std
    test_norm  = (test_t  - mu) / std       # use train stats for test

    train_s = stereo_project(train_norm)    # (N, D+1)
    test_s  = stereo_project(test_norm)     # (N, D+1)

    # Sanity checks.
    train_norms = train_s.norm(dim=-1)
    print(f"[stereo] sphere dim: {train_s.shape[1]}  (latent_dim + 1 = {latent_dim} + 1)")
    print(f"[stereo] train norm  mean={train_norms.mean():.6f}  "
          f"min={train_norms.min():.6f}  max={train_norms.max():.6f}  (should all be 1.0)")
    roundtrip_err = (stereo_inverse(train_s) - train_norm).abs().max().item()
    print(f"[stereo] max roundtrip error: {roundtrip_err:.2e}  (should be ~1e-6)")

    # Save.
    out_dir = os.path.dirname(args.ae_ckpt)

    np.save(os.path.join(out_dir, "latent_mean.npy"), mu.numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "latent_std.npy"),  std.numpy().astype(np.float32))

    for name, s, y in [("train", train_s, train_y), ("test", test_s, test_y)]:
        np.save(os.path.join(out_dir, f"{name}_latents.npy"), s.numpy().astype(np.float32))
        np.save(os.path.join(out_dir, f"{name}_labels.npy"),  y)

    print(f"Saved sphere latents (shape {train_s.shape}) and stats to {out_dir}/")


if __name__ == "__main__":
    main()