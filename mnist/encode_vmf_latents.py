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

from mnist.vmf_vae import ConvModelVAE

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
    ae: ConvModelVAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a full split and return (latents [N, D], labels [N])."""
    all_z, all_y = [], []
    for imgs, labels in loader:
        z, _ = ae.encode(imgs.to(device))
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # Load AE.

    ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    ae = ConvModelVAE(z_dim=16, distribution="vmf").to(device)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    ae.load_state_dict(state)
    ae.eval()
    print(f"Loaded AE: latent_dim={15}")

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

    # Save.
    out_dir = os.path.dirname(args.ae_ckpt)

    for name, s, y in [("train", train_t, train_y), ("test", test_t, test_y)]:
        np.save(os.path.join(out_dir, f"vmf_{name}_latents.npy"), s.numpy().astype(np.float32))
        np.save(os.path.join(out_dir, f"vmf_{name}_labels.npy"),  y)

    print(f"Saved sphere latents (shape {train_t.shape}) and stats to {out_dir}/")


if __name__ == "__main__":
    main()