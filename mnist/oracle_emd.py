"""Evaluate MNIST autoencoder reconstruction drift: class-level EMD in image space.

For each class c in {0,...,9}:
    1. Load original MNIST test images for class c.
    2. Reconstruct those same test images with ConvAE.
    3. Compute EMD / optimal transport cost between:
           original test images      x_test[c]    in R^784
           reconstructed test images x_recon[c]   in R^784

This script does no generator sampling and no latent-space comparison.
It compares test vs reconstructed-test in original pixel/image space only.

Usage:
    python -m mnist.eval_emd_test_vs_reconstructed_test \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --device cuda:0

Optional subsampling:
    python -m mnist.eval_emd_test_vs_reconstructed_test \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --n-samples 500 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import ot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Autoencoder model
# ---------------------------------------------------------------------------

class ConvAE(nn.Module):
    """Convolutional autoencoder: MNIST 1x28x28 <-> latent_dim."""

    def __init__(self, latent_dim: int = 6):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MNIST class-level EMD: original test images vs AE reconstructions."
    )
    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-samples", type=int, default=0,
                   help="Samples per class (0 = use all test samples per class)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--download", action="store_true",
                   help="Download MNIST if it is not present at --data-root")
    p.add_argument("--out", type=str, default=None,
                   help="Optional output JSON path. Default: <ae_dir>/emd_test_vs_recon.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EMD
# ---------------------------------------------------------------------------

def compute_emd(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute squared 2-Wasserstein / EMD cost between two point clouds.

    X: [N, D], Y: [M, D]
    Returns: W_2^2 value using squared Euclidean ground cost.
    """
    n, m = len(X), len(Y)
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    C = ot.dist(X, Y, metric="sqeuclidean")
    return float(ot.emd2(a, b, C, numItermax=500000))


# ---------------------------------------------------------------------------
# Data + reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_test_and_recon_by_class(
    ae: ConvAE,
    *,
    n_per_class: int,
    device: torch.device,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    seed: int,
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int]]:
    """Return original test images and reconstructed test images by class.

    Returns:
        test_images_by_class:  {c: [N_c, 784]}
        recon_images_by_class: {c: [N_c, 784]}
        counts_by_class:       {c: N_c}
    """
    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=data_root, train=False, download=download, transform=tf)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_test, all_recon, all_labels = [], [], []
    for imgs, labels in loader:
        imgs_device = imgs.to(device, non_blocking=True)
        recon, _ = ae(imgs_device)

        all_test.append(imgs.view(imgs.shape[0], -1).cpu().numpy())
        all_recon.append(recon.detach().cpu().view(recon.shape[0], -1).numpy())
        all_labels.append(labels.cpu().numpy())

    test_np = np.concatenate(all_test, axis=0)
    recon_np = np.concatenate(all_recon, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    num_classes = int(labels_np.max()) + 1
    test_by_class: Dict[int, np.ndarray] = {}
    recon_by_class: Dict[int, np.ndarray] = {}
    counts_by_class: Dict[int, int] = {}

    for c in range(num_classes):
        idx = np.where(labels_np == c)[0]
        if n_per_class > 0:
            rng = np.random.RandomState(seed + c)
            idx = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)

        test_by_class[c] = test_np[idx]
        recon_by_class[c] = recon_np[idx]
        counts_by_class[c] = len(idx)

    return test_by_class, recon_by_class, counts_by_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=True)
    latent_dim = int(ae_ckpt["latent_dim"])

    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()

    print(f"AE checkpoint: {args.ae_ckpt}")
    print(f"AE latent_dim: {latent_dim}")
    print(f"Device: {device}")

    print("Loading MNIST test set and reconstructing test images...")
    test_images, recon_images, counts_per_class = get_test_and_recon_by_class(
        ae,
        n_per_class=args.n_samples,
        device=device,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        seed=args.seed,
    )

    results_per_class = {}
    emd_all = []

    print("\nComputing EMD: original test vs reconstructed test")
    print(f"{'=' * 60}")
    for c in sorted(counts_per_class):
        emd_img = compute_emd(test_images[c], recon_images[c])
        emd_all.append(emd_img)
        results_per_class[c] = {"emd_image": emd_img, "n": counts_per_class[c]}
        print(f"  Class {c}: n={counts_per_class[c]:4d}  EMD_image={emd_img:.6f}")

    avg_emd = float(np.mean(emd_all))
    print("  ---")
    print(f"  Average: EMD_image={avg_emd:.6f}")

    result = {
        "comparison": "mnist_test_vs_ae_reconstructed_test",
        "ae_ckpt": args.ae_ckpt,
        "latent_dim": latent_dim,
        "n_per_class": dict(counts_per_class),
        "ground_cost": "sqeuclidean_pixels_784d",
        "metric": "emd2 / squared 2-Wasserstein cost",
        "per_class": results_per_class,
        "avg_emd_image": avg_emd,
    }

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.ae_ckpt), "emd_test_vs_recon.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
