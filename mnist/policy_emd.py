from __future__ import annotations

import argparse
import json
import os

import numpy as np
import ot
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist.policy_gradient import ConvAE, Generator


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST class-level EMD evaluation.")
    p.add_argument("--gen-ckpt", type=str, nargs="+", required=True,
                   help="One or more generator checkpoint paths (state_dict .pt files)")
    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-samples", type=int, default=0,
                   help="Samples per class (0 = use all test data per class)")
    p.add_argument("--noise-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EMD
# ---------------------------------------------------------------------------

def compute_emd(X: np.ndarray, Y: np.ndarray) -> float:
    """Wasserstein-2² between two point clouds. X: [N, D], Y: [M, D]."""
    n, m = len(X), len(Y)
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    C = ot.dist(X, Y, metric="sqeuclidean")
    return float(ot.emd2(a, b, C, numItermax=500_000))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    policy: Generator,
    ae: ConvAE,
    counts_per_class: dict[int, int],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Sample from the policy, decode to images.

    Returns:
        latents_by_class: {c: [N_c, latent_dim]}  raw AE latents
        images_by_class:  {c: [N_c, 784]}
    """
    latents_by_class: dict[int, np.ndarray] = {}
    images_by_class:  dict[int, np.ndarray] = {}

    for c, n_c in counts_per_class.items():
        epsilon, cls = policy.rollout(n_c, device, class_labels=c)
        z = policy(epsilon, cls)                    # [N_c, latent_dim]
        imgs = ae.decode(z)                         # [N_c, 1, 28, 28]

        latents_by_class[c] = z.cpu().numpy()
        images_by_class[c]  = imgs.cpu().view(n_c, -1).numpy()

    return latents_by_class, images_by_class


# ---------------------------------------------------------------------------
# Real samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_real_samples(
    ae: ConvAE,
    *,
    n_per_class: int,
    device: torch.device,
    data_root: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """Encode real test samples into AE latent space.

    Returns:
        latents_by_class: {c: [N_c, latent_dim]}
        images_by_class:  {c: [N_c, 784]}
        counts_by_class:  {c: N_c}
    """
    tf = transforms.ToTensor()
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    all_imgs, all_labels, all_latents = [], [], []
    for imgs, labels in loader:
        z = ae.encode(imgs.to(device))
        all_imgs.append(imgs.view(imgs.shape[0], -1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_latents.append(z.cpu().numpy())

    all_imgs_np    = np.concatenate(all_imgs)
    all_labels_np  = np.concatenate(all_labels)
    all_latents_np = np.concatenate(all_latents)

    latents_by_class: dict[int, np.ndarray] = {}
    images_by_class:  dict[int, np.ndarray] = {}
    counts_by_class:  dict[int, int]        = {}

    for c in range(10):
        idx = np.where(all_labels_np == c)[0]
        if n_per_class > 0:
            rng = np.random.RandomState(123 + c)
            idx = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)

        latents_by_class[c] = all_latents_np[idx]
        images_by_class[c]  = all_imgs_np[idx]
        counts_by_class[c]  = len(idx)

    return latents_by_class, images_by_class, counts_by_class


# ---------------------------------------------------------------------------
# Load policy
# ---------------------------------------------------------------------------

def load_policy(
    ckpt_path: str,
    noise_dim: int,
    latent_dim: int,
    device: torch.device,
) -> Generator:
    state = torch.load(ckpt_path, map_location="cpu")
    # handle both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    policy = Generator(
        num_classes=10,
        noise_dim=noise_dim,
        latent_dim=latent_dim,
    ).to(device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Evaluate one checkpoint
# ---------------------------------------------------------------------------

def evaluate_one(
    gen_ckpt: str,
    ae: ConvAE,
    real_latents: dict[int, np.ndarray],
    real_images:  dict[int, np.ndarray],
    counts_per_class: dict[int, int],
    *,
    noise_dim: int,
    latent_dim: int,
    device: torch.device,
) -> dict:
    policy   = load_policy(gen_ckpt, noise_dim, latent_dim, device)
    run_name = os.path.basename(os.path.dirname(gen_ckpt)) or os.path.basename(gen_ckpt)
    total_n  = sum(counts_per_class.values())

    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name}")
    print(f"  Checkpoint : {gen_ckpt}")
    print(f"  total_samples={total_n}")
    print(f"{'='*60}")

    gen_latents, gen_images = generate_samples(
        policy, ae, counts_per_class, device
    )

    results_per_class: dict[int, dict] = {}
    emd_latent_all, emd_image_all = [], []

    for c in range(10):
        emd_lat = compute_emd(gen_latents[c], real_latents[c])
        emd_img = compute_emd(gen_images[c],  real_images[c])
        emd_latent_all.append(emd_lat)
        emd_image_all.append(emd_img)
        results_per_class[c] = {"emd_latent": emd_lat, "emd_image": emd_img}
        print(f"  Class {c}: EMD_latent={emd_lat:.6f}  EMD_image={emd_img:.6f}")

    avg_latent = float(np.mean(emd_latent_all))
    avg_image  = float(np.mean(emd_image_all))
    print(f"  ---")
    print(f"  Average:  EMD_latent={avg_latent:.6f}  EMD_image={avg_image:.6f}")

    result = {
        "run_name":        run_name,
        "gen_ckpt":        gen_ckpt,
        "n_per_class":     dict(counts_per_class),
        "per_class":       results_per_class,
        "avg_emd_latent":  avg_latent,
        "avg_emd_image":   avg_image,
    }

    out_path = os.path.join(os.path.dirname(gen_ckpt), "emd_results.json")
    with open(out_path, "w") as f:
        json.dump({**result, "per_class": {str(k): v for k, v in results_per_class.items()}}, f, indent=2)
    print(f"  Saved: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(all_results: list[dict]) -> None:
    if len(all_results) < 2:
        return
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'':>8}", end="")
    for r in all_results:
        print(f"  | {r['run_name'][:30]:^30}", end="")
    print()
    print(f"{'Class':>8}", end="")
    for _ in all_results:
        print(f"  | {'EMD_latent':>13} {'EMD_image':>13}", end="")
    print()
    print("-" * (8 + 32 * len(all_results)))
    for c in range(10):
        print(f"{c:>8}", end="")
        for r in all_results:
            lat = r["per_class"][c]["emd_latent"]
            img = r["per_class"][c]["emd_image"]
            print(f"  | {lat:>13.6f} {img:>13.6f}", end="")
        print()
    print("-" * (8 + 32 * len(all_results)))
    print(f"{'Average':>8}", end="")
    for r in all_results:
        print(f"  | {r['avg_emd_latent']:>13.6f} {r['avg_emd_image']:>13.6f}", end="")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load AE
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    ae = ConvAE(latent_dim=args.latent_dim).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"AE loaded: latent_dim={args.latent_dim}")

    # Get real samples once, shared across all generators
    print("Loading real test samples...")
    real_latents, real_images, counts_per_class = get_real_samples(
        ae,
        n_per_class=args.n_samples,
        device=device,
        data_root=args.data_root,
    )
    for c in sorted(counts_per_class):
        print(f"  Class {c}: {counts_per_class[c]} samples")

    # Evaluate each checkpoint
    all_results = []
    for ckpt_path in args.gen_ckpt:
        result = evaluate_one(
            ckpt_path, ae, real_latents, real_images, counts_per_class,
            noise_dim=args.noise_dim,
            latent_dim=args.latent_dim,
            device=device,
        )
        all_results.append(result)

    print_comparison(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
