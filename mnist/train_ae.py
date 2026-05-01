"""Train ConvAE on MNIST: 28x28 -> latent_dim -> 28x28.

Improvements for hyperspherical drifting compatibility:
  1. Latent perturbation: train decoder on slightly perturbed unit vectors
     so it is robust across the sphere, not just near encoder outputs.
  2. Uniform sphere regularisation: push encoder outputs toward a more
     uniform distribution on S^{latent_dim-1} via a von Mises-Fisher
     entropy proxy (maximise pairwise distances within each batch).
  3. Combined MSE + SSIM loss for better perceptual reconstruction quality.
  4. Cosine LR schedule with warmup.

Usage:
    python -m mnist.train_ae --epochs 100 --latent-dim 6 --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mnist.models import ConvAE


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST ConvAE.")
    p.add_argument("--latent-dim",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--data-root",    type=str,   default="./data")
    p.add_argument("--run-root",     type=str,   default="runs/mnist_ae")
    p.add_argument("--save-every",   type=int,   default=10)

    # Sphere-robustness knobs
    p.add_argument("--perturb-std",  type=float, default=0.1,
                   help="Std of noise added to unit latents before decoding. "
                        "Makes decoder robust across the sphere. 0 = disabled.")
    p.add_argument("--uniform-weight", type=float, default=0.01,
                   help="Weight for within-batch uniformity loss on S^{d-1}. "
                        "0 = disabled.")
    p.add_argument("--ssim-weight",  type=float, default=0.5,
                   help="Weight for SSIM loss (0 = pure MSE).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """
    Simplified single-scale SSIM loss (1 - SSIM), averaged over batch.
    x, y: [B, 1, H, W] in [0, 1].
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    # Use average pooling as a local mean estimator.
    k = window_size
    pad = k // 2
    mu_x = F.avg_pool2d(x, k, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, k, stride=1, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sig_x2 = F.avg_pool2d(x * x, k, stride=1, padding=pad) - mu_x2
    sig_y2 = F.avg_pool2d(y * y, k, stride=1, padding=pad) - mu_y2
    sig_xy = F.avg_pool2d(x * y, k, stride=1, padding=pad) - mu_xy
    ssim_map = (
        (2 * mu_xy + C1) * (2 * sig_xy + C2)
        / ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    )
    return 1.0 - ssim_map.mean()


def uniformity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Uniformity loss on S^{d-1}: -log E[exp(-2 ||z_i - z_j||^2)].

    From Wang & Isola (2020). Minimising this spreads points uniformly
    over the sphere, counteracting the encoder collapsing to a small region.
    z: [B, D] unit vectors.
    """
    # Pairwise squared L2 = 2(1 - cosine_sim) for unit vectors.
    cos = torch.matmul(z, z.T)                       # [B, B]
    sq_dist = 2.0 * (1.0 - cos)                      # [B, B]
    # Exclude diagonal (self-pairs).
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return -torch.log(torch.exp(-2.0 * sq_dist[mask]).mean() + 1e-8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, f"{ts}_latent{args.latent_dim}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "logs.jsonl")

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=args.data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(root=args.data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    ae  = ConvAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)

    # Cosine LR schedule with linear warmup (10% of training).
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = total_steps // 10
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n_params = sum(p.numel() for p in ae.parameters())
    print(f"ConvAE: latent_dim={args.latent_dim}, params={n_params:,}")
    print(f"  perturb_std={args.perturb_std}  uniform_weight={args.uniform_weight}"
          f"  ssim_weight={args.ssim_weight}")
    print(f"Run dir: {run_dir}")

    global_step = 0
    for epoch in tqdm(range(1, args.epochs + 1), total=args.epochs):
        ae.train()
        train_mse_sum = train_ssim_sum = train_unif_sum = 0.0
        train_count = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            z    = ae.encode(imgs)          # [B, D]  — already unit vectors

            # --- Sphere perturbation ---
            # Add Gaussian noise then re-normalise → nearby point on sphere.
            # This trains the decoder to be smooth everywhere on S^{D-1},
            # not just at the encoder's output locations.
            if args.perturb_std > 0:
                noise = torch.randn_like(z) * args.perturb_std
                z_dec = F.normalize(z + noise, dim=-1)
            else:
                z_dec = z

            recon = ae.decode(z_dec)

            # --- Reconstruction loss ---
            mse  = F.mse_loss(recon, imgs)
            if args.ssim_weight > 0:
                ss   = ssim_loss(recon, imgs)
                recon_loss = (1.0 - args.ssim_weight) * mse + args.ssim_weight * ss
            else:
                recon_loss = mse
                ss = torch.tensor(0.0)

            # --- Uniformity loss (spread encoder outputs over the sphere) ---
            if args.uniform_weight > 0:
                unif = uniformity_loss(z)
                total = recon_loss + args.uniform_weight * unif
            else:
                unif  = torch.tensor(0.0)
                total = recon_loss

            opt.zero_grad()
            total.backward()
            opt.step()
            scheduler.step()
            global_step += 1

            b = imgs.shape[0]
            train_mse_sum  += mse.item()  * b
            train_ssim_sum += ss.item()   * b
            train_unif_sum += unif.item() * b
            train_count    += b

        train_mse  = train_mse_sum  / train_count
        train_ssim = train_ssim_sum / train_count
        train_unif = train_unif_sum / train_count

        # Eval — no perturbation, measure clean reconstruction.
        ae.eval()
        test_mse_sum = 0.0
        test_count   = 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                recon, _ = ae(imgs)
                test_mse_sum += F.mse_loss(recon, imgs).item() * imgs.shape[0]
                test_count   += imgs.shape[0]
        test_loss = test_mse_sum / test_count

        rec = {
            "epoch": epoch,
            "train_mse":  round(train_mse,  6),
            "train_ssim": round(train_ssim, 6),
            "train_unif": round(train_unif, 6),
            "test_loss":  round(test_loss,  6),
            "lr":         round(scheduler.get_last_lr()[0], 8),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"Epoch {epoch}/{args.epochs}  "
              f"mse={train_mse:.4f}  ssim={train_ssim:.4f}  "
              f"unif={train_unif:.4f}  test={test_loss:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "model":      ae.state_dict(),
                "latent_dim": args.latent_dim,
                "epoch":      epoch,
                "test_loss":  test_loss,
                "args":       vars(args),
            }
            torch.save(ckpt, os.path.join(run_dir, f"ae_epoch{epoch}.pt"))
            if epoch == args.epochs:
                final_path = os.path.join(run_dir, "ae_final.pt")
                torch.save(ckpt, final_path)
                print(f"Saved: {final_path}")

            with torch.no_grad():
                sample_imgs  = next(iter(test_loader))[0][:16].to(device)
                sample_recon, _ = ae(sample_imgs)
                comparison = torch.cat([sample_imgs, sample_recon], dim=0)
                save_image(comparison,
                           os.path.join(run_dir, f"recon_epoch{epoch}.png"),
                           nrow=16)

    print("Done.")


if __name__ == "__main__":
    main()