"""Train MLP generator on MNIST latents with REINFORCE KDE loss.

Replaces drifting_loss_for_feature_set with a KDE-based REINFORCE objective:
    log q̂ estimated via leave-one-out Laplace KDE, used as both reward and
    score-function weight.

Usage:
    python -m mnist.train_reinforce \
        --ae-ckpt runs/mnist_ae/<run>/ae_final.pt \
        --device cuda:0 --run-name mnist_reinforce
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import save_image

from core.models.ema import EMA
from mnist.models import ConvAE, MLPGenerator


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def estimate_kl(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """KDE log-density estimates under Laplace kernel k(x,y) = exp(-‖x-y‖/temp).

    Returns (log_q, log_p) per query point in x.
    """
    N, N_pos, N_neg = x.shape[0], y_pos.shape[0], y_neg.shape[0]

    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    if N == N_neg:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    log_p = torch.logsumexp(-dist_pos / temp, dim=1) - np.log(N_pos)
    log_q = torch.logsumexp(-dist_neg / temp, dim=1) - np.log(N_neg - 1 if N == N_neg else N_neg)

    return log_q, log_p


def reinforce_loss(gen: Tensor, pos: Tensor, temp: float = 0.05) -> Tensor:
    """REINFORCE with KDE reward: advantage * score-function gradient of log q.

    gen: [N, D] generated samples (requires_grad)
    pos: [N_pos, D] real samples (no grad needed)
    """
    with torch.no_grad():
        log_q_val, log_p_val = estimate_kl(gen, pos, gen, temp=temp)
        advantage = (log_q_val - log_p_val)
        # normalized becomes worse; subtract mean becomes worse; divide by std is better
        advantage = advantage * 0 # / (advantage.std() + 1e-6)

    # Score function: query detached, centers live so gradients flow
    x_query = gen.detach()
    x_centers = gen
    dist = torch.cdist(x_query, x_centers)
    dist = dist + torch.eye(len(gen), device=gen.device) * 1e6
    log_q_score = torch.logsumexp(-dist / temp, dim=1) - np.log(len(gen) - 1)

    return (log_q_score * advantage).mean()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST MLP generator with REINFORCE KDE loss.")

    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--noise-dim", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)

    p.add_argument("--temp", type=float, default=0.05,
                   help="Laplace kernel temperature for KDE loss")

    p.add_argument("--nneg", type=int, default=64, help="Generated samples per class per step")
    p.add_argument("--npos", type=int, default=64, help="Real samples per class per step")

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=750)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--ema-decay", type=float, default=0.999)

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--sample-nrow", type=int, default=10)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-root", type=str, default="runs/mnist_reinforce")
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class LatentDataset:
    """In-memory dataset of latents grouped by class."""

    def __init__(self, latents: np.ndarray, labels: np.ndarray, device: torch.device):
        self.device = device
        self.num_classes = int(labels.max()) + 1
        self.latents = torch.from_numpy(latents).float().to(device)
        self.class_indices: list[torch.Tensor] = []
        for c in range(self.num_classes):
            idx = torch.from_numpy(np.where(labels == c)[0]).long().to(device)
            self.class_indices.append(idx)

    def sample_class(self, c: int, n: int) -> torch.Tensor:
        idx = self.class_indices[c]
        sel = idx[torch.randint(len(idx), (n,), device=self.device)]
        return self.latents[sel]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def set_lr(optimizer: torch.optim.Optimizer, step: int, warmup: int, base_lr: float) -> float:
    lr = base_lr * (step + 1) / warmup if (warmup > 0 and step < warmup) else base_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_grid(
    gen: MLPGenerator,
    ae: ConvAE,
    nrow: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a 10-class image grid (omega fixed to 1.0 as a dummy input)."""
    all_imgs = []
    omega_val = torch.ones(nrow, device=device)
    for c in range(gen.num_classes):
        noise = torch.randn(nrow, gen.noise_dim, device=device)
        labels = torch.full((nrow,), c, dtype=torch.long, device=device)
        z = gen(noise, labels, omega_val)
        all_imgs.append(ae.decode(z))
    return torch.cat(all_imgs, dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Load AE ---
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    latent_dim = ae_ckpt["latent_dim"]
    ae = ConvAE(latent_dim=latent_dim).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"AE: latent_dim={latent_dim}, epoch={ae_ckpt['epoch']}, test_loss={ae_ckpt['test_loss']:.6f}")

    # --- Load precomputed latents ---
    ae_dir = os.path.dirname(args.ae_ckpt)
    train_z = np.load(os.path.join(ae_dir, "train_latents.npy"))
    train_y = np.load(os.path.join(ae_dir, "train_labels.npy"))
    dataset = LatentDataset(train_z, train_y, device)
    print(f"Loaded {len(train_z)} train latents, {dataset.num_classes} classes")

    # --- Generator (omega is a dummy input; always passed as 1.0) ---
    gen = MLPGenerator(
        num_classes=dataset.num_classes,
        noise_dim=args.noise_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    ema = EMA(gen, decay=args.ema_decay)
    print(f"MLPGenerator: params={sum(p.numel() for p in gen.parameters()):,}")

    # --- Optimizer ---
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # --- Run directory ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"{ts}_reinforce_temp{str(args.temp).replace('.', 'p')}"
    run_dir = os.path.join(args.run_root, name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "logs.jsonl")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run dir: {run_dir}")

    # --- Training loop ---
    nc = dataset.num_classes
    omega_dummy = torch.ones(args.nneg, device=device)
    gen.train()

    for step in range(1, args.steps + 1):
        lr = set_lr(opt, step - 1, args.warmup_steps, args.lr)
        opt.zero_grad()

        total_loss = 0.0

        for c in range(nc):
            noise = torch.randn(args.nneg, args.noise_dim, device=device)
            c_labels = torch.full((args.nneg,), c, dtype=torch.long, device=device)
            x_lat = gen(noise, c_labels, omega_dummy)          # [nneg, latent_dim]

            pos_lat = dataset.sample_class(c, args.npos)       # [npos, latent_dim]

            loss = reinforce_loss(x_lat, pos_lat, temp=args.temp)
            (loss / nc).backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
        opt.step()
        ema.update(gen)

        # --- Logging ---
        if step % args.log_every == 0 or step == 1:
            avg_loss = total_loss / nc
            rec = {"step": step, "loss": round(avg_loss, 6), "lr": round(lr, 8)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{step}/{args.steps}] loss={avg_loss:.6f} lr={lr:.2e}")

        # --- Sampling ---
        if step % args.sample_every == 0 or step == args.steps:
            gen.eval()
            orig_state = {n: p.clone() for n, p in gen.named_parameters() if p.requires_grad}
            ema.copy_to(gen)

            grid = sample_grid(gen, ae, nrow=args.sample_nrow, device=device)
            fig_path = os.path.join(run_dir, f"sample_step{step}.png")
            save_image(grid, fig_path, nrow=args.sample_nrow, padding=1)
            print(f"  -> saved sample grid: {fig_path}")

            for n, p in gen.named_parameters():
                if n in orig_state:
                    p.data.copy_(orig_state[n])
            gen.train()

        # --- Checkpoint ---
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "model": gen.state_dict(),
                "ema_state": {n: ema.shadow[n].clone() for n in ema.shadow},
                "optimizer": opt.state_dict(),
                "gen_config": {
                    "num_classes": dataset.num_classes,
                    "noise_dim": args.noise_dim,
                    "latent_dim": latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                "args": vars(args),
            }
            tag = f"ckpt_step{step}.pt" if step < args.steps else "ckpt_final.pt"
            torch.save(ckpt, os.path.join(run_dir, tag))
            if step == args.steps:
                print(f"Saved final: {os.path.join(run_dir, tag)}")

    print("Done.")


if __name__ == "__main__":
    main()
