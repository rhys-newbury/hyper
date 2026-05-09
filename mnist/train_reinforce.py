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
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from core.models.ema import EMA
from mnist.models import ConvAE
from mnist.eval_emd import compute_emd


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def kde_logp(
    query: Tensor,
    center: Tensor,
    temp: float = 0.05,
    leave_one_out: bool = False,
):
    dist = torch.cdist(query, center)
    if leave_one_out:
        dist = dist + torch.eye(len(query), device=query.device) * 1e6
        log_q = torch.logsumexp(-dist / temp, dim=1) - np.log(len(query) - 1)
    else:
        log_q = torch.logsumexp(-dist / temp, dim=1) - np.log(len(center))
    return log_q


def reinforce_loss(gen: Tensor, pos: Tensor, temp: float = 0.05, maxent_coef=0.0) -> Tensor:
    """REINFORCE with KDE reward: advantage * score-function gradient of log q.

    gen: [N, D] generated samples (requires_grad)
    pos: [N_pos, D] real samples (no grad needed)
    """
    log_q = kde_logp(gen.detach(), gen, temp=temp, leave_one_out=True)  # [N] log q̂ leave-one-out KDE estimate for each gen point
    log_p = kde_logp(gen.detach(), pos, temp=temp, leave_one_out=False)  # [N] log p̂ KDE estimate for each gen point
    
    reward = ((1 + maxent_coef) * log_q - log_p).detach()
    loss  = (log_q * reward).mean()
    return loss

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST MLP generator with REINFORCE KDE loss.")

    p.add_argument("--ae-ckpt", type=str, required=True, help="Path to ae_final.pt")

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

    p.add_argument("--data-root", type=str, default="./data")
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
# Model
# ---------------------------------------------------------------------------


class MLPGenerator(nn.Module):
    """MLP generator for drifting in latent space.

    Interface matches DiTLatentB2: forward(noise, class_labels, omega) -> latent.
    """

    def __init__(
        self,
        num_classes: int = 10,
        noise_dim: int = 6,
        latent_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 3,
        class_emb_dim: int = 32,
        omega_emb_dim: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        # Class conditioning: +1 for unconditional token (index = num_classes)
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)

        # Omega conditioning
        self.omega_mlp = nn.Sequential(
            nn.Linear(1, omega_emb_dim),
            nn.SiLU(),
            nn.Linear(omega_emb_dim, omega_emb_dim),
        )

        # MLP body: [noise + class_emb + omega_emb] -> hidden -> ... -> latent_dim
        in_dim = noise_dim + class_emb_dim + omega_emb_dim
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        noise: torch.Tensor,
        class_labels: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """
        noise:        [B, noise_dim]
        class_labels: [B] int64, 0..num_classes-1 or -1 for unconditional
        omega:        [B] float, CFG scale
        returns:      [B, latent_dim]
        """
        b = noise.shape[0]
        cls = class_labels.clone()
        cls = torch.where(cls < 0, torch.full_like(cls, self.num_classes), cls)
        c_emb = self.class_emb(cls)
        o_emb = self.omega_mlp(omega.view(b, 1))
        inp = torch.cat([noise, c_emb, o_emb], dim=-1)
        return self.mlp(inp)


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

    # --- Encode MNIST train set online ---
    tf = transforms.ToTensor()
    mnist_train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tf)
    loader = DataLoader(mnist_train, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    print("Encoding MNIST train set...")
    all_z, all_y = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_z.append(ae.encode(imgs.to(device)).cpu().numpy())
            all_y.append(labels.numpy())
    train_z = np.concatenate(all_z, axis=0)
    train_y = np.concatenate(all_y, axis=0)
    dataset = LatentDataset(train_z, train_y, device)
    print(f"Encoded {len(train_z)} train latents, {dataset.num_classes} classes")

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

    # --- EMD evaluation ---
    print("\nEvaluating EMD...")
    ema.copy_to(gen)
    gen.eval()
    tf_test = transforms.ToTensor()
    test_ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=tf_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    all_z_real, all_y_real = [], []
    all_imgs_real = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            all_z_real.append(ae.encode(imgs.to(device)).cpu().numpy())
            all_imgs_real.append(imgs.view(imgs.shape[0], -1).numpy())
            all_y_real.append(labels.numpy())
    z_real_all  = np.concatenate(all_z_real,  axis=0)
    imgs_real_all = np.concatenate(all_imgs_real, axis=0)
    y_real_all  = np.concatenate(all_y_real,  axis=0)

    omega_eval = torch.ones(args.nneg, device=device)
    emd_lat_list, emd_img_list = [], []
    for c in range(dataset.num_classes):
        idx = np.where(y_real_all == c)[0]
        z_real_c   = z_real_all[idx]
        imgs_real_c = imgs_real_all[idx]

        with torch.no_grad():
            noise  = torch.randn(len(idx), args.noise_dim, device=device)
            labels = torch.full((len(idx),), c, dtype=torch.long, device=device)
            z_gen  = gen(noise, labels, torch.ones(len(idx), device=device)).cpu().numpy()
            imgs_gen = ae.decode(torch.from_numpy(z_gen).to(device)).cpu().view(len(idx), -1).numpy()

        emd_lat = compute_emd(z_gen, z_real_c)
        emd_img = compute_emd(imgs_gen, imgs_real_c)
        emd_lat_list.append(emd_lat)
        emd_img_list.append(emd_img)
        print(f"  Class {c}: EMD_latent={emd_lat:.6f}  EMD_image={emd_img:.6f}")

    print(f"  Average:  EMD_latent={np.mean(emd_lat_list):.6f}  EMD_image={np.mean(emd_img_list):.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
