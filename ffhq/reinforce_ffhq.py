"""train_reinforce_ffhq.py
-------------------------
https://drive.google.com/drive/folders/1qJ6IytgWxORKbYRyhs1z7cd1G9XWqrb7
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import ot
except ImportError:
    ot = None

try:
    from sklearn.decomposition import PCA as SklearnPCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "male_children", "male_adult",    "male_old",
    "female_children","female_adult", "female_old",
]

CLASS_COLORS = [
    "#3B82F6", "#1D4ED8", "#1E3A5F",
    "#F472B6", "#EC4899", "#9D174D",
]

LATENT_DIM = 512
N_CLASSES  = len(CLASS_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FFHQ conditional MLP generator with REINFORCE KDE loss."
    )

    # I/O
    p.add_argument("--train-npz", type=str,
                   default="ffhq_latents_6class/train_latents_by_class.npz")
    p.add_argument("--test-npz",  type=str,
                   default="ffhq_latents_6class/test_latents_by_class.npz")
    p.add_argument("--run-root",  type=str, default="runs/ffhq_reinforce")
    p.add_argument("--run-name",  type=str, default=None)

    # Architecture  (mirrors drift_ffhq.py defaults)
    p.add_argument("--d-z",      type=int, default=512,
                   help="Noise / source dimension.")
    p.add_argument("--d-e",      type=int, default=64,
                   help="Learned class-embedding dimension.")
    p.add_argument("--hidden",   type=int, default=1024,
                   help="Hidden layer width.")
    p.add_argument("--n-hidden", type=int, default=3,
                   help="Number of hidden layers.")

    # Training
    p.add_argument("--steps",        type=int,   default=5000)
    p.add_argument("--lr",           type=float, default=3e-4,
                   help="MLP learning rate.")
    p.add_argument("--emb-lr",       type=float, default=1e-3,
                   help="Embedding table learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int,   default=500)
    p.add_argument("--grad-clip",    type=float, default=5.0)
    p.add_argument("--ema-decay",    type=float, default=0.999)

    # REINFORCE KDE loss
    p.add_argument("--temp",         type=float, default=0.05,
                   help="Laplace kernel temperature for KDE loss.")
    p.add_argument("--maxent-coef",  type=float, default=0.0,
                   help="Entropy bonus coefficient (>0 encourages spread).")
    p.add_argument("--nneg",         type=int,   default=128,
                   help="Generated samples per class per step.")
    p.add_argument("--npos",         type=int,   default=128,
                   help="Real samples per class per step.")

    # Evaluation
    p.add_argument("--emd-every",    type=int, default=500)
    p.add_argument("--emd-samples",  type=int, default=256,
                   help="Samples per class for EMD evaluation.")
    p.add_argument("--log-every",    type=int, default=50)
    p.add_argument("--save-every",   type=int, default=1000)
    p.add_argument("--pca-plot",     type=str, default="reinforce_ffhq_pca.png")

    # Misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed",   type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_latents(npz_path: str) -> Dict[str, torch.Tensor]:
    data = np.load(npz_path, allow_pickle=True)
    return {name: torch.from_numpy(data[name].astype(np.float32))
            for name in CLASS_NAMES}


class LatentDataset:
    """Per-class latent store with random sampling."""

    def __init__(self, npz_path: str, device: torch.device):
        self.device  = device
        raw          = load_latents(npz_path)
        self.pools: Dict[int, Tensor] = {
            i: raw[name].to(device) for i, name in enumerate(CLASS_NAMES)
        }

    def sample_class(self, cid: int, n: int) -> Tensor:
        pool = self.pools[cid]
        idx  = torch.randint(0, pool.shape[0], (n,), device=self.device)
        return pool[idx]

    def full_class(self, cid: int) -> Tensor:
        return self.pools[cid]


# ─────────────────────────────────────────────────────────────────────────────
# Model  (unchanged from drift_ffhq.py)
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalDriftMLP(nn.Module):
    """
    f(n, c) = MLP( concat(n, E[c]) )

    Parameters
    ----------
    d_z      : noise / source dimension
    d_e      : embedding dimension
    d_out    : output dimension (= 512 for FFHQ ALAE)
    hidden   : hidden layer width
    n_hidden : number of hidden layers
    n_classes: number of classes (6)
    """

    def __init__(
        self,
        d_z:       int = 512,
        d_e:       int = 64,
        d_out:     int = 512,
        hidden:    int = 1024,
        n_hidden:  int = 3,
        n_classes: int = 6,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(n_classes, d_e)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        layers: List[nn.Module] = []
        prev = d_z + d_e
        for _ in range(n_hidden):
            lin = nn.Linear(prev, hidden)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.ReLU(inplace=True)])
            prev = hidden

        out_lin = nn.Linear(prev, d_out)
        nn.init.xavier_uniform_(out_lin.weight)
        nn.init.zeros_(out_lin.bias)
        layers.append(out_lin)

        self.mlp   = nn.Sequential(*layers)
        self.d_z   = d_z
        self.d_e   = d_e
        self.d_out = d_out

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        """
        noise  : (B, d_z)
        labels : (B,)  int64  in {0 .. n_classes-1}
        returns: (B, d_out)
        """
        e = self.embedding(labels)               # (B, d_e)
        x = torch.cat([noise, e], dim=1)         # (B, d_z + d_e)
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay  = decay
        self.shadow = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE KDE loss
# ─────────────────────────────────────────────────────────────────────────────

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

    reward = (reward - reward.mean()) / (reward.std() + 1e-8)           # normalise

    loss  = (log_q * reward).mean()
    return loss, (log_q - log_p).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def set_lr(optimizer: torch.optim.Optimizer, step: int,
           warmup: int, base_lr: float, emb_base_lr: float) -> float:
    """Linear warmup then constant; separate base LRs per group."""
    scale = (step + 1) / max(warmup, 1) if (warmup > 0 and step < warmup) else 1.0
    lrs   = [base_lr * scale, emb_base_lr * scale]
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr
    return lrs[0]


# ─────────────────────────────────────────────────────────────────────────────
# EMD via POT
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def emd_pot(x: Tensor, y: Tensor) -> float:
    if ot is None:
        raise RuntimeError("POT not installed.  pip install pot")
    x_np = x.cpu().float().numpy().astype(np.float64)
    y_np = y.cpu().float().numpy().astype(np.float64)
    n, m = len(x_np), len(y_np)
    a    = np.full(n, 1.0 / n)
    b    = np.full(m, 1.0 / m)
    C    = ot.dist(x_np, y_np, metric="euclidean") ** 2
    return float(ot.emd2(a, b, C))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_emd_global(iters: List[int], vals: List[float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, vals, color="#2563EB", linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Step"); ax.set_ylabel("EMD²")
    ax.set_title("Global EMD² (generated vs test)"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}")


def _plot_emd_perclass(
    iters: List[int],
    by_class: Dict[int, List[float]],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cid, name in enumerate(CLASS_NAMES):
        vals = by_class.get(cid, [])
        if not vals:
            continue
        ax.plot(iters[:len(vals)], vals, color=CLASS_COLORS[cid],
                linewidth=1.6, marker="o", markersize=3, label=name)
    ax.set_xlabel("Step"); ax.set_ylabel("EMD²")
    ax.set_title("Per-class EMD² over training")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}")


def _plot_loss(steps: List[int], losses: List[float],
               kls: List[float], path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(steps, losses, color="#DC2626", linewidth=1.2)
    ax1.set_title("REINFORCE loss (per class avg)"); ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3)
    ax2.plot(steps, kls, color="#16A34A", linewidth=1.2)
    ax2.set_title("KL proxy: E[log_q − log_p]"); ax2.set_xlabel("Step")
    ax2.set_ylabel("log_q − log_p"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}")


def plot_pca_trajectory(
    snapshots_by_class: Dict[str, Dict[int, torch.Tensor]],
    target_latents: Dict[int, torch.Tensor],
    save_path: str,
) -> None:
    """
    PCA scatter coloured by class showing:
      • target distribution (solid markers)
      • initial generator state (faint)
      • final generator state (vivid)
    """
    if not _HAS_SKLEARN:
        print("[plot] scikit-learn not found; skipping PCA plot.")
        return

    # Collect all points for PCA fitting
    all_pts = []
    for cid in range(N_CLASSES):
        for snap_dict in snapshots_by_class.values():
            if cid in snap_dict:
                all_pts.append(snap_dict[cid])
        if cid in target_latents:
            all_pts.append(target_latents[cid])
    all_pts_np = torch.cat(all_pts, dim=0).cpu().numpy()

    pca = SklearnPCA(n_components=2)
    pca.fit(all_pts_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Initial (step 1)", "Final (last step)"]
    keys   = ["source", "final"]
    alphas = [0.3, 0.65]

    for ax, key, alpha, title in zip(axes, keys, alphas, titles):
        snap = snapshots_by_class.get(key, {})
        for cid, name in enumerate(CLASS_NAMES):
            color = CLASS_COLORS[cid]
            # Generated
            if cid in snap:
                pts = pca.transform(snap[cid].cpu().numpy())
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=4, alpha=alpha, color=color,
                           label=f"gen {name}" if key == "final" else None,
                           rasterized=True)
            # Target
            if cid in target_latents:
                tgt = pca.transform(target_latents[cid].cpu().numpy())
                ax.scatter(tgt[:, 0], tgt[:, 1],
                           s=4, alpha=0.2, color=color,
                           marker="^", rasterized=True,
                           label=f"tgt {name}" if key == "final" else None)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
        ax.grid(alpha=0.2)

    # Single legend from the "final" panel
    axes[1].legend(fontsize=6, markerscale=2, ncol=2,
                   bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.suptitle("Conditional REINFORCE: PCA snapshots (initial vs target)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or "cpu" in args.device else "cpu")
    print(f"[device]  {device}")

    # ── Run directory ─────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{ts}_reinforce_temp{str(args.temp).replace('.','p')}"
    run_dir  = os.path.join(args.run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(run_dir, "logs.jsonl")
    print(f"[run]     {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[data]    loading {args.train_npz} …")
    train_ds = LatentDataset(args.train_npz, device)
    print(f"[data]    loading {args.test_npz} …")
    test_ds  = LatentDataset(args.test_npz,  device)

    # Fixed evaluation targets (subsampled from test)
    seed_all(args.seed + 777)
    eval_target: Dict[int, Tensor] = {}
    for cid in range(N_CLASSES):
        pool = test_ds.full_class(cid)
        n    = min(args.emd_samples, pool.shape[0])
        idx  = torch.randperm(pool.shape[0], device=device)[:n]
        eval_target[cid] = pool[idx]
    seed_all(args.seed)

    # ── Model ─────────────────────────────────────────────────────────────────
    gen = ConditionalDriftMLP(
        d_z      = args.d_z,
        d_e      = args.d_e,
        d_out    = LATENT_DIM,
        hidden   = args.hidden,
        n_hidden = args.n_hidden,
        n_classes= N_CLASSES,
    ).to(device)

    ema      = EMA(gen, decay=args.ema_decay)
    n_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"[model]   ConditionalDriftMLP  ({n_params:,} params)")
    print(f"[loss]    REINFORCE KDE  temp={args.temp}  maxent_coef={args.maxent_coef}")
    print(f"          nneg={args.nneg}  npos={args.npos}\n")

    # ── Optimiser: separate LRs (mirror drift_ffhq.py) ────────────────────────
    opt = torch.optim.AdamW([
        {"params": gen.mlp.parameters(),       "lr": args.lr,     "weight_decay": args.weight_decay},
        {"params": gen.embedding.parameters(), "lr": args.emb_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))

    # Fixed eval noise (same noise for every EMD snapshot → comparable)
    seed_all(args.seed + 999)
    eval_noise  = torch.randn(args.emd_samples * N_CLASSES, args.d_z, device=device)
    eval_labels = torch.repeat_interleave(
        torch.arange(N_CLASSES, device=device), args.emd_samples
    )
    seed_all(args.seed)

    # ── Metric storage ────────────────────────────────────────────────────────
    log_steps:    List[int]              = []
    log_losses:   List[float]            = []
    log_kls:      List[float]            = []
    emd_steps:    List[int]              = []
    emd_global:   List[float]            = []
    emd_perclass: Dict[int, List[float]] = {i: [] for i in range(N_CLASSES)}

    # ── PCA snapshot storage ──────────────────────────────────────────────────
    pca_snapshots: Dict[str, Dict[int, torch.Tensor]] = {}

    def take_snapshot(tag: str) -> None:
        # Swap in EMA weights temporarily
        live_state = {n: p.clone() for n, p in gen.named_parameters()
                      if p.requires_grad}
        ema.copy_to(gen)
        gen.eval()
        with torch.no_grad():
            gen_eval = gen(eval_noise, eval_labels)                # (emd_samples*6, 512)
        snap = {}
        for cid in range(N_CLASSES):
            mask = (eval_labels == cid)
            snap[cid] = gen_eval[mask].cpu()
        pca_snapshots[tag] = snap
        # Restore live weights
        for n, p in gen.named_parameters():
            if n in live_state:
                p.data.copy_(live_state[n])
        gen.train()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"{'Step':>6}  {'Loss':>12}  {'KL-proxy':>10}  {'EMD²(global)':>14}  {'LR':>9}")
    print("─" * 60)

    gen.train()
    for step in range(1, args.steps + 1):
        if step == 1:
            take_snapshot("source")

        lr = set_lr(opt, step - 1, args.warmup_steps, args.lr, args.emb_lr)
        opt.zero_grad()

        total_loss = 0.0
        last_kl    = 0.0

        for cid in range(N_CLASSES):
            # Generated samples for this class
            noise    = torch.randn(args.nneg, args.d_z, device=device)
            c_labels = torch.full((args.nneg,), cid, dtype=torch.long, device=device)
            x_gen    = gen(noise, c_labels)                           # (nneg, 512)

            # Real samples for this class
            x_pos    = train_ds.sample_class(cid, args.npos)          # (npos, 512)

            loss, kl = reinforce_loss(
                x_gen, x_pos,
                temp        = args.temp,
                maxent_coef = args.maxent_coef,
            )
            (loss / N_CLASSES).backward()
            total_loss += loss.item()
            last_kl    += kl

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
        opt.step()
        ema.update(gen)

        avg_loss = total_loss / N_CLASSES
        avg_kl   = last_kl   / N_CLASSES

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0 or step == 1:
            log_steps.append(step)
            log_losses.append(avg_loss)
            log_kls.append(avg_kl)
            rec = {"step": step, "loss": round(avg_loss, 6),
                   "kl_proxy": round(avg_kl, 6), "lr": round(lr, 8)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        # ── EMD evaluation ────────────────────────────────────────────────────
        emd_str = "             —"
        if (step % args.emd_every == 0 or step == 1) and ot is not None:
            # Swap in EMA weights temporarily
            live_state = {n: p.clone() for n, p in gen.named_parameters()
                          if p.requires_grad}
            ema.copy_to(gen)
            gen.eval()

            with torch.no_grad():
                gen_eval = gen(eval_noise, eval_labels)                # (emd_samples*6, 512)

            all_gen = gen_eval
            all_tgt = torch.cat([eval_target[c] for c in range(N_CLASSES)], 0)

            try:
                g_emd = emd_pot(all_gen, all_tgt)
                emd_steps.append(step)
                emd_global.append(g_emd)
                emd_str = f"{g_emd:>14.5f}"
                for cid in range(N_CLASSES):
                    mask  = (eval_labels == cid)
                    c_emd = emd_pot(gen_eval[mask], eval_target[cid])
                    emd_perclass[cid].append(c_emd)
            except Exception as e:
                print(f"  [emd warning] {e}")

            # Restore live weights
            for n, p in gen.named_parameters():
                if n in live_state:
                    p.data.copy_(live_state[n])
            gen.train()

        # ── Console ───────────────────────────────────────────────────────────
        if step % args.log_every == 0 or step == 1:
            print(f"{step:>6}  {avg_loss:>12.6f}  {avg_kl:>10.4f}  {emd_str}  {lr:>9.2e}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % args.save_every == 0 or step == args.steps:
            ema.copy_to(gen)
            ckpt = {
                "step":      step,
                "model":     gen.state_dict(),
                "ema_state": {n: ema.shadow[n].clone() for n in ema.shadow},
                "optimizer": opt.state_dict(),
                "model_config": dict(
                    d_z=args.d_z, d_e=args.d_e, d_out=LATENT_DIM,
                    hidden=args.hidden, n_hidden=args.n_hidden, n_classes=N_CLASSES,
                ),
                "class_names": CLASS_NAMES,
                "args":        vars(args),
            }
            tag  = "ckpt_final.pt" if step == args.steps else f"ckpt_step{step}.pt"
            path = os.path.join(run_dir, tag)
            torch.save(ckpt, path)
            print(f"  [ckpt] {path}")

            # Restore live weights after save
            for n, p in gen.named_parameters():
                if p.requires_grad:
                    p.data.copy_(ema.shadow[n])   # ema already copied above

    # ── Final snapshot ────────────────────────────────────────────────────────
    take_snapshot("final")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if emd_steps:
        _plot_emd_global(emd_steps, emd_global,
                         os.path.join(run_dir, "emd_global.png"))
        _plot_emd_perclass(emd_steps, emd_perclass,
                           os.path.join(run_dir, "emd_perclass.png"))

    if log_steps:
        _plot_loss(log_steps, log_losses, log_kls,
                   os.path.join(run_dir, "loss.png"))

    # PCA plot
    pca_target_cpu = {cid: t.cpu() for cid, t in eval_target.items()}
    plot_pca_trajectory(pca_snapshots, pca_target_cpu,
                        os.path.join(run_dir, args.pca_plot))

    # ── Final summary ─────────────────────────────────────────────────────────
    if emd_global:
        pct = (1 - emd_global[-1] / max(emd_global[0], 1e-9)) * 100
        print(f"\n[summary]  Initial EMD² (global): {emd_global[0]:.5f}")
        print(f"[summary]  Final   EMD² (global): {emd_global[-1]:.5f}")
        print(f"[summary]  Reduction:             {pct:.1f} %")
        print("\n[summary]  Per-class final EMD²:")
        for cid, name in enumerate(CLASS_NAMES):
            vals = emd_perclass[cid]
            if vals:
                print(f"             {name:<20s} {vals[-1]:.5f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
