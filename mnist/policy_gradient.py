"""
KDE Policy Gradient Generator for MNIST"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


# ── Autoencoder (provided) ────────────────────────────────────────────────────

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ── Generator / Policy π_θ ───────────────────────────────────────────────────

class Generator(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        noise_dim: int = 16,
        latent_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 3,
        class_emb_dim: int = 32,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes

        # +1 token for unconditional (index = num_classes)
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)

        # MLP body: [noise + class_emb] -> hidden x num_layers -> latent
        in_dim = noise_dim + class_emb_dim
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU(inplace=True)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        noise:        [B, noise_dim]
        class_labels: [B] int64 — 0..num_classes-1, or -1 for unconditional
        returns:      [B, latent_dim]
        """
        cls = class_labels.clone()
        cls[cls < 0] = self.num_classes
        c_emb = self.class_emb(cls)                      # [B, class_emb_dim]
        inp = torch.cat([noise, c_emb], dim=-1)          # [B, noise_dim + class_emb_dim]
        return self.mlp(inp)                             # [B, latent_dim]

    def rollout(self, n: int, device: torch.device, class_labels=None):
        """
        Sample n noise vectors and optionally assign class labels.

        class_labels: int, list[int], or None (-> random classes)
        Returns (epsilon, class_labels) both on device.
        """
        epsilon = torch.randn(n, self.noise_dim, device=device)
        if class_labels is None:
            cls = torch.randint(0, self.num_classes, (n,), device=device)
        elif isinstance(class_labels, int):
            cls = torch.full((n,), class_labels, dtype=torch.long, device=device)
        else:
            cls = torch.tensor(class_labels, dtype=torch.long, device=device)
        return epsilon, cls


# ── KDE Critic (non-parametric) ───────────────────────────────────────────────

class KDECritic:
    """Non-parametric critic via Gaussian KDE."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.sigma2 = sigma ** 2

    def log_density(self, queries, bank):
        """
        Log KDE density at query points given a bank of samples.
        queries: [Q, D], bank: [N, D] -> [Q]
        """
        diffs = queries.unsqueeze(1) - bank.unsqueeze(0)       # [Q, N, D]
        sq_dists = (diffs ** 2).sum(-1)                        # [Q, N]
        log_kernels = -sq_dists / (2 * self.sigma2)
        return torch.logsumexp(log_kernels, dim=1) - np.log(bank.shape[0])

    def score(self, queries, bank):
        """
        ∇_x log p_KDE(x) at query points.
        queries: [Q, D], bank: [N, D] -> [Q, D]
        """
        diffs = queries.unsqueeze(1) - bank.unsqueeze(0)       # [Q, N, D]
        sq_dists = (diffs ** 2).sum(-1)                        # [Q, N]
        log_w = -sq_dists / (2 * self.sigma2)
        w = torch.softmax(log_w, dim=1)                        # [Q, N]
        return (-diffs * w.unsqueeze(-1)).sum(1) / self.sigma2  # [Q, D]

    def advantage(self, z_gen, real_bank, gen_bank):
        score_real = self.score(z_gen, real_bank)    # ∇_z log p(z|c)  [Q, D]
        score_gen  = self.score(z_gen, gen_bank)     # ∇_z log q(z|c)  [Q, D]  (baseline)
        return score_real - score_gen                # net advantage    [Q, D]


# ── Policy Gradient Loss ──────────────────────────────────────────────────────

def pg_loss(z_gen, advantage):
    """Reparameterized policy gradient loss."""
    return -(z_gen * advantage.detach()).sum(dim=-1).mean()


# ── Precompute Latent Bank ────────────────────────────────────────────────────

@torch.no_grad()
def precompute_latents(ae, loader, device, num_classes=10):
    """Encode the entire training set once."""
    print("Precomputing latent bank for all classes...")
    zs, ys = [], []
    for imgs, labels in tqdm(loader, desc="Encoding"):
        zs.append(ae.encode(imgs.to(device)).cpu())
        ys.append(labels.cpu())

    z_all = torch.cat(zs)   # [N, latent_dim]
    y_all = torch.cat(ys)   # [N]

    by_class = {c: z_all[y_all == c] for c in range(num_classes)}
    sizes = {c: v.shape[0] for c, v in by_class.items()}
    print(f"Latent bank ready. Samples per class: {sizes}")
    return by_class


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(
    ae_checkpoint: str,
    n_epochs: int = 30,
    batch_size: int = 256,
    noise_dim: int = 6,
    ae_latent_dim: int = 6,
    kde_sigma: float = 1.5,
    lr: float = 3e-4,
    log_every: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    tf = transforms.ToTensor()
    mnist = datasets.MNIST("./data", train=True, download=True, transform=tf)

    # Use drop_last=False for the precompute pass so we get the full dataset
    precompute_loader = DataLoader(mnist, batch_size=512, shuffle=False, drop_last=False)
    train_loader      = DataLoader(mnist, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── AE (frozen feature extractor / environment) ───────────────────────────
    ae = ConvAE(latent_dim=ae_latent_dim).to(device)
    ae_ckpt = torch.load(ae_checkpoint, map_location=device)
    ae.load_state_dict(ae_ckpt["model"])
    print(f"Loaded AE from {ae_checkpoint}")

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    real_bank = precompute_latents(ae, precompute_loader, device)

    # for c in range(10):
    #     z = real_bank[c]
    #     dists = torch.cdist(z, z)
    #     print(f"class {c}: median pairwise dist = {dists.median():.3f}")

    # ── Policy (the only trainable network) ───────────────────────────────────
    policy = Generator(
        num_classes=10,
        noise_dim=noise_dim,
        latent_dim=ae_latent_dim,
    ).to(device)
    optimiser = optim.Adam(policy.parameters(), lr=lr)

    # ── Critic (no parameters, no optimiser) ──────────────────────────────────
    critic = KDECritic(sigma=kde_sigma)

    losses, kl_estimates = [], []
    print("\n=== KDE Policy Gradient Training (class-conditioned) ===")
    beta=0.1
    for epoch in range(1, n_epochs + 1):
        epoch_losses, epoch_kl = [], []

        for _, real_labels in tqdm(train_loader, total=len(train_loader)):
            real_labels = real_labels.to(device)   # [B] digit 0-9

            # Step 1: on-policy rollout, conditioned on the same class distribution
            epsilon, cls = policy.rollout(batch_size, device, class_labels=real_labels)
            z_gen = policy(epsilon, cls)            # [B, D]  has grad

            # Step 2: per-class conditional advantage using the full precomputed bank.
            # Each class gets ~6000 real samples → KDE is essentially exact.
            # No augmentation, no one-hot tricks, no scaling hacks.
            with torch.no_grad():
                z_gen_sg = z_gen.detach()
                adv = torch.zeros_like(z_gen_sg)

                for c in cls.unique():
                    c_int = c.item()
                    mask = cls == c_int
                    real_c = real_bank[c_int].to(device)   # [~6000, D]

                    adv[mask] = critic.advantage(
                        z_gen_sg[mask],
                        real_c,   # ∇ log p(z|c)  — pull toward real distribution
                        z_gen_sg[mask],    # ∇ log q(z|c)  — push away from current generated density
                    )

                    entropy_score = -critic.score(z_gen_sg[mask], z_gen_sg[mask])   # ∇_z H(q) ≈ -∇_z log q
                    adv[mask] += beta * entropy_score

                # normalise per-sample direction, not per-dimension
                adv = adv / (adv.norm(dim=-1, keepdim=True) + 1e-8)
                adv = adv / (adv.std(0) + 1e-8)

                log_q = torch.zeros(batch_size, device=device)
                log_p = torch.zeros(batch_size, device=device)

                for c in cls.unique():
                    c_int = c.item()
                    mask = cls == c_int
                    real_c = real_bank[c_int].to(device)
                    gen_c  = z_gen_sg[mask]

                    log_p[mask] = critic.log_density(gen_c, real_c)   # log p(z|c)
                    log_q[mask] = critic.log_density(gen_c, gen_c)    # log q(z|c)

                kl_est = (log_q - log_p).mean().item()

            # Step 3: policy gradient update
            loss = pg_loss(z_gen, adv)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()

            epoch_losses.append(loss.item())
            epoch_kl.append(kl_est)

        mean_loss = np.mean(epoch_losses)
        mean_kl   = np.mean(epoch_kl)
        losses.append(mean_loss)
        kl_estimates.append(mean_kl)

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs}  "
                  f"pg_loss={mean_loss:.4f}  KL(q||p)≈{mean_kl:.4f}")

            # ── Visualise — one row per digit ─────────────────────────────────
            policy.eval()
            with torch.no_grad():
                all_imgs = []
                for digit in range(10):
                    eps, cls_v = policy.rollout(8, device, class_labels=digit)
                    z = policy(eps, cls_v)
                    all_imgs.append(ae.decode(z).cpu())
                imgs = torch.cat(all_imgs, dim=0)   # [80, 1, 28, 28]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            grid = make_grid(imgs, nrow=8, padding=2, normalize=True)
            axes[0].imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
            axes[0].set_title("Generated — 8 samples per digit (rows 0-9)", fontsize=11)
            axes[0].axis("off")

            ax = axes[1]
            ax.plot(losses, label="PG loss", color="steelblue")
            ax2 = ax.twinx()
            ax2.plot(kl_estimates, color="coral", linestyle="--", label="KL(q||p)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("PG Loss", color="steelblue")
            ax2.set_ylabel("KL estimate", color="coral")
            ax.set_title("Training curves")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig("kde_pg_results.png", dpi=150)
            plt.close()
            policy.train()

    return policy, ae


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    policy, ae = train(
        ae_checkpoint="pretrained/ae/ae_final.pt",
        n_epochs=30,
        batch_size=1024,
        noise_dim=32,
        ae_latent_dim=6,
        kde_sigma=1.5,         # key hyperparameter: kernel bandwidth
        lr=3e-4,
        log_every=1,
    )

    torch.save(policy.state_dict(), "kde_pg_policy.pt")
    print("Saved -> kde_pg_policy.pt")
