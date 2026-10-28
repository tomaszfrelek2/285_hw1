import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import torch  
from torchvision.utils import save_image  


def plot_losses(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "history.csv"))))
    h = {k: [float(r[k]) for r in rows] for k in rows[0]}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, key, title in zip(axes, ["loss", "recon", "kl"], ["Total loss (neg. ELBO)", "Reconstruction", "KL divergence"]):
        ax.plot(h["epoch"], h[f"train_{key}"], color="#2a6fdb", lw=1.4, label="train")
        ax.plot(h["epoch"], h[f"val_{key}"], color="#d1495b", lw=1.4, label="validation")
        vals = h[f"train_{key}"][1:] + h[f"val_{key}"][1:]
        if vals:
            pad = 0.08 * (max(vals) - min(vals) or 1)
            ax.set_ylim(min(vals) - pad, max(vals) + pad)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("nats / image")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(run_dir, f"loss_curves.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_grid(images, path, nrow=8):
    save_image(images.clamp(0, 1), path, nrow=nrow, padding=2, pad_value=1.0)


@torch.no_grad()
def save_reconstructions(model, real, path, n=8):
    """Top row: real, bottom row: reconstructed."""
    save_grid(torch.cat([real[:n], model.reconstruct(real[:n])]).cpu(), path, nrow=n)


@torch.no_grad()
def save_interpolations(model, real, path, n_pairs=4, steps=10):
    """Each row walks linearly through latent space between two encoded images."""
    mu = model.encoder(real[: 2 * n_pairs])[0]
    t = torch.linspace(0, 1, steps, device=mu.device)[:, None]
    rows = [model.decoder((1 - t) * mu[2 * i] + t * mu[2 * i + 1]) for i in range(n_pairs)]
    save_grid(torch.cat(rows).cpu(), path, nrow=steps)
