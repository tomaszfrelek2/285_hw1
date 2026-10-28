"""Train the VAE on chest X-rays. Outputs go to --out (config, losses, plots, samples, checkpoints)."""

import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_split
from figures import plot_losses, save_grid, save_interpolations, save_reconstructions
from vae import VAE, vae_loss

p = argparse.ArgumentParser()
p.add_argument("--data_root", default="data")
p.add_argument("--out", default="runs/vae64")
p.add_argument("--image_size", type=int, default=64)
p.add_argument("--latent_dim", type=int, default=128)
p.add_argument("--base_channels", type=int, default=32)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--kl_warmup_epochs", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(f"{args.out}/checkpoints", exist_ok=True)
os.makedirs(f"{args.out}/progress", exist_ok=True)

# Kaggle's val split has only 16 images, so the test split is used for validation
train_x, train_y = load_split(args.data_root, "train", args.image_size)
val_x, val_y = load_split(args.data_root, "test", args.image_size)
train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=256)

model = VAE(args.image_size, args.latent_dim, args.base_channels).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

cfg = vars(args) | {"n_train": len(train_x), "n_val": len(val_x),
                    "n_params": sum(p.numel() for p in model.parameters())}
json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=2)

fixed_z = torch.randn(64, args.latent_dim, device=device)
val_batch = val_x[torch.randperm(len(val_x), generator=torch.Generator().manual_seed(1))[:16]].to(device)


def run_epoch(loader, epoch=None):
    """One pass over loader; trains if epoch is given. Logged loss always uses the full beta."""
    model.train(epoch is not None)
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    with torch.set_grad_enabled(epoch is not None):
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            if epoch is not None:  # linear KL warm-up
                beta = args.beta * min(1.0, (epoch - 1 + (i + 1) / len(loader)) / args.kl_warmup_epochs)
                loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                _, rec, kl = vae_loss(x_hat, x, mu, logvar)
            tot["loss"] += (rec.item() + args.beta * kl.item()) * len(x)
            tot["recon"] += rec.item() * len(x)
            tot["kl"] += kl.item() * len(x)
    n = len(loader.dataset) if epoch is None else len(loader) * loader.batch_size
    return {k: v / n for k, v in tot.items()}


log = csv.writer(open(f"{args.out}/history.csv", "w", newline="", buffering=1))  # line-buffered
log.writerow(["epoch", "train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl", "seconds"])
best = float("inf")
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr = run_epoch(train_dl, epoch)
    va = run_epoch(val_dl)
    sched.step()
    log.writerow([epoch, *[f"{d[k]:.4f}" for d in (tr, va) for k in ("loss", "recon", "kl")], f"{time.time() - t0:.1f}"])
    print(f"epoch {epoch:3d} | train {tr['loss']:.2f} (rec {tr['recon']:.2f}, kl {tr['kl']:.2f}) | "
          f"val {va['loss']:.2f} (rec {va['recon']:.2f}, kl {va['kl']:.2f})", flush=True)

    state = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
    torch.save(state, f"{args.out}/checkpoints/last.pt")
    if va["loss"] < best:
        best = va["loss"]
        torch.save(state, f"{args.out}/checkpoints/best.pt")
    if epoch == 1 or epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            save_grid(model.decoder(fixed_z).cpu(), f"{args.out}/progress/epoch_{epoch:03d}.png")

model.eval()
plot_losses(args.out)
save_grid(model.sample(64, device).cpu(), f"{args.out}/samples_final.png")
save_reconstructions(model, val_batch, f"{args.out}/reconstructions.png")
save_interpolations(model, val_batch, f"{args.out}/interpolations.png")
print(f"done, best val loss {best:.2f}")
