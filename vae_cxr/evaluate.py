"""Generate X-rays with the trained VAE and score them with IS and FID (PyTorch-Ignite tutorial code).

Default FID is the tutorial's FID() on Inception's 1000 class probabilities; fid_features pool uses the standard 2048-d pool features instead.
"""

import argparse
import json
import os

import ignite.distributed as idist
import numpy as np
import PIL.Image as Image
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from data import load_split
from figures import save_grid
from vae import VAE

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/vae64")
p.add_argument("--data_root", default="data")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--fid_features", choices=["probs", "pool"], default="probs")
args = p.parse_args()

device = idist.device()
out = f"{args.run}/eval" if args.fid_features == "probs" else f"{args.run}/eval_pool"
os.makedirs(out, exist_ok=True)

state = torch.load(f"{args.run}/checkpoints/best.pt", map_location=device, weights_only=False)
cfg = state["config"]
model = VAE(cfg["image_size"], cfg["latent_dim"], cfg["base_channels"]).to(device).eval()
model.load_state_dict(state["model"])

train_x, _ = load_split(args.data_root, "train", cfg["image_size"])
test_x, _ = load_split(args.data_root, "test", cfg["image_size"])
train_dl = DataLoader(TensorDataset(train_x), batch_size=args.batch_size)
test_dl = DataLoader(TensorDataset(test_x), batch_size=args.batch_size)


def interpolate(batch):
    """Resize to 299x299 for Inception (from the tutorial)."""
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img.cpu().clamp(0, 1).repeat(3, 1, 1))
        arr.append(transforms.ToTensor()(pil_img.resize((299, 299), Image.BILINEAR)))
    return torch.stack(arr)


def make_fid():
    if args.fid_features == "pool":
        return FID(num_features=2048, feature_extractor=InceptionModel(return_features=True, device=device), device=device)
    return FID(device=device)


def evaluation_step(engine, batch):
    """One generated image per real image; returns (fake, real)."""
    with torch.no_grad():
        noise = torch.randn(len(batch[0]), cfg["latent_dim"], device=device)
        return interpolate(model.decoder(noise)), interpolate(batch[0])


fids, iss, is_real = [], [], None
for r in range(args.repeats):
    torch.manual_seed(r)
    evaluator = Engine(evaluation_step)
    make_fid().attach(evaluator, "fid")
    InceptionScore(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")
    if r == 0:  # IS of real images, for reference
        InceptionScore(device=device, output_transform=lambda x: x[1]).attach(evaluator, "is_real")
    m = evaluator.run(train_dl).metrics
    fids.append(m["fid"])
    iss.append(m["is"])
    if r == 0:
        is_real = m["is_real"]
    print(f"repeat {r + 1}: FID = {m['fid']:.4f}, IS = {m['is']:.4f}")

# reference FID between two sets of real images (624 test vs. first 624 train)
train_iter = iter(train_dl)
ref = Engine(lambda e, b: (interpolate(b[0]), interpolate(next(train_iter)[0][: len(b[0])])))
make_fid().attach(ref, "fid")
fid_real = ref.run(test_dl).metrics["fid"]

results = {"epoch": state["epoch"], "n_images": len(train_x), "fid_features": args.fid_features,
           "fid_mean": float(np.mean(fids)), "fid_std": float(np.std(fids)), "fid_all": fids,
           "is_mean": float(np.mean(iss)), "is_std": float(np.std(iss)), "is_all": iss,
           "is_real": is_real, "fid_real_test_vs_train": fid_real}
json.dump(results, open(f"{out}/metrics.json", "w"), indent=2)
print(json.dumps(results, indent=2))

torch.manual_seed(0)
fake = model.sample(64, device).cpu()
save_grid(fake, f"{out}/samples_grid.png")
real = train_x[torch.randperm(len(train_x), generator=torch.Generator().manual_seed(3))[:32]]
g_real, g_fake = make_grid(real, nrow=8, pad_value=1.0), make_grid(fake[:32], nrow=8, pad_value=1.0)
save_image(torch.cat([g_real, torch.ones(3, g_real.shape[1], 12), g_fake], dim=2), f"{out}/real_vs_generated.png")
