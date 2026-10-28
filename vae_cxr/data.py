import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def find_root(data_root):
    """Shallowest folder containing train/ (the Kaggle zip has a nested copy and __MACOSX)."""
    dirs = [p.parent for p in Path(data_root).rglob("train") if "__MACOSX" not in p.parts]
    return min(dirs, key=lambda p: len(p.parts))


def load_split(data_root, split, image_size, cache_dir="cache"):
    """Returns float images [N,1,H,W] in [0,1] and labels (0=NORMAL, 1=PNEUMONIA); cached after first call."""
    cache = Path(cache_dir) / f"{split}_{image_size}.pt"
    if not cache.exists():
        os.makedirs(cache_dir, exist_ok=True)
        tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size), transforms.PILToTensor()])
        files, labels = [], []
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            f = sorted((find_root(data_root) / split / cls).glob("*.jpeg"))
            files += f
            labels += [label] * len(f)
        imgs = torch.stack([tf(Image.open(f)) for f in files])
        torch.save({"images": imgs, "labels": torch.tensor(labels)}, cache)
        print(f"{split}: {len(files)} images cached to {cache}")
    d = torch.load(cache)
    return d["images"].float() / 255, d["labels"]
