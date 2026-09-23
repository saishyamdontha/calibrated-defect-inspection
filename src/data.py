"""MVTec AD dataset loader.

Returns per sample: normalized image, binary defect mask, label (0 good / 1 defect),
defect type and file path. `max_images` subsamples the split for training-data curves.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", resize=256, crop=224, max_images=None, seed=0):
        self.root = Path(root) / category
        if not self.root.exists():
            raise FileNotFoundError(f"Category folder not found: {self.root}")
        self.split = split
        self.crop = crop
        self.img_tf = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(crop),
            T.ToTensor(),
        ])
        self.samples = self._collect()
        if max_images is not None and max_images < len(self.samples):
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(self.samples), size=max_images, replace=False))
            self.samples = [self.samples[i] for i in idx]

    def _collect(self):
        split_dir = self.root / self.split
        samples = []
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if defect != "good":
                    mask_path = self.root / "ground_truth" / defect / f"{img_path.stem}_mask.png"
                samples.append((img_path, defect, mask_path))
        if not samples:
            raise RuntimeError(f"No images found in {split_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, defect, mask_path = self.samples[i]
        image = self.img_tf(Image.open(img_path).convert("RGB"))
        if mask_path is None:
            mask = torch.zeros(1, self.crop, self.crop)
        else:
            mask = (self.mask_tf(Image.open(mask_path).convert("L")) > 0.5).float()
        return {
            "image": image,
            "mask": mask,
            "label": 0 if defect == "good" else 1,
            "defect": defect,
            "path": str(img_path),
        }
