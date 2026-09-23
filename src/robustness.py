"""Robustness of detection and conformal decisions under image shifts.

The model is fitted and calibrated on clean images; only the test images are shifted,
simulating conditions that drift after deployment (lighting, fixture angle, focus).

Usage:
  python -m src.robustness --category screw --backbone wrn50
  python -m src.robustness --category screw --backbone dinov2_s --seeds 0 1 2
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.conformal import conformal_threshold, summarize, three_way_decisions
from src.data import IMAGENET_MEAN, IMAGENET_STD, MVTecDataset
from src.visualize import fit_model

MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

SHIFTS = {
    "clean": None,
    "darker_0.7": ("brightness", 0.7),
    "brighter_1.3": ("brightness", 1.3),
    "rotate_5": ("rotate", 5.0),
    "rotate_15": ("rotate", 15.0),
    "blur_1": ("blur", 1.0),
    "blur_2": ("blur", 2.0),
}


def apply_shift(images, shift):
    if shift is None:
        return images
    kind, value = shift
    x = (images * STD + MEAN).clamp(0, 1)
    if kind == "brightness":
        x = TF.adjust_brightness(x, value)
    elif kind == "rotate":
        rotated = []
        for img in x:
            fill = img[:, 0, :].mean(dim=1).tolist()  # fill corners with the top-row background colour
            rotated.append(TF.rotate(img, value, interpolation=TF.InterpolationMode.BILINEAR, fill=fill))
        x = torch.stack(rotated)
    elif kind == "blur":
        k = 2 * math.ceil(3 * value) + 1
        x = TF.gaussian_blur(x, kernel_size=k, sigma=value)
    return (x.clamp(0, 1) - MEAN) / STD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", choices=["wrn50", "dinov2_s"], default="wrn50")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--alpha-ok", type=float, default=0.30)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(MVTecDataset(a.data_root, a.category, "test"),
                             batch_size=16, shuffle=False, num_workers=0)
    rows = []

    for seed in a.seeds:
        model, cal = fit_model(a.data_root, a.category, a.backbone, seed, 0.2, device)
        t_defect = conformal_threshold(cal, a.alpha_defect)
        t_ok = conformal_threshold(cal, a.alpha_ok)

        for name, shift in SHIFTS.items():
            scores, labels = [], []
            for b in test_loader:
                x = apply_shift(b["image"], shift)
                s, _ = model.predict(x, x.shape[-2:])
                scores.append(s)
                labels.append(b["label"])
            scores = torch.cat(scores).numpy()
            labels = torch.cat(labels).numpy()

            single = summarize(np.where(scores > t_defect, "defect", "ok"), labels)
            three = summarize(three_way_decisions(scores, t_ok, t_defect), labels)
            row = {
                "category": a.category, "backbone": a.backbone, "seed": seed, "shift": name,
                "image_auroc": float(roc_auc_score(labels, scores)),
                "false_alarm": single["false_alarm_rate"],
                "miss_single": single["miss_rate"],
                "miss_three_way": three["miss_rate"],
                "unsure_rate": three["unsure_rate"],
            }
            rows.append(row)
            print(f"seed {seed} | {name:<13} AUROC {row['image_auroc']:.3f} | false alarm {row['false_alarm']:.3f} "
                  f"| miss {row['miss_single']:.3f} -> {row['miss_three_way']:.3f} | unsure {row['unsure_rate']:.3f}")

    df = pd.DataFrame(rows)
    out = Path(a.out_dir) / f"robustness_{a.category}_{a.backbone}.csv"
    df.to_csv(out, index=False)
    agg = df.groupby("shift", sort=False).mean(numeric_only=True).drop(columns="seed")
    print(f"\nMean over seeds {a.seeds} ({a.category}, {a.backbone}):")
    print(agg.round(3).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
