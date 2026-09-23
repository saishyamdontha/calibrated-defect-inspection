"""Recalibration after drift: can a few good images under the new conditions restore the
false-alarm target?

The memory bank is always fitted on clean images. For each shift, thresholds come from
(a) the original clean calibration images, or (b) calibration images under the same shift,
using the first k of them (k in --recal-sizes) or all of them.

Usage:
  python -m src.recalibrate --category metal_nut --backbone wrn50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.conformal import conformal_threshold, summarize, three_way_decisions
from src.data import MVTecDataset
from src.patchcore import PatchCore, build_extractor
from src.robustness import SHIFTS, apply_shift


def score(model, loader, shift):
    scores, labels = [], []
    for b in loader:
        x = apply_shift(b["image"], shift)
        s, _ = model.predict(x, x.shape[-2:])
        scores.append(s)
        labels.append(torch.as_tensor(b["label"]))
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", choices=["wrn50", "dinov2_s"], default="wrn50")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--recal-sizes", type=int, nargs="+", default=[20])
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--alpha-ok", type=float, default=0.30)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = MVTecDataset(a.data_root, a.category, "train")
    test_loader = DataLoader(MVTecDataset(a.data_root, a.category, "test"),
                             batch_size=16, shuffle=False, num_workers=0)
    rows = []

    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(full))
        n_cal = max(1, int(round(len(full) * 0.2)))
        cal_idx, fit_idx = perm[:n_cal].tolist(), perm[n_cal:].tolist()

        model = PatchCore(build_extractor(a.backbone), device, seed=seed)
        model.fit(DataLoader(Subset(full, fit_idx), batch_size=16, shuffle=False, num_workers=0))
        cal_loader = DataLoader(Subset(full, cal_idx), batch_size=16, shuffle=False, num_workers=0)
        clean_cal, _ = score(model, cal_loader, None)

        for name, shift in SHIFTS.items():
            test_s, test_y = score(model, test_loader, shift)
            shifted_cal, _ = score(model, cal_loader, shift)
            auroc = float(roc_auc_score(test_y, test_s))

            settings = {"clean_calib": clean_cal}
            for k in a.recal_sizes:
                if k < len(shifted_cal):
                    settings[f"recal_{k}"] = shifted_cal[:k]
            settings["recal_all"] = shifted_cal

            for setting, cal in settings.items():
                t_defect = conformal_threshold(cal, a.alpha_defect)
                t_ok = conformal_threshold(cal, a.alpha_ok)
                three = summarize(three_way_decisions(test_s, t_ok, t_defect), test_y)
                rows.append({
                    "category": a.category, "backbone": a.backbone, "seed": seed,
                    "shift": name, "calibration": setting, "n_calib": len(cal),
                    "image_auroc": auroc,
                    "false_alarm": three["false_alarm_rate"],
                    "miss_three_way": three["miss_rate"],
                    "unsure_rate": three["unsure_rate"],
                })
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    out = Path(a.out_dir) / f"recalibration_{a.category}_{a.backbone}.csv"
    df.to_csv(out, index=False)

    for metric in ["false_alarm", "miss_three_way", "unsure_rate"]:
        table = df.pivot_table(index="shift", columns="calibration", values=metric, aggfunc="mean", sort=False)
        print(f"\n{metric} ({a.category}, {a.backbone}, mean over seeds {a.seeds}):")
        print(table.round(3).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
