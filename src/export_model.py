"""Fit and calibrate one deployable model and save it to models/.

Usage:
  python -m src.export_model --category screw --backbone dinov2_s --image-size 448
  python -m src.export_model --category metal_nut --backbone wrn50
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.conformal import conformal_threshold
from src.data import MVTecDataset
from src.patchcore import PatchCore, build_extractor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", choices=["wrn50", "dinov2_s"], default="wrn50")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--alpha-ok", type=float, default=0.30)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--out-dir", default="models")
    a = p.parse_args()

    if a.backbone == "dinov2_s" and a.image_size % 14 != 0:
        raise SystemExit("DINOv2 needs --image-size to be a multiple of 14.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize = int(round(a.image_size * 256 / 224))
    full = MVTecDataset(a.data_root, a.category, "train", resize=resize, crop=a.image_size)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(full))
    n_cal = max(1, int(round(len(full) * a.calib_frac)))
    cal_idx, fit_idx = perm[:n_cal].tolist(), perm[n_cal:].tolist()

    model = PatchCore(build_extractor(a.backbone, img_size=a.image_size), device,
                      coreset_ratio=a.coreset_ratio, seed=a.seed)
    model.fit(DataLoader(Subset(full, fit_idx), batch_size=16, shuffle=False, num_workers=0))

    cal_scores = []
    for b in DataLoader(Subset(full, cal_idx), batch_size=16, shuffle=False, num_workers=0):
        s, _ = model.predict(b["image"], b["image"].shape[-2:])
        cal_scores.append(s)
    cal_scores = torch.cat(cal_scores)

    t_defect = conformal_threshold(cal_scores.numpy(), a.alpha_defect)
    t_ok = conformal_threshold(cal_scores.numpy(), a.alpha_ok)
    if not np.isfinite(t_defect):
        print(f"Warning: {n_cal} calibration images are too few for alpha_defect={a.alpha_defect}; "
              "the model will never output 'defect'.")

    config = {
        "category": a.category,
        "backbone": a.backbone,
        "image_size": a.image_size,
        "resize": resize,
        "coreset_ratio": a.coreset_ratio,
        "seed": a.seed,
        "n_fit": len(fit_idx),
        "n_calib": len(cal_idx),
        "alpha_defect": a.alpha_defect,
        "alpha_ok": a.alpha_ok,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ckpt = {
        "config": config,
        "memory_bank": model.memory_bank.float().cpu(),
        "cal_scores": cal_scores.float().cpu(),
        "t_ok": float(t_ok),
        "t_defect": float(t_defect),
    }

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if a.image_size == 224 else f"_s{a.image_size}"
    out = out_dir / f"{a.category}_{a.backbone}{suffix}.pt"
    torch.save(ckpt, out)
    size_mb = out.stat().st_size / 1024**2
    print(f"Saved {out} ({size_mb:.0f} MB) | memory bank {tuple(model.memory_bank.shape)} | "
          f"t_ok={t_ok:.3f} t_defect={t_defect:.3f}")


if __name__ == "__main__":
    main()
