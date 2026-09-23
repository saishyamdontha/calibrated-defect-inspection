"""Fit PatchCore on one MVTec AD category, score a held-out calibration set of normal
images and the test set, and report detection, localization, speed and memory.

The normal training images are split into:
  fit   - used to build the memory bank
  calib - never seen during fitting; their scores set the conformal thresholds

Usage (from the project root):
  python -m src.run_patchcore --category screw --backbone wrn50
  python -m src.run_patchcore --category screw --backbone dinov2_s --seed 1
  python -m src.run_patchcore --category screw --max-train 25
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data import MVTecDataset
from src.patchcore import PatchCore, build_extractor


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def score_loader(model, loader, device, with_maps):
    scores, labels, defects, maps, masks = [], [], [], [], []
    seconds, n = 0.0, 0
    for batch in loader:
        sync(device)
        t = time.perf_counter()
        s, amap = model.predict(batch["image"], image_size=batch["image"].shape[-2:])
        sync(device)
        seconds += time.perf_counter() - t
        n += len(s)
        scores.append(s)
        labels.append(batch["label"])
        defects.extend(batch["defect"])
        if with_maps:
            maps.append(amap)
            masks.append(batch["mask"].squeeze(1))
    out = {
        "scores": torch.cat(scores).numpy(),
        "labels": torch.cat(labels).numpy(),
        "defects": np.array(defects),
        "seconds": seconds,
        "n": n,
    }
    if with_maps:
        out["maps"] = torch.cat(maps).numpy()
        out["masks"] = torch.cat(masks).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", choices=["wrn50", "dinov2_s"], default="wrn50")
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--calib-frac", type=float, default=0.2, help="share of normal train images held out for calibration")
    p.add_argument("--max-train", type=int, default=None, help="cap on normal images used for fitting")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_train = MVTecDataset(a.data_root, a.category, "train")
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(full_train))
    n_cal = max(1, int(round(len(full_train) * a.calib_frac)))
    cal_idx = perm[:n_cal].tolist()
    fit_idx = perm[n_cal:].tolist()
    if a.max_train is not None:
        fit_idx = fit_idx[:a.max_train]

    fit_ds = Subset(full_train, fit_idx)
    cal_ds = Subset(full_train, cal_idx)
    test_ds = MVTecDataset(a.data_root, a.category, "test")

    fit_loader = DataLoader(fit_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    extractor = build_extractor(a.backbone)
    model = PatchCore(extractor, device, coreset_ratio=a.coreset_ratio, seed=a.seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    t0 = time.perf_counter()
    model.fit(fit_loader)
    sync(device)
    fit_seconds = time.perf_counter() - t0

    cal = score_loader(model, cal_loader, device, with_maps=False)
    test = score_loader(model, test_loader, device, with_maps=True)

    image_auroc = float(roc_auc_score(test["labels"], test["scores"]))
    pixel_masks = (test["masks"].ravel() > 0.5).astype(np.uint8)
    pixel_auroc = float(roc_auc_score(pixel_masks, test["maps"].ravel()))
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None

    result = {
        "method": "patchcore",
        "backbone": extractor.name,
        "category": a.category,
        "n_fit_images": len(fit_ds),
        "n_calib_images": len(cal_ds),
        "n_test_images": len(test_ds),
        "coreset_ratio": a.coreset_ratio,
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "seed": a.seed,
        "device": device,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "fit_seconds": fit_seconds,
        "ms_per_image_batched": 1000 * test["seconds"] / test["n"],
        "batch_size": a.batch_size,
        "peak_gpu_mem_mb": peak_mem_mb,
    }

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"patchcore_{extractor.name}_{a.category}_fit{len(fit_ds)}_seed{a.seed}"
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        out_dir / f"{tag}_scores.npz",
        test_scores=test["scores"],
        test_labels=test["labels"],
        test_defects=test["defects"],
        cal_scores=cal["scores"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
