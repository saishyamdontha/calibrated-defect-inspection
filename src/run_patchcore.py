"""Fit PatchCore on one MVTec AD category and evaluate it.

Usage (from the project root):
  python -m src.run_patchcore --category screw
  python -m src.run_patchcore --category screw --max-train 25 --seed 1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import MVTecDataset
from src.patchcore import CNNFeatureExtractor, PatchCore


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--category", required=True)
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--max-train", type=int, default=None, help="subsample normal training images")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MVTecDataset(a.data_root, a.category, "train", max_images=a.max_train, seed=a.seed)
    test_ds = MVTecDataset(a.data_root, a.category, "test")
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    extractor = CNNFeatureExtractor()
    model = PatchCore(extractor, device, coreset_ratio=a.coreset_ratio, seed=a.seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    t0 = time.perf_counter()
    model.fit(train_loader)
    sync(device)
    fit_seconds = time.perf_counter() - t0

    scores, labels, defects, maps, masks = [], [], [], [], []
    infer_seconds, n_images = 0.0, 0
    for batch in test_loader:
        sync(device)
        t = time.perf_counter()
        s, amap = model.predict(batch["image"], image_size=batch["image"].shape[-2:])
        sync(device)
        infer_seconds += time.perf_counter() - t
        n_images += len(s)
        scores.append(s)
        labels.append(batch["label"])
        defects.extend(batch["defect"])
        maps.append(amap)
        masks.append(batch["mask"].squeeze(1))

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    pixel_maps = torch.cat(maps).numpy().ravel()
    pixel_masks = (torch.cat(masks).numpy().ravel() > 0.5).astype(np.uint8)

    image_auroc = float(roc_auc_score(labels, scores))
    pixel_auroc = float(roc_auc_score(pixel_masks, pixel_maps))
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None

    result = {
        "method": "patchcore",
        "backbone": extractor.name,
        "category": a.category,
        "n_train_images": len(train_ds),
        "n_test_images": len(test_ds),
        "coreset_ratio": a.coreset_ratio,
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "seed": a.seed,
        "device": device,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "fit_seconds": fit_seconds,
        "ms_per_image": 1000 * infer_seconds / n_images,
        "batch_size": a.batch_size,
        "peak_gpu_mem_mb": peak_mem_mb,
    }

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"patchcore_{extractor.name}_{a.category}_n{len(train_ds)}_seed{a.seed}"
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / f"{tag}_scores.npz", scores=scores, labels=labels, defects=np.array(defects))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
