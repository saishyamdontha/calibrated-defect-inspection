"""Save anomaly heatmap overlays for specific test images.

Uses the same fit/calibration split as run_patchcore for the given seed.

Usage:
  python -m src.visualize --category metal_nut --backbone wrn50 --images data/mvtec/metal_nut/test/good/016.png data/mvtec/metal_nut/test/good/007.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.conformal import conformal_threshold, three_way_decisions
from src.data import IMAGENET_MEAN, IMAGENET_STD, MVTecDataset
from src.patchcore import PatchCore, build_extractor


def fit_model(data_root, category, backbone, seed, calib_frac, device):
    full = MVTecDataset(data_root, category, "train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(full))
    n_cal = max(1, int(round(len(full) * calib_frac)))
    cal_ds = Subset(full, perm[:n_cal].tolist())
    fit_ds = Subset(full, perm[n_cal:].tolist())

    model = PatchCore(build_extractor(backbone), device, seed=seed)
    model.fit(DataLoader(fit_ds, batch_size=16, shuffle=False, num_workers=0))

    cal_scores = []
    for b in DataLoader(cal_ds, batch_size=16, shuffle=False, num_workers=0):
        s, _ = model.predict(b["image"], b["image"].shape[-2:])
        cal_scores.append(s)
    return model, torch.cat(cal_scores).numpy()


def denormalize(image):
    img = image.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", choices=["wrn50", "dinov2_s"], default="wrn50")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--alpha-ok", type=float, default=0.30)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--out-dir", default="results/figures")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cal_scores = fit_model(a.data_root, a.category, a.backbone, a.seed, a.calib_frac, device)
    t_defect = conformal_threshold(cal_scores, a.alpha_defect)
    t_ok = conformal_threshold(cal_scores, a.alpha_ok)

    test_ds = MVTecDataset(a.data_root, a.category, "test")
    index = {Path(s[0]).resolve(): i for i, s in enumerate(test_ds.samples)}

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in a.images:
        key = Path(img_path).resolve()
        if key not in index:
            print(f"Skipping (not in {a.category} test set): {img_path}")
            continue
        item = test_ds[index[key]]
        score, amap = model.predict(item["image"].unsqueeze(0), item["image"].shape[-2:])
        score = float(score[0])
        amap = amap[0].numpy()
        decision = three_way_decisions([score], t_ok, t_defect)[0]
        mask = item["mask"][0].numpy()

        vmax = max(t_defect * 1.2, float(amap.max())) if np.isfinite(t_defect) else float(amap.max())
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(denormalize(item["image"]))
        axes[0].set_title(f"{item['defect']} / {key.name}")
        axes[1].imshow(denormalize(item["image"]))
        hm = axes[1].imshow(amap, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        if mask.sum() > 0:
            for ax in axes:
                ax.contour(mask, levels=[0.5], colors="lime", linewidths=1)
        axes[1].set_title(f"score {score:.2f} | t_defect {t_defect:.2f} -> {decision}")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(hm, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        out = out_dir / f"{a.category}_{a.backbone}_{item['defect']}_{key.stem}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"{key.name}: score={score:.3f}, decision={decision} -> {out}")


if __name__ == "__main__":
    main()
