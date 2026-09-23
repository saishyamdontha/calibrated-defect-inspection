"""Score-level ensemble of the CNN and DINOv2 detectors, from saved score files (no refitting).

Each backbone's scores are rescaled by its own calibration scores
((score - median) / (q95 - median)), then averaged. Both backbones use the same
fit/calibration split for a given seed, so calibration images line up.

Note: the rescaling uses the calibration scores that also set the thresholds, which
weakens the exact conformal guarantee slightly; results should be read as empirical.

Usage:
  python -m src.ensemble
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.conformal import conformal_threshold, summarize, three_way_decisions

PATTERN = re.compile(r"patchcore_(wrn50|dinov2_s)_(.+)_fit(\d+)_seed(\d+)_scores\.npz$")


def rescale(x, cal):
    med = np.median(cal)
    q95 = np.quantile(cal, 0.95)
    return (x - med) / max(q95 - med, 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--alpha-ok", type=float, default=0.30)
    a = p.parse_args()

    files = {}
    for f in Path(a.results_dir).glob("*_scores.npz"):
        m = PATTERN.search(f.name)
        if m:
            files[(m.group(2), int(m.group(3)), int(m.group(4)), m.group(1))] = f

    rows = []
    for cat, n_fit, seed in sorted({(c, n, s) for c, n, s, _ in files}):
        k_cnn, k_vit = (cat, n_fit, seed, "wrn50"), (cat, n_fit, seed, "dinov2_s")
        if k_cnn not in files or k_vit not in files:
            continue
        cnn, vit = np.load(files[k_cnn]), np.load(files[k_vit])
        labels = cnn["test_labels"]
        if not np.array_equal(labels, vit["test_labels"]) or len(cnn["cal_scores"]) != len(vit["cal_scores"]):
            raise SystemExit(f"Mismatched files for {cat} fit{n_fit} seed{seed}")

        variants = {
            "wrn50": (cnn["test_scores"], cnn["cal_scores"]),
            "dinov2_s": (vit["test_scores"], vit["cal_scores"]),
            "ensemble": (
                (rescale(cnn["test_scores"], cnn["cal_scores"]) + rescale(vit["test_scores"], vit["cal_scores"])) / 2,
                (rescale(cnn["cal_scores"], cnn["cal_scores"]) + rescale(vit["cal_scores"], vit["cal_scores"])) / 2,
            ),
        }
        for name, (test, cal) in variants.items():
            t_defect = conformal_threshold(cal, a.alpha_defect)
            t_ok = conformal_threshold(cal, a.alpha_ok)
            single = summarize(np.where(test > t_defect, "defect", "ok"), labels)
            three = summarize(three_way_decisions(test, t_ok, t_defect), labels)
            rows.append({
                "category": cat, "n_fit": n_fit, "seed": seed, "model": name,
                "image_auroc": float(roc_auc_score(labels, test)),
                "false_alarm": single["false_alarm_rate"],
                "miss_single": single["miss_rate"],
                "miss_three_way": three["miss_rate"],
                "unsure_rate": three["unsure_rate"],
            })

    if not rows:
        raise SystemExit("No matching CNN/DINOv2 score file pairs found.")

    df = pd.DataFrame(rows)
    df = df[df.n_fit == df.groupby("category").n_fit.transform("max")]
    agg = (df.groupby(["category", "model"])
             .agg(n_seeds=("seed", "nunique"),
                  image_auroc_mean=("image_auroc", "mean"),
                  image_auroc_std=("image_auroc", "std"),
                  false_alarm=("false_alarm", "mean"),
                  miss_single=("miss_single", "mean"),
                  miss_three_way=("miss_three_way", "mean"),
                  unsure_rate=("unsure_rate", "mean"))
             .reset_index())
    out = Path(a.results_dir) / "ensemble_summary.csv"
    agg.to_csv(out, index=False)
    print(agg.round(3).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
