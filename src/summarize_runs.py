"""Aggregate run JSONs into a results table and a training-data curve.

Usage:
  python -m src.summarize_runs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()
    out = Path(a.results_dir)

    rows = []
    for f in sorted(out.glob("patchcore_*_fit*_seed*.json")):
        r = json.loads(f.read_text())
        if "n_fit_images" in r:
            rows.append(r)
    if not rows:
        raise SystemExit("No run JSONs found.")

    df = pd.DataFrame(rows)
    agg = (df.groupby(["category", "backbone", "n_fit_images"])
             .agg(n_seeds=("seed", "nunique"),
                  image_auroc_mean=("image_auroc", "mean"),
                  image_auroc_std=("image_auroc", "std"),
                  pixel_auroc_mean=("pixel_auroc", "mean"),
                  pixel_auroc_std=("pixel_auroc", "std"),
                  ms_per_image_batched=("ms_per_image_batched", "mean"),
                  peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean"),
                  memory_bank=("memory_bank_size", "mean"))
             .reset_index())
    agg.to_csv(out / "runs_summary.csv", index=False)

    full = agg[agg.n_fit_images == agg.groupby(["category", "backbone"]).n_fit_images.transform("max")]
    print("Main results (all fit images, mean and std over seeds):")
    print(full.round(3).to_string(index=False))

    cats = sorted(c for c in agg.category.unique() if agg[agg.category == c].n_fit_images.nunique() > 1)
    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 4), squeeze=False)
    for ax, cat in zip(axes[0], cats):
        for bb, sub in agg[agg.category == cat].groupby("backbone"):
            sub = sub.sort_values("n_fit_images")
            ax.errorbar(sub.n_fit_images, sub.image_auroc_mean, yerr=sub.image_auroc_std.fillna(0),
                        marker="o", capsize=3, label=bb)
        ax.set_xscale("log")
        ax.set_title(cat)
        ax.set_xlabel("Normal images used for fitting")
        ax.set_ylabel("Image AUROC")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("How many good images does each backbone need?")
    fig.tight_layout()
    fig.savefig(out / "data_curve.png", dpi=150)
    print(f"\nSaved {out / 'runs_summary.csv'} and {out / 'data_curve.png'}")


if __name__ == "__main__":
    main()

