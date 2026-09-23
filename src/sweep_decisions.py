"""Sweep the OK threshold to trace the trade-off between missed defects and human review,
averaged over seeds, and check the false-alarm guarantee across seeds.

Usage:
  python -m src.sweep_decisions
  python -m src.sweep_decisions --alpha-defect 0.05
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.conformal import conformal_threshold, summarize, three_way_decisions

PATTERN = re.compile(r"patchcore_(wrn50|dinov2_s)_(.+)_fit(\d+)_seed(\d+)_scores\.npz$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--fit", type=int, default=None, help="only runs with this many fit images (default: largest per category)")
    a = p.parse_args()

    out = Path(a.results_dir)
    alphas_ok = np.round(np.arange(a.alpha_defect, 0.65, 0.05), 2)
    rows = []

    for f in sorted(out.glob("*_scores.npz")):
        m = PATTERN.search(f.name)
        if not m:
            continue
        backbone, category, n_fit, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        d = np.load(f, allow_pickle=True)
        if "cal_scores" not in d.files:
            continue
        cal, scores, labels = d["cal_scores"], d["test_scores"], d["test_labels"]
        t_defect = conformal_threshold(cal, a.alpha_defect)
        for alpha_ok in alphas_ok:
            t_ok = conformal_threshold(cal, alpha_ok) if alpha_ok > a.alpha_defect else t_defect
            s = summarize(three_way_decisions(scores, t_ok, t_defect), labels)
            rows.append({"backbone": backbone, "category": category, "n_fit": n_fit,
                         "seed": seed, "alpha_ok": float(alpha_ok), **s})

    if not rows:
        raise SystemExit("No matching *_fit*_seed*_scores.npz files found.")

    df = pd.DataFrame(rows)
    if a.fit is not None:
        df = df[df.n_fit == a.fit]
    else:
        df = df[df.n_fit == df.groupby(["backbone", "category"]).n_fit.transform("max")]
    df.to_csv(out / "decision_sweep_raw.csv", index=False)

    agg = (df.groupby(["category", "backbone", "alpha_ok"])
             .agg(n_seeds=("seed", "nunique"),
                  false_alarm_mean=("false_alarm_rate", "mean"),
                  false_alarm_std=("false_alarm_rate", "std"),
                  miss_mean=("miss_rate", "mean"),
                  miss_std=("miss_rate", "std"),
                  unsure_mean=("unsure_rate", "mean"),
                  unsure_std=("unsure_rate", "std"))
             .reset_index())
    agg.to_csv(out / "decision_sweep_summary.csv", index=False)

    base = agg[np.isclose(agg.alpha_ok, a.alpha_defect)]
    print(f"\nFalse-alarm check (target <= {a.alpha_defect:.0%}), single-threshold setting, mean over seeds:")
    print(base[["category", "backbone", "n_seeds", "false_alarm_mean", "false_alarm_std",
                "miss_mean", "miss_std"]].round(3).to_string(index=False))

    cats = sorted(agg.category.unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 4), squeeze=False)
    for ax, cat in zip(axes[0], cats):
        for bb, sub in agg[agg.category == cat].groupby("backbone"):
            sub = sub.sort_values("alpha_ok")
            ax.errorbar(sub.unsure_mean, sub.miss_mean, yerr=sub.miss_std.fillna(0),
                        marker="o", capsize=3, label=bb)
        ax.set_title(cat)
        ax.set_xlabel("Share of parts sent to human review")
        ax.set_ylabel("Missed defect rate")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Missed defects vs human review (false-alarm target {a.alpha_defect:.0%})")
    fig.tight_layout()
    fig.savefig(out / "decision_tradeoff.png", dpi=150)
    print(f"\nSaved {out / 'decision_tradeoff.png'} and CSV summaries in {out}")


if __name__ == "__main__":
    main()
