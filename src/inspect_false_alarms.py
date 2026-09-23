"""List normal test images flagged as defects, and in how many seeds.

Usage:
  python -m src.inspect_false_alarms --category metal_nut --backbone wrn50
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.conformal import conformal_threshold
from src.data import MVTecDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", choices=["wrn50", "dinov2_s"], required=True)
    p.add_argument("--alpha-defect", type=float, default=0.05)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-root", default="data/mvtec")
    a = p.parse_args()

    test_ds = MVTecDataset(a.data_root, a.category, "test")
    paths = [str(s[0]) for s in test_ds.samples]

    pattern = re.compile(rf"patchcore_{a.backbone}_{a.category}_fit(\d+)_seed(\d+)_scores\.npz$")
    files = [f for f in sorted(Path(a.results_dir).glob("*_scores.npz")) if pattern.search(f.name)]
    if not files:
        raise SystemExit("No matching score files.")
    max_fit = max(int(pattern.search(f.name).group(1)) for f in files)
    files = [f for f in files if int(pattern.search(f.name).group(1)) == max_fit]

    counts = Counter()
    n_good = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        labels, scores = d["test_labels"], d["test_scores"]
        n_good = int((labels == 0).sum())
        t_defect = conformal_threshold(d["cal_scores"], a.alpha_defect)
        flagged = np.where((labels == 0) & (scores > t_defect))[0]
        counts.update(flagged.tolist())

    print(f"{a.backbone} / {a.category}: {len(files)} seeds, {n_good} good test images")
    if not counts:
        print("No good test image was flagged in any seed.")
        return
    print("Seeds flagged | image")
    for idx, c in counts.most_common():
        print(f"{c:>13} | {paths[idx]}")


if __name__ == "__main__":
    main()
