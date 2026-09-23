"""Evaluate conformal three-way decisions from a saved scores file.

Usage:
  python -m src.evaluate_decisions --scores results/patchcore_wrn50_screw_fit256_seed0_scores.npz
"""
import argparse
import json

import numpy as np

from src.conformal import conformal_threshold, summarize, three_way_decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--alpha-defect", type=float, default=0.05, help="target false-alarm rate")
    p.add_argument("--alpha-ok", type=float, default=0.30, help="share of normal parts allowed above the OK threshold")
    a = p.parse_args()

    d = np.load(a.scores, allow_pickle=True)
    cal, scores, labels = d["cal_scores"], d["test_scores"], d["test_labels"]
    n_cal = len(cal)

    if a.alpha_ok <= a.alpha_defect:
        raise SystemExit("--alpha-ok must be larger than --alpha-defect")

    t_defect = conformal_threshold(cal, a.alpha_defect)
    t_ok = conformal_threshold(cal, a.alpha_ok)

    binary = np.where(scores > t_defect, "defect", "ok")
    three_way = three_way_decisions(scores, t_ok, t_defect)

    report = {
        "scores_file": a.scores,
        "n_calib": n_cal,
        "min_achievable_alpha": round(1 / (n_cal + 1), 4),
        "alpha_defect": a.alpha_defect,
        "alpha_ok": a.alpha_ok,
        "t_ok": t_ok,
        "t_defect": t_defect,
        "single_threshold": summarize(binary, labels),
        "three_way": summarize(three_way, labels),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
