"""Split-conformal thresholds and three-way decisions (ok / defect / unsure).

Calibration uses only normal images, so the guarantee covers false alarms:
for a new normal part exchangeable with the calibration parts,
P(score > t_defect) <= alpha_defect.
Missed defects are measured on the test set, not guaranteed.
"""
import numpy as np


def conformal_threshold(cal_scores, alpha):
    """Smallest threshold with a finite-sample false-alarm guarantee of alpha.

    Returns inf when the calibration set is too small for this alpha
    (requires alpha >= 1 / (n + 1)).
    """
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def three_way_decisions(scores, t_ok, t_defect):
    scores = np.asarray(scores, dtype=float)
    decisions = np.full(len(scores), "unsure", dtype=object)
    decisions[scores <= t_ok] = "ok"
    decisions[scores > t_defect] = "defect"
    return decisions


def summarize(decisions, labels):
    decisions = np.asarray(decisions)
    labels = np.asarray(labels)
    good, bad = labels == 0, labels == 1
    decided = decisions != "unsure"
    correct = ((decisions == "ok") & good) | ((decisions == "defect") & bad)
    return {
        "false_alarm_rate": float(np.mean(decisions[good] == "defect")) if good.any() else float("nan"),
        "miss_rate": float(np.mean(decisions[bad] == "ok")) if bad.any() else float("nan"),
        "unsure_rate": float(np.mean(~decided)),
        "unsure_rate_good": float(np.mean(decisions[good] == "unsure")) if good.any() else float("nan"),
        "unsure_rate_defect": float(np.mean(decisions[bad] == "unsure")) if bad.any() else float("nan"),
        "accuracy_on_decided": float(correct[decided].mean()) if decided.any() else float("nan"),
    }
