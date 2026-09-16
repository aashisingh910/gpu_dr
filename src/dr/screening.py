"""Screening-triage operating points.

A DR screening service does not need to name the grade.  It needs to decide,
safely, who can be sent home and who must see a clinician.  The metric that
governs that decision is **negative predictive value** - of everyone the system
clears, what fraction genuinely has no disease - together with how much
clinician workload the system removes.

    referral_rate      fraction of patients sent on to a human
    workload_reduction 1 - referral_rate, the point of the system
    NPV                safety of the "cleared" bucket
    missed             number of diseased patients in the cleared bucket

Thresholds are chosen on the **validation** split against a safety constraint
and then applied unchanged to test / external data.  Selecting a threshold on
the same data you report it on inflates NPV and is the most common way a
screening result becomes meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class OperatingPoint:
    task: str
    constraint: str
    target: float
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    referral_rate: float
    workload_reduction: float
    missed: int
    n: int
    prevalence: float

    def as_dict(self) -> dict:
        return asdict(self)


def confusion_at(y: np.ndarray, scores: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def metrics_at(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    tp, fp, tn, fn = confusion_at(y, scores, thr)
    n = len(y)
    return {
        "threshold": float(thr),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1) if (tp + fp) else float("nan"),
        "npv": tn / max(tn + fn, 1) if (tn + fn) else float("nan"),
        "referral_rate": (tp + fp) / n,
        "workload_reduction": (tn + fn) / n,
        "missed": fn,
        "n": n,
        "prevalence": float((y == 1).mean()),
    }


def select_operating_point(y: np.ndarray, scores: np.ndarray, task: str,
                           constraint: str, target: float) -> OperatingPoint:
    """Lowest referral rate that still satisfies the safety constraint.

    constraint: "sensitivity" (catch >= target of disease) or
                "npv"         (cleared bucket is >= target clean)
    """
    cands = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best = None
    for thr in cands:
        m = metrics_at(y, scores, thr)
        value = m[constraint]
        if not np.isfinite(value) or value < target:
            continue
        # among safe thresholds, prefer the one that refers fewest patients
        if best is None or m["referral_rate"] < best["referral_rate"]:
            best = m
    if best is None:                       # constraint unreachable -> refer all
        best = metrics_at(y, scores, 0.0)
        best["threshold"] = 0.0
    return OperatingPoint(task=task, constraint=constraint, target=target,
                          **{k: v for k, v in best.items()})


def apply_operating_point(y: np.ndarray, scores: np.ndarray, thr: float,
                          task: str, constraint: str, target: float) -> OperatingPoint:
    """Evaluate a *pre-selected* threshold on held-out data."""
    m = metrics_at(y, scores, thr)
    return OperatingPoint(task=task, constraint=constraint, target=target,
                          **{k: v for k, v in m.items()})


def triage(score: float, thresholds: dict) -> tuple[str, str]:
    """Three-way screening decision for a single eye.

    thresholds carries `refer` (sight-threatening / referable cut) and an
    optional `review` band below it for equivocal cases.
    """
    refer = thresholds["refer"]
    review = thresholds.get("review", refer * 0.6)
    if score >= refer:
        return "REFER", f"score {score:.3f} >= referral threshold {refer:.3f}"
    if score >= review:
        return "REVIEW", (f"score {score:.3f} sits in the equivocal band "
                          f"[{review:.3f}, {refer:.3f})")
    return "CLEAR", f"score {score:.3f} < review threshold {review:.3f}"


def screening_selection_score(y_grade: np.ndarray, class_probs: np.ndarray,
                              cfg) -> dict:
    """Score a checkpoint as a screening service rather than as a grader.

    Returns the fraction of patients the model can safely clear while holding
    the cleared bucket at `cfg.npv_target`, plus the ranking AUC that breaks
    ties between checkpoints that clear the same fraction.

    The threshold is fitted on the same split the score is read from, which is
    what makes this usable as an *in-training selection signal* - it is not a
    reportable number.  The held-out figure is produced by 09_screening.py,
    which fits on val and applies the frozen threshold to test.
    """
    k = cfg.task_grade
    y = (y_grade >= k).astype(int)
    scores = class_probs[:, k:].sum(1)

    if y.sum() == 0 or y.sum() == len(y):        # degenerate split, no signal
        return {"cleared": 0.0, "npv": float("nan"), "auc": float("nan"),
                "threshold": float("nan"), "missed": 0, "score": 0.0,
                "constraint_met": False}

    op = select_operating_point(y, scores, "sight_threatening", "npv",
                                cfg.npv_target)
    met = bool(np.isfinite(op.npv) and op.npv >= cfg.npv_target)
    cleared = op.workload_reduction if met else 0.0

    # rank-AUC of the binary task, computed without a sklearn dependency
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so a flat/degenerate score cannot inflate the AUC
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {"cleared": float(cleared), "npv": float(op.npv), "auc": auc,
            "threshold": float(op.threshold), "missed": int(op.missed),
            "constraint_met": met,
            "score": float(cleared + cfg.auc_tiebreak * auc)}
