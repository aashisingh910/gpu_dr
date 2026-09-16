"""The 38 evaluation parameters, grouped exactly as in the architecture figure.

  1. Diagnostic          11
  2. Ordinal              3
  3. Calibration          4
  4. Lesion / XAI         6
  5. Prognosis            4     (requires longitudinal data - see a7_temporal)
  6. Robustness           4
  7. Deployment           4
  + Subgroup / fairness   2
  ------------------------------
  total                  38

Every function takes plain numpy so it can be reused on any model's outputs,
which is what the 20+-model comparison harness in scripts/07 relies on.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)

GROUP_SIZES = {"diagnostic": 11, "ordinal": 3, "calibration": 4, "lesion_xai": 6,
               "prognosis": 4, "robustness": 4, "deployment": 4, "subgroup": 2}


# --------------------------------------------------------------------------
# 1. Diagnostic (11)
# --------------------------------------------------------------------------
def diagnostic_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    K = probs.shape[1]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        spec = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)
        ppv = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        npv = np.where((tn + fn) > 0, tn / (tn + fn), np.nan)

    present = np.unique(y_true)
    def _auc(yt, sc):
        return float(roc_auc_score(yt, sc)) if len(np.unique(yt)) > 1 else float("nan")

    try:
        auroc = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                    average="macro", labels=list(range(K)))) \
            if len(present) > 1 else float("nan")
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(
            np.eye(K)[y_true], probs, average="macro"))
    except ValueError:
        auprc = float("nan")

    # the two clinically decisive binary tasks
    ref_true = (y_true >= 2).astype(int)
    ref_score = probs[:, 2:].sum(1)
    stg_true = (y_true >= 3).astype(int)
    stg_score = probs[:, 3:].sum(1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_macro": float(np.nanmean(sens)),
        "specificity_macro": float(np.nanmean(spec)),
        "precision_macro": float(np.nanmean(ppv)),
        "npv_macro": float(np.nanmean(npv)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc_macro_ovr": auroc,
        "auprc_macro": auprc,
        "auc_referable_dr": _auc(ref_true, ref_score),
        "auc_sight_threatening": _auc(stg_true, stg_score),
    }


def referable_operating_point(y_true: np.ndarray, probs: np.ndarray,
                              min_sensitivity: float = 0.90) -> dict:
    """Sensitivity/specificity for referable DR at a clinically chosen threshold.

    Screening programmes fix sensitivity first (missing disease is the costly
    error) and accept whatever specificity follows.
    """
    yt = (np.asarray(y_true) >= 2).astype(int)
    sc = probs[:, 2:].sum(1)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan")}
    order = np.unique(sc)
    best = None
    for t in order:
        pred = (sc >= t).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= min_sensitivity and (best is None or spec > best[2]):
            best = (float(t), sens, spec)
    if best is None:
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan")}
    return {"threshold": best[0], "sensitivity": best[1], "specificity": best[2]}


# --------------------------------------------------------------------------
# 2. Ordinal (3)
# --------------------------------------------------------------------------
def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        qwk = float("nan")
    return {
        "quadratic_weighted_kappa": qwk,
        "mae_grade": float(np.abs(y_true - y_pred).mean()),
        "adjacent_accuracy": float((np.abs(y_true - y_pred) <= 1).mean()),
    }


# --------------------------------------------------------------------------
# 3. Calibration (4)
# --------------------------------------------------------------------------
def calibration_metrics(y_true: np.ndarray, probs: np.ndarray,
                        n_bins: int = 15) -> dict:
    y_true = np.asarray(y_true)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += (m.sum() / n) * gap
        mce = max(mce, gap)

    K = probs.shape[1]
    onehot = np.eye(K)[y_true]
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(n), y_true], 1e-12, 1.0)).mean())
    return {"ece": float(ece), "mce": float(mce), "brier": brier, "nll": nll}


# --------------------------------------------------------------------------
# 4. Lesion / XAI (6)
# --------------------------------------------------------------------------
def lesion_xai_metrics(dice: list[float], iou: list[float], pointing: list[float],
                       consistency: list[float], delta_p: list[float],
                       expert_auroc: float) -> dict:
    def _m(v):
        v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        return float(v.mean()) if v.size else float("nan")
    return {
        "lesion_attr_dice": _m(dice),
        "lesion_attr_iou": _m(iou),
        "pointing_game": _m(pointing),
        "attribution_consistency": _m(consistency),
        "counterfactual_delta_p": _m(delta_p),
        "lesion_expert_auroc": float(expert_auroc),
    }


# --------------------------------------------------------------------------
# 5. Prognosis (4)
# --------------------------------------------------------------------------
def prognosis_metrics(risk: np.ndarray | None, events: np.ndarray | None,
                      times: np.ndarray | None) -> dict:
    """Harrell's C-index + per-horizon AUC. Returns NaNs when no longitudinal
    data exists (which is the case for EyePACS)."""
    keys = ["c_index", "auc_12m", "auc_24m", "time_dependent_brier"]
    if risk is None or events is None or times is None:
        return {k: float("nan") for k in keys}

    num = den = 0
    for i in range(len(risk)):
        for j in range(len(risk)):
            if times[i] < times[j] and events[i] == 1:
                den += 1
                num += 1.0 if risk[i] > risk[j] else (0.5 if risk[i] == risk[j] else 0.0)
    c = num / den if den else float("nan")

    def _auc_at(h_idx):
        yt = (events == 1) & (times <= h_idx)
        if len(np.unique(yt)) < 2:
            return float("nan")
        return float(roc_auc_score(yt.astype(int), risk))

    brier = float(np.mean((risk - (events == 1).astype(float)) ** 2))
    return {"c_index": float(c), "auc_12m": _auc_at(1), "auc_24m": _auc_at(2),
            "time_dependent_brier": brier}


# --------------------------------------------------------------------------
# 6. Robustness (4)
# --------------------------------------------------------------------------
def robustness_metrics(clean_acc: float, corruption_accs: dict[str, float],
                       external_qwk: float, quality_stratified: dict[str, float]
                       ) -> dict:
    vals = [v for v in corruption_accs.values() if np.isfinite(v)]
    mca = float(np.mean(vals)) if vals else float("nan")
    rel = float(mca / clean_acc) if clean_acc > 0 and np.isfinite(mca) else float("nan")
    qs = [v for v in quality_stratified.values() if np.isfinite(v)]
    spread = float(max(qs) - min(qs)) if len(qs) > 1 else float("nan")
    return {
        "mean_corruption_accuracy": mca,
        "relative_robustness": rel,
        "external_domain_qwk": float(external_qwk),
        "quality_stratified_spread": spread,
    }


# --------------------------------------------------------------------------
# 7. Deployment (4)
# --------------------------------------------------------------------------
def deployment_metrics(size_mb: float, latency_ms: float, throughput: float,
                       student_retention: float) -> dict:
    return {
        "model_size_mb": float(size_mb),
        "latency_ms_per_image": float(latency_ms),
        "throughput_img_per_s": float(throughput),
        "distilled_accuracy_retention": float(student_retention),
    }


# --------------------------------------------------------------------------
# + Subgroup / fairness (2)
# --------------------------------------------------------------------------
def subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     groups: np.ndarray, min_n: int = 30) -> dict:
    accs = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_n:
            continue
        accs[str(g)] = float(accuracy_score(y_true[m], y_pred[m]))
    if not accs:
        return {"worst_subgroup_accuracy": float("nan"),
                "max_subgroup_gap": float("nan"), "_per_group": {}}
    v = list(accs.values())
    return {"worst_subgroup_accuracy": float(min(v)),
            "max_subgroup_gap": float(max(v) - min(v)), "_per_group": accs}


def count_reported(report: dict) -> int:
    """How many of the 38 parameters actually carry a finite value."""
    n = 0
    for group, vals in report.items():
        if not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                n += 1
    return n
