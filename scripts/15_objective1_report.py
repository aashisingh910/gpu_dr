#!/usr/bin/env python3
"""Step 15 - Objective 1 evidence pack: JSON + PDF.

Objective 1 claims the system detects DR across *all five severity grades*,
the minority grades included, and that the class-imbalance machinery (the
n^-0.5 sampler, hard-example mining, moderate class weighting) is what makes
that possible.  Steps 03/04 produce the per-grade numbers; this script turns
them into a self-contained evidence document - every measured value, the
decode experiment that explains them, the live training constants, and the
literature each design choice is answerable to.

Unlike Objective 2, the wording of Objective 1 is NOT recorded anywhere in the
repository.  It is reconstructed here from the two places the code refers to
it (scripts/13_linear_probe.py:125 and scripts/14_objective2_report.py:369),
and that reconstruction is flagged on the front page.  Correct it there if the
thesis wording differs.

    python scripts/15_objective1_report.py

Reads   outputs/<run>/evaluation.json, history.json, best.pt   (three runs)
        outputs/decode_ablation_{argmax,coral}.json
        data/cache/meta.csv            (live grade counts, not transcribed)
        src/dr/config.py               (live sampling constants)
Writes  outputs/objective1_evidence.json
        outputs/objective1_evidence.pdf
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.modules.a6_ordinal import coral_decode, coral_decode_numpy  # noqa: E402

GRADES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
MINORITY = (1, 2, 3)          # the grades Objective 1 turns on
INK, MUTED, RULE = "#16181d", "#5b6270", "#d4d7de"
GRADE_C = ["#c9cfda", "#7ba7d0", "#4a7fb5", "#2c5f8d", "#1f4e79"]

# Runs that carry a usable ordinal head, newest evidence last.
RUNS = [
    ("retfound_plus_laft_xai", "RETFound ViT-L/16, 4 stages (flagship)"),
    ("vits_objectives", "ViT-S/16 ImageNet, screening-selected"),
    ("retfound_stage1_evidence", "RETFound ViT-L/16, stage 1 only (aborted run)"),
]
ABLATION = [("outputs/decode_ablation_coral.json", "coral", "CORAL rank decode"),
            ("outputs/decode_ablation_argmax.json", "argmax", "argmax over class_probs")]

# --------------------------------------------------------------------------
# Literature this objective and its remedy are answerable to.
# `supports` states what the citation is used FOR; `value` carries a number
# only where that number is a stable, textbook-level constant.  Nothing here
# was re-derived in this session - verify before quoting in a thesis.
# --------------------------------------------------------------------------
LITERATURE = [
    dict(key="cao2020", cite="Cao, W., Mirjalili, V., Raschka, S. (2020). Rank consistent "
         "ordinal regression for neural networks with application to age estimation. "
         "Pattern Recognition Letters 140, 325-331.",
         supports="CORAL: a single shared projection z with K-1 learned biases b_k, giving "
                  "rank-monotone cumulative probabilities P(Y>k) = sigmoid(z + b_k). This is "
                  "the head under test (a6_ordinal.OrdinalHead).",
         value="K-1 = 4 biases for 5 grades; one shared weight vector",
         used_as="Defines both the decode rule and the failure geometry on page 4: every "
                 "interior grade owns exactly the z-interval between two adjacent b_k."),
    dict(key="niu2016", cite="Niu, Z. et al. (2016). Ordinal regression with multiple output "
         "CNN for age estimation. CVPR 2016, 4920-4928.",
         supports="The pre-CORAL multi-output formulation whose rank inconsistency CORAL's "
                  "shared weight vector exists to remove.",
         value=None,
         used_as="Why the head cannot simply be given per-grade weight vectors to widen the "
                 "interior bands - that reintroduces non-monotone cumulatives."),
    dict(key="frank2001", cite="Frank, E., Hall, M. (2001). A simple approach to ordinal "
         "classification. ECML 2001, LNCS 2167, 145-156.",
         supports="The ordinal-to-binary decomposition P(Y=k) = P(Y>k-1) - P(Y>k), which is "
                  "exactly the differencing step that bounds the interior classes.",
         value="P(Y=k) is a difference of two cumulatives",
         used_as="Source of the tanh(gap/4) bound derived on page 4."),
    dict(key="menon2021", cite="Menon, A. K. et al. (2021). Long-tail learning via logit "
         "adjustment. ICLR 2021.",
         supports="Adjusting decision logits by the log class prior is the principled "
                  "correction for long-tailed data, and is provably consistent for balanced "
                  "error - it acts on the decision boundary, not on the sampling.",
         value="logit adjustment by log(prior)",
         used_as="The remedy proposed on page 6: initialise b_k at logit(P(Y>k)) instead of a "
                 "uniform ladder. For CORAL the b_k ARE the adjustable logit offsets."),
    dict(key="lin2017", cite="Lin, T.-Y. et al. (2017). Focal loss for dense object detection. "
         "ICCV 2017, 2980-2988.",
         supports="Prior-based bias initialisation: setting the final bias so the model starts "
                  "at the observed base rate, which the paper reports is required for stable "
                  "training under extreme imbalance.",
         value="bias init b = -log((1-pi)/pi) at prior pi",
         used_as="Precedent for the same construction applied to the CORAL ladder; the b_k "
                 "here are currently initialised at a data-independent constant instead."),
    dict(key="kang2020", cite="Kang, B. et al. (2020). Decoupling representation and "
         "classifier for long-tailed recognition. ICLR 2020.",
         supports="Representation learning and classifier balancing are separable problems; "
                  "re-sampling helps the second far more than the first, and a mis-set "
                  "classifier is not recoverable by re-sampling.",
         value=None,
         used_as="Why the sampler in src/dr/sampling.py cannot fix a frozen decision ladder - "
                 "the finding on page 5."),
    dict(key="buda2018", cite="Buda, M., Maki, A., Mazurowski, M. A. (2018). A systematic "
         "study of the class imbalance problem in convolutional neural networks. "
         "Neural Networks 106, 249-259.",
         supports="Oversampling is the generally effective baseline for CNN class imbalance, "
                  "and does not cause the overfitting classically feared.",
         value=None,
         used_as="Justifies the n^power sampler being tried first; power = %s here."),
    dict(key="cui2019", cite="Cui, Y. et al. (2019). Class-balanced loss based on effective "
         "number of samples. CVPR 2019, 9268-9277.",
         supports="Re-weighting by effective number rather than raw inverse frequency; "
                  "stacking full inverse-frequency sampling on full class-balanced weighting "
                  "over-corrects.",
         value="weight proportional to (1-beta)/(1-beta^n)",
         used_as="The reason class_weight_power is a moderate %s rather than -1.0, recorded "
                 "in the module docstring of src/dr/sampling.py."),
    dict(key="shrivastava2016", cite="Shrivastava, A., Gupta, A., Girshick, R. (2016). "
         "Training region-based object detectors with online hard example mining. "
         "CVPR 2016, 761-769.",
         supports="Selecting training examples by their current loss, rather than by class "
                  "identity alone.",
         value=None,
         used_as="The H_i = L_i / mean(L) term of the sampler; eta = %s, EMA = %s."),
    dict(key="wilkinson2003", cite="Wilkinson, C. P. et al. (2003). Proposed international "
         "clinical diabetic retinopathy and diabetic macular edema disease severity scales. "
         "Ophthalmology 110(9), 1677-1682.",
         supports="The five-grade scale being predicted, and the definition of each grade. "
                  "Grade 1 is microaneurysms only; grade 3 is the 4-2-1 rule.",
         value="0 none, 1 mild NPDR, 2 moderate NPDR, 3 severe NPDR, 4 PDR",
         used_as="Defines what 'detect all severity grades' means, and why grades 1 and 3 "
                 "are the two that matter clinically."),
    dict(key="cohen1968", cite="Cohen, J. (1968). Weighted kappa: nominal scale agreement "
         "with provision for scaled disagreement or partial credit. "
         "Psychological Bulletin 70(4), 213-220.",
         supports="Quadratic weighted kappa, the standard DR grading metric, which charges "
                  "squared distance and so is only weakly penalised by never predicting an "
                  "interior grade.",
         value="QWK penalty proportional to (i-j)^2",
         used_as="Why QWK alone cannot detect the failure this document reports - page 3."),
    dict(key="krause2018", cite="Krause, J. et al. (2018). Grader variability and the "
         "importance of reference standards for evaluating machine learning models for "
         "diabetic retinopathy. Ophthalmology 125(8), 1264-1272.",
         supports="The reference standard is itself noisy, and adjacent-grade disagreement "
                  "between graders is common, so a 5-class ceiling well below 100% is "
                  "expected.",
         value=None,
         used_as="Framing for the target: Objective 1 asks for non-zero minority recall, "
                 "not for parity with grade 0."),
    dict(key="gulshan2016", cite="Gulshan, V. et al. (2016). Development and validation of a "
         "deep learning algorithm for detection of diabetic retinopathy in retinal fundus "
         "photographs. JAMA 316(22), 2402-2410.",
         supports="The binary referral framing under which this system is deployed, and "
                  "which remains met while Objective 1 fails.",
         value=None,
         used_as="Context for the separation on page 3: screening NPV is intact, per-grade "
                 "recall is not."),
    dict(key="zhou2023", cite="Zhou, Y. et al. (2023). A foundation model for generalizable "
         "disease detection from retinal images. Nature 622, 156-163. (RETFound)",
         supports="The backbone feeding the ordinal head in two of the three runs tabulated.",
         value="ViT-Large/16, MAE-pretrained on 1.6M retinal images",
         used_as="Identifies the flagship run; the failure reported here is downstream of "
                 "the backbone and reproduces on ViT-S as well."),
]


def resolve_literature(cfg) -> list:
    """Fill the %s placeholders from live config, so no constant is transcribed."""
    s = cfg.sampling
    args = {
        "buda2018": (s.power,),
        "cui2019": (s.class_weight_power,),
        "shrivastava2016": (s.hard_eta, s.hard_ema),
    }
    out = []
    for ref in LITERATURE:
        r = dict(ref)
        if ref["key"] in args:
            r["used_as"] = ref["used_as"] % args[ref["key"]]
        out.append(r)
    return out


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "n/a"
    return f"{v:.{nd}f}"


# ==========================================================================
#  Measurement 1 - the learned CORAL ladder in each checkpoint
# ==========================================================================
def ladder_from_state(sd) -> np.ndarray:
    """b_0 > b_1 > ... > b_{K-2}, reconstructed exactly as OrdinalHead does."""
    b0 = sd["ordinal.b0"].float().reshape(1)
    deltas = sd["ordinal.deltas"].float()
    steps = F.softplus(deltas)
    return torch.cat([b0, b0 - torch.cumsum(steps, 0)]).numpy()


def init_ladder(n_grades: int = 5) -> np.ndarray:
    """The ladder the head is BORN with: b0 = 0, deltas = 0.5 (a6_ordinal.py)."""
    b0 = torch.zeros(1)
    steps = F.softplus(torch.full((n_grades - 2,), 0.5))
    return torch.cat([b0, b0 - torch.cumsum(steps, 0)]).numpy()


def read_ladders() -> list:
    out = []
    for tag, desc in RUNS:
        ck_path = OUT_DIR / tag / "best.pt"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        thr = ladder_from_state(sd)
        init = init_ladder(len(thr) + 1)
        out.append({
            "run": tag, "description": desc,
            "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
            "stage": ck.get("stage") if isinstance(ck, dict) else None,
            "thresholds": [float(x) for x in thr],
            "gaps": [float(x) for x in np.diff(-thr)],
            "max_abs_drift_from_init": float(np.abs(thr - init).max()),
        })
    return out


# ==========================================================================
#  Measurement 2 - which grades each decode rule can emit, at any z
# ==========================================================================
def decode_reachability(thr: np.ndarray, z_lim: float = 10.0, n: int = 40001) -> dict:
    """Sweep the shared rank z and record what each decode rule outputs.

    This is a property of the head's parameters alone - no images, no data.
    A grade absent from `reachable` cannot be predicted for ANY input.
    """
    K = len(thr) + 1
    t = torch.as_tensor(thr, dtype=torch.float32)
    z = torch.linspace(-z_lim, z_lim, n).unsqueeze(1)
    cum = torch.sigmoid(z + t.unsqueeze(0))

    ones = torch.ones_like(cum[:, :1])
    zeros = torch.zeros_like(cum[:, :1])
    probs = (torch.cat([ones, cum], 1) - torch.cat([cum, zeros], 1)).clamp_min(1e-8)
    probs = probs / probs.sum(1, keepdim=True)
    p = probs.numpy().astype(np.float32)

    rules = {
        "coral_native": coral_decode(cum).numpy(),          # scripts/03_train.py:93
        "coral_roundtrip": coral_decode_numpy(p),           # scripts/04_evaluate.py:158
        "argmax": p.argmax(1),
    }
    zz = z.numpy().ravel()
    out = {"z_range": [-z_lim, z_lim], "n_grid": n, "rules": {}}
    for name, pred in rules.items():
        per = {}
        for g in range(K):
            m = pred == g
            per[GRADES[g]] = {
                "reachable": bool(m.any()),
                "share_of_z_axis": float(m.mean()),
                "z_interval": [float(zz[m].min()), float(zz[m].max())] if m.any() else None,
            }
        out["rules"][name] = {"reachable_grades": [int(g) for g in np.unique(pred)],
                              "per_grade": per}
    out["native_vs_roundtrip_disagreement"] = float(
        (rules["coral_native"] != rules["coral_roundtrip"]).mean())
    out["max_attainable_class_probability"] = {
        GRADES[g]: float(p[:, g].max()) for g in range(K)}
    out["interior_bound_tanh_gap_over_4"] = {
        GRADES[g + 1]: float(np.tanh(gap / 4.0))
        for g, gap in enumerate(np.diff(-thr))}
    out["_pred_grid"] = {k: v for k, v in rules.items()}      # dropped before JSON
    out["_z"] = zz
    return out


# ==========================================================================
#  Measurement 3 - the prior-matched ladder the data actually asks for
# ==========================================================================
def prior_ladder(counts: np.ndarray) -> dict:
    """b_k = logit P(Y>k) on the training distribution (Menon 2021, Lin 2017)."""
    counts = np.asarray(counts, float)
    n = counts.sum()
    tail = counts[::-1].cumsum()[::-1]          # tail[k] = #{y >= k}
    b, p = [], []
    for k in range(len(counts) - 1):
        pk = tail[k + 1] / n                    # P(Y > k)
        p.append(float(pk))
        b.append(float(np.log(pk / (1.0 - pk))))
    return {"train_counts": [int(c) for c in counts], "n_train": int(n),
            "p_y_gt_k": p, "thresholds": b, "gaps": [float(x) for x in np.diff(-np.array(b))]}


# ==========================================================================
#  Measured results from the recorded runs
# ==========================================================================
def per_grade_recall(cm: np.ndarray) -> list:
    tot = cm.sum(1)
    return [float(cm[i, i] / tot[i]) if tot[i] else float("nan") for i in range(len(tot))]


def read_run_results() -> list:
    out = []
    for tag, desc in RUNS:
        f = OUT_DIR / tag / "evaluation.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        meta = d.get("_meta", {})
        cm = np.array(meta.get("confusion_matrix", []), dtype=float)
        if cm.size == 0:
            continue
        out.append({
            "run": tag, "description": desc,
            "n_images": int(meta.get("n_images", cm.sum())),
            "n_patients": meta.get("n_patients"),
            "confusion_matrix": cm.astype(int).tolist(),
            "per_grade_recall": per_grade_recall(cm),
            "n_predicted_as": [int(x) for x in cm.sum(0)],
            "n_true": [int(x) for x in cm.sum(1)],
            "accuracy": d["diagnostic"]["accuracy"],
            "balanced_accuracy": d["diagnostic"]["balanced_accuracy"],
            "f1_macro": d["diagnostic"]["f1_macro"],
            "quadratic_weighted_kappa": d["ordinal"]["quadratic_weighted_kappa"],
            "auc_sight_threatening": d["diagnostic"]["auc_sight_threatening"],
            "minority_recall_mean": float(np.mean(
                [per_grade_recall(cm)[g] for g in MINORITY])),
        })
    return out


def read_decode_ablation() -> list:
    root = Path(__file__).resolve().parents[1]
    out = []
    for rel, key, label in ABLATION:
        f = root / rel
        if not f.exists():
            continue
        d = json.load(open(f))
        cm = np.array(d.get("_meta", {}).get("confusion_matrix", []), dtype=float)
        rec = per_grade_recall(cm) if cm.size else [float("nan")] * 5
        out.append({
            "decode": key, "label": label, "source": rel,
            "accuracy": d["diagnostic"]["accuracy"],
            "balanced_accuracy": d["diagnostic"]["balanced_accuracy"],
            "f1_macro": d["diagnostic"]["f1_macro"],
            "quadratic_weighted_kappa": d["ordinal"]["quadratic_weighted_kappa"],
            "adjacent_accuracy": d["ordinal"]["adjacent_accuracy"],
            "auc_sight_threatening": d["diagnostic"]["auc_sight_threatening"],
            "per_grade_recall": rec,
            "n_predicted_as": [int(x) for x in cm.sum(0)] if cm.size else None,
            "grades_never_predicted": [GRADES[i] for i in range(5)
                                       if cm.size and cm[:, i].sum() == 0],
        })
    return out


def read_val_history(tag: str = "vits_objectives") -> list:
    f = OUT_DIR / tag / "history.json"
    if not f.exists():
        return []
    return [{
        "epoch": e["epoch"], "stage": e["stage"],
        "val_accuracy": e["val_accuracy"],
        "val_qwk": e["val_quadratic_weighted_kappa"],
        "val_per_grade_recall": [e[f"val_recall_grade_{g}"] for g in range(5)],
        "val_screen_npv": e.get("val_screen_npv"),
        "val_screen_cleared": e.get("val_screen_cleared"),
        "mine_hardness_by_grade": [e.get(f"mine_hardness_grade_{g}") for g in range(5)],
    } for e in json.load(open(f))]


def read_grade_counts() -> dict:
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    tab = meta.groupby("split")["grade"].value_counts().unstack(fill_value=0)
    tab = tab.reindex(columns=range(5), fill_value=0)
    return {s: [int(v) for v in tab.loc[s]] for s in tab.index}


# ==========================================================================
#  JSON
# ==========================================================================
def build_json(cfg, ladders, runs, ablation, hist, counts, reach, prior) -> dict:
    s = cfg.sampling
    init = init_ladder()
    frozen = [l for l in ladders if l["max_abs_drift_from_init"] < 0.01]

    return {
        "document": {
            "title": "Objective 1 - evidence pack",
            "objective_as_reconstructed":
                "the system detects diabetic retinopathy across all five severity grades, "
                "the minority grades included, by means of the class-imbalance machinery "
                "(n^power sampling, hard-example mining, moderate class weighting)",
            "wording_provenance": "RECONSTRUCTED. Objective 1 is not stated verbatim anywhere "
                                  "in this repository. It is inferred from the two places the "
                                  "code refers to it: scripts/13_linear_probe.py:125 ('the "
                                  "loss re-weighting of objective 1') and "
                                  "scripts/14_objective2_report.py:369 (\"Objective 1's "
                                  "evidence, which currently fails (grade-1 recall 0.000)\"). "
                                  "Replace this field with the thesis wording before quoting.",
            "generated": date.today().isoformat(),
            "generator": "scripts/15_objective1_report.py",
            "status": "NOT MET. Across every recorded run and both decode rules, at least one "
                      "severity grade is predicted for zero images. The cause is located in "
                      "this document and is not the sampler: the CORAL decision ladder is "
                      "still at its initialisation, so the interior grades own fixed, narrow "
                      "intervals of the shared rank z.",
        },
        "claims_under_test": {
            "claim_A_all_grades_detected": {
                "statement": "Every one of the five severity grades is predicted with non-zero "
                             "recall on the held-out test split.",
                "verdict": "NOT SUPPORTED. Flagship run: Mild recall 0.000 (0 of 5268 images "
                           "predicted Mild). argmax decode: Mild AND Severe recall 0.000.",
            },
            "claim_B_imbalance_machinery_is_the_mechanism": {
                "statement": "The n^%s sampler, hard-example mining and class weighting are "
                             "what lift the minority grades." % s.power,
                "verdict": "NOT SUPPORTED, and shown to be unable to help while the ladder is "
                           "frozen: reachability is a property of the head parameters alone "
                           "(page 4), so no re-weighting of the data can restore a grade that "
                           "the decode rule cannot emit.",
            },
        },
        "method": {
            "measured_from": "recorded run artefacts (evaluation.json, history.json) plus a "
                             "parameter-only sweep of the ordinal head",
            "decode_rules": {
                "coral_native": "yhat = sum_k 1[P(Y>k) > 0.5] on the head's own cum_probs "
                                "(scripts/03_train.py:93, used for validation during training)",
                "coral_roundtrip": "the same rule, with P(Y>k) re-derived from class_probs "
                                   "(scripts/04_evaluate.py:158, used for all reported test "
                                   "numbers)",
                "argmax": "argmax over the differenced categorical class_probs",
            },
            "reachability_sweep": {
                "what": "z swept over %s on a %s-point grid; each decode rule applied to the "
                        "resulting cumulatives. A grade absent from the output cannot be "
                        "predicted for any image."
                        % (reach["z_range"], reach["n_grid"]),
                "ladder_used": "the learned thresholds of %s" % reach["_run"],
            },
            "grade_counts": counts,
            "sampling_config_live": {
                "power": s.power, "hard_eta": s.hard_eta, "hard_ema": s.hard_ema,
                "hard_clip": s.hard_clip, "boundary_bonus": s.boundary_bonus,
                "lowconf_bonus": s.lowconf_bonus,
                "class_weight_power": s.class_weight_power,
                "source": "src/dr/config.py SamplingCfg, read live",
            },
            "selection_objective": "0.5*QWK + 0.3*MacroF1 + 0.2*minority recall "
                                   "(scripts/03_train.py:74)",
        },
        "results": {
            "per_run": runs,
            "decode_ablation": ablation,
            "validation_history": hist,
        },
        "ladders": {
            "initialisation": {
                "b0": 0.0, "deltas": 0.5,
                "thresholds": [float(x) for x in init],
                "gaps": [float(x) for x in np.diff(-init)],
                "note": "softplus(0.5) = %.5f, so the head is born with a uniform ladder "
                        "of that spacing and b0 = 0." % float(F.softplus(torch.tensor(0.5))),
            },
            "learned": ladders,
            "runs_still_at_initialisation": [l["run"] for l in frozen],
            "prior_matched": prior,
        },
        "reachability": {k: v for k, v in reach.items() if not k.startswith("_")},
        "headline_numbers": {
            "grades_never_predicted_argmax": next(
                (a["grades_never_predicted"] for a in ablation if a["decode"] == "argmax"), None),
            "flagship_minority_recall": next(
                (r["per_grade_recall"] for r in runs
                 if r["run"] == "retfound_plus_laft_xai"), None),
            "interior_share_of_z_axis": {
                g: reach["rules"]["coral_native"]["per_grade"][g]["share_of_z_axis"]
                for g in GRADES[1:4]},
            "max_ladder_drift_from_init": {l["run"]: l["max_abs_drift_from_init"]
                                           for l in ladders},
            "native_vs_roundtrip_disagreement": reach["native_vs_roundtrip_disagreement"],
        },
        "open_discrepancy": {
            "what": "The same checkpoint reports Mild recall 0.00 on validation during "
                    "training and 0.531 on test at evaluation.",
            "runs": "vits_objectives, best.pt at epoch 3",
            "ruled_out": "The two decode paths were tested against each other on a %s-point z "
                         "grid and agree exactly (%.4f%% disagreement), so the round-trip "
                         "through class_probs is NOT the cause."
                         % (reach["n_grid"], 100 * reach["native_vs_roundtrip_disagreement"]),
            "still_open": "Validation used a 600-image subset (--max-val 600) of the 920-image "
                          "val split; test is 926 images. Whether that, or something else, "
                          "explains a swing this large is unresolved.",
            "consequence": "The non-zero minority recalls in the vits_objectives test row are "
                           "NOT safe to quote until this is reconciled.",
        },
        "limitations": [
            "The objective wording is reconstructed, not quoted - see document.wording_provenance.",
            "Two of the three runs are short (3-4 epochs, one aborted in stage 2), so the "
            "ladder had limited opportunity to move; the drift measurement states how much it "
            "moved, not how much it would move given the full 45-epoch schedule.",
            "The reachability sweep is exact for the head, but says nothing about which z the "
            "backbone actually produces - it bounds what is possible, not what is likely.",
            "The prior-matched ladder is a proposed remedy computed from the training "
            "distribution. It has NOT been trained or evaluated; no claim is made that it "
            "fixes the objective.",
            "Single seed throughout. No confidence intervals on the per-grade recalls.",
        ],
        "reproduce": {
            "this_document": "python scripts/15_objective1_report.py",
            "machine_readable_twin": "outputs/objective1_evidence.json",
            "decode_ablation": "python scripts/04_evaluate.py --ckpt outputs/vits_objectives/"
                               "best.pt --decode {coral,argmax}",
            "training": "see run_50pct.sh and run_pipeline.sh",
        },
        "literature": resolve_literature(cfg),
    }


# ==========================================================================
#  PDF
# ==========================================================================
class Doc:
    W, H = 8.27, 11.69

    def __init__(self, pdf):
        self.pdf = pdf
        self.page = 0

    def new(self, title, sub=None):
        self.fig = plt.figure(figsize=(self.W, self.H), facecolor="white")
        self.page += 1
        self.y = 0.945
        self.fig.text(0.075, 0.965, title, size=15, weight="bold", color=INK, va="bottom")
        if sub:
            self.fig.text(0.075, 0.952, sub, size=8.5, color=MUTED, va="top")
            self.y = 0.925
        self.fig.lines.append(plt.Line2D([0.075, 0.925], [self.y, self.y],
                                         color=RULE, lw=0.8, transform=self.fig.transFigure))
        self.y -= 0.028
        return self.fig

    def close(self):
        self.fig.text(0.925, 0.035, f"{self.page}", size=8, color=MUTED, ha="right")
        self.fig.text(0.075, 0.035, "RETFound Plus-LAFT-XAI  ·  Objective 1 evidence pack",
                      size=7.5, color=MUTED)
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def h2(self, t, gap=0.020):
        self.y -= gap
        self.fig.text(0.075, self.y, t, size=10.5, weight="bold", color=INK)
        self.y -= 0.016

    def para(self, t, size=8.6, mono=False, color=INK, indent=0.0, lead=0.0145,
             wrap=True):
        """Lines already carrying \\n are respected; anything longer than the
        text block is wrapped, so strings pulled straight out of the JSON
        cannot run off the right margin."""
        lines = t.split("\n")
        if wrap:
            # usable width 0.075..0.925 = 6.85 in, minus the indent
            factor = 0.60 if mono else 0.55
            budget = max(20, int(((6.85 - indent * 8.27) * 72) / (factor * size)))
            out = []
            for line in lines:
                if len(line) <= budget:
                    out.append(line)
                else:
                    out.extend(textwrap.wrap(line, budget) or [""])
            lines = out
        for line in lines:
            self.fig.text(0.075 + indent, self.y, line, size=size, color=color,
                          family="monospace" if mono else "sans-serif", va="top")
            self.y -= lead
        self.y -= 0.004

    def kv(self, rows, w=0.34, size=8.4):
        for k, v in rows:
            self.fig.text(0.078, self.y, k, size=size, color=MUTED, va="top")
            self.fig.text(0.078 + w, self.y, str(v), size=size, color=INK, va="top",
                          family="monospace")
            self.y -= 0.0155
        self.y -= 0.004

    def table(self, cols, rows, widths, size=8.0, head_size=7.6, row_h=0.0175,
              mono_from=None, hl=None):
        """hl: set of row indices to print in ink-bold-ish (used for failures)."""
        x0 = 0.075
        xs, acc = [], x0
        for w in widths:
            xs.append(acc); acc += w
        for c, x in zip(cols, xs):
            self.fig.text(x, self.y, c, size=head_size, weight="bold", color=MUTED, va="top")
        self.y -= 0.016
        self.fig.lines.append(plt.Line2D([x0, acc - 0.01], [self.y + 0.004, self.y + 0.004],
                                         color=RULE, lw=0.7, transform=self.fig.transFigure))
        self.y -= 0.004
        hl = hl or set()
        for i, r in enumerate(rows):
            for j, (cell, x) in enumerate(zip(r, xs)):
                self.fig.text(x, self.y, str(cell), size=size,
                              color=INK, va="top",
                              weight="bold" if i in hl else "normal",
                              family="monospace" if (mono_from is not None and j >= mono_from)
                              else "sans-serif")
            self.y -= row_h
        self.y -= 0.006

    def axes(self, h, left=0.115, width=0.80, top_gap=0.040, bottom_gap=0.050):
        top = self.y - top_gap
        bottom = top - h
        ax = self.fig.add_axes([left, bottom, width, h])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
        self.y = bottom - bottom_gap
        return ax


def build_pdf(J, reach, path: Path):
    doc = J["document"]
    runs = J["results"]["per_run"]
    abl = J["results"]["decode_ablation"]
    lads = J["ladders"]

    with PdfPages(path) as pdf:
        D = Doc(pdf)

        # ---------------- page 1: verdict -----------------------------------
        D.new("Objective 1 — detection across all five severity grades",
              "Evidence pack · generated %s · RETFound Plus–LAFT–XAI" % doc["generated"])
        D.para("Objective as reconstructed: " + doc["objective_as_reconstructed"],
               size=9, color=MUTED)
        D.para("", size=4)
        D.para("WORDING IS RECONSTRUCTED, NOT QUOTED. Objective 1 appears nowhere verbatim in\n"
               "this repository. It is inferred from scripts/13_linear_probe.py:125 and\n"
               "scripts/14_objective2_report.py:369. Replace it before thesis submission.",
               size=8, color="#8a5a1b")

        D.h2("Verdict")
        D.para("NOT MET. In every recorded run, and under both decode rules, at least one of\n"
               "the five severity grades is predicted for exactly zero images.\n"
               "\n"
               "The cause is identified here and it is not the class-imbalance machinery.\n"
               "The CORAL decision ladder is still, after training, at the value it was\n"
               "initialised with. Because that ladder is a uniform one, each interior grade\n"
               "owns a fixed and narrow interval of the shared rank z — and reachability is a\n"
               "property of those parameters alone, independent of the data. No amount of\n"
               "re-sampling or loss re-weighting can restore a grade the decode rule is\n"
               "structurally unable to emit.", size=9)

        D.h2("Headline")
        flag = next(r for r in runs if r["run"] == "retfound_plus_laft_xai")
        argm = next((a for a in abl if a["decode"] == "argmax"), None)
        D.kv([
            ("flagship run, Mild recall", "%.3f  (0 of %d images predicted Mild)"
             % (flag["per_grade_recall"][1], flag["n_images"])),
            ("flagship run, Severe recall", "%.3f" % flag["per_grade_recall"][3]),
            ("argmax decode, grades never emitted",
             ", ".join(argm["grades_never_predicted"]) if argm else "-"),
            ("interior grades, share of z-axis",
             "  ".join("%s %.1f%%" % (g, 100 * v)
                       for g, v in J["headline_numbers"]["interior_share_of_z_axis"].items())),
            ("ladder drift from init, ViT-S run",
             "%.2e  (init value 0.97408)" % lads["learned"][1]["max_abs_drift_from_init"]
             if len(lads["learned"]) > 1 else "-"),
            ("screening NPV, unaffected", "intact — see page 3"),
        ], w=0.36)

        D.h2("What this document contains")
        D.para("p2  the objective, its provenance, and what was measured\n"
               "p3  measured per-grade recall, every run and both decode rules\n"
               "p4  the reachability experiment — which grades each rule CAN emit\n"
               "p5  the learned ladders, and the finding that they never moved\n"
               "p6  the prior-matched ladder the training distribution asks for\n"
               "p7  an unresolved validation/test discrepancy, stated as unresolved\n"
               "p8  limitations and reproduction\n"
               "p9  literature", size=8.6)
        D.close()

        # ---------------- page 2: what was measured -------------------------
        D.new("What was measured", "Objective 1 · method")
        D.h2("The two claims, separated")
        for k, v in J["claims_under_test"].items():
            D.para(k.replace("claim_", "").replace("_", " ").upper(), size=8.6, color=MUTED)
            D.para(v["statement"], size=8.6, indent=0.012)
            D.para(v["verdict"], size=8.6, indent=0.012, color="#8a2b2b")
            D.para("", size=4)

        D.h2("Decode rules under test")
        for k, v in J["method"]["decode_rules"].items():
            D.para(k, size=7.8, mono=True, color=MUTED)
            D.para(v, size=7.8, mono=True, indent=0.022, lead=0.0132)

        D.h2("Grade counts (live from data/cache/meta.csv)")
        cts = J["method"]["grade_counts"]
        D.table(["split"] + GRADES + ["total"],
                [[s] + [str(c) for c in cts[s]] + [str(sum(cts[s]))] for s in
                 ("train", "val", "test") if s in cts],
                [0.10, 0.11, 0.10, 0.12, 0.10, 0.09, 0.10], mono_from=1)
        tr = cts.get("train", [0] * 5)
        ntr = max(sum(tr), 1)
        D.para("Grade %s is %.2f%% of train and grade %s is %.2f%% - the two rarest. This is "
               "the imbalance the sampler exists to correct."
               % (GRADES[4], 100 * tr[4] / ntr, GRADES[3], 100 * tr[3] / ntr),
               size=8.2, color=MUTED)

        D.h2("Class-imbalance machinery under test (live from src/dr/config.py)")
        sc = J["method"]["sampling_config_live"]
        D.para("P_i  proportional to  n_class^power  *  min((1 + H_i)^eta, clip)", size=8.4, mono=True)
        D.kv([("power", sc["power"]), ("hard_eta", sc["hard_eta"]),
              ("hard_ema", sc["hard_ema"]), ("hard_clip", sc["hard_clip"]),
              ("boundary_bonus", sc["boundary_bonus"]),
              ("lowconf_bonus", sc["lowconf_bonus"]),
              ("class_weight_power", sc["class_weight_power"])], w=0.30)
        D.para("Model selection: " + J["method"]["selection_objective"], size=8.2, color=MUTED)
        D.close()

        # ---------------- page 3: measured results --------------------------
        D.new("Result — per-grade recall", "Objective 1 · measured on held-out test splits")
        D.h2("By run (CORAL rank decode, as reported)")
        rows, hl = [], set()
        for i, r in enumerate(runs):
            rec = r["per_grade_recall"]
            rows.append([r["run"][:22], str(r["n_images"])]
                        + ["%.3f" % x for x in rec]
                        + ["%.3f" % r["quadratic_weighted_kappa"]])
            if min(rec) == 0.0:
                hl.add(i)
        D.table(["run", "n"] + GRADES + ["QWK"], rows,
                [0.20, 0.06, 0.075, 0.075, 0.085, 0.075, 0.065, 0.07], mono_from=1, hl=hl)
        D.para("Bold = at least one grade with recall exactly zero.", size=7.8, color=MUTED)

        D.h2("By decode rule (same checkpoint, vits_objectives epoch 3)")
        rows = []
        for a in abl:
            rows.append([a["label"][:26]] + ["%.3f" % x for x in a["per_grade_recall"]]
                        + ["%.3f" % a["accuracy"], "%.3f" % a["quadratic_weighted_kappa"]])
        D.table(["decode"] + GRADES + ["acc", "QWK"], rows,
                [0.19, 0.075, 0.075, 0.085, 0.075, 0.065, 0.065, 0.065], mono_from=1)
        D.para("The AUCs are identical across the two rows (sight-threatening %.4f), because\n"
               "the scores are the same and only the decision rule differs. Accuracy and QWK\n"
               "are not — argmax buys both by never emitting two of the five grades."
               % abl[0]["auc_sight_threatening"], size=8.2, color=MUTED)

        D.h2("Predictions issued per grade — the failure in raw counts")
        rows = []
        for a in abl:
            rows.append([a["label"][:26]] + [str(x) for x in (a["n_predicted_as"] or [])])
        for r in runs[:1]:
            rows.append([r["run"][:26]] + [str(x) for x in r["n_predicted_as"]])
        D.table(["run / decode"] + GRADES, rows,
                [0.22, 0.10, 0.10, 0.11, 0.10, 0.10], mono_from=1)

        ax = D.axes(0.155, bottom_gap=0.072)
        w, xs = 0.8 / max(len(abl) + 1, 1), np.arange(5)
        series = [(a["label"][:22], a["per_grade_recall"]) for a in abl]
        series.append((runs[0]["run"][:22], runs[0]["per_grade_recall"]))
        for i, (lab, rec) in enumerate(series):
            ax.bar(xs + i * w - 0.4 + w / 2, rec, width=w * 0.92,
                   color=GRADE_C[[1, 3, 4][i % 3]], label=lab)
        ax.set_xticks(xs); ax.set_xticklabels(GRADES, fontsize=7.5)
        ax.set_ylabel("recall", fontsize=8, color=MUTED)
        ax.set_ylim(0, 1.0)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.legend(fontsize=6.6, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13))
        ax.set_title("per-grade recall — zero bars are the objective failing",
                     fontsize=8.5, color=INK, pad=6)

        D.h2("What is NOT failing")
        D.para("Sight-threatening AUC is %.4f on the flagship run and the screening triage\n"
               "meets its NPV constraint. Objective 1 is a per-grade claim, and the binary\n"
               "referral pathway can be intact while it fails — those are different questions\n"
               "(Gulshan 2016 vs Wilkinson 2003)."
               % flag["auc_sight_threatening"], size=8.6)
        D.close()

        # ---------------- page 4: reachability ------------------------------
        D.new("Why re-weighting cannot fix it", "Objective 1 · reachability of the decode rules")
        D.para("This experiment uses NO images. It sweeps the shared CORAL rank z over %s on a\n"
               "%s-point grid and asks, of each decode rule, which grades it is capable of\n"
               "emitting. A grade absent from the output cannot be predicted for any input,\n"
               "at any point in training, under any sampling scheme."
               % (reach["z_range"], reach["n_grid"]), size=8.8)
        D.para("Ladder used: the learned thresholds of %s." % reach["_run"],
               size=8.2, color=MUTED)

        D.h2("Share of the z-axis owned by each grade")
        rows = []
        for name in ("coral_native", "coral_roundtrip", "argmax"):
            pg = reach["rules"][name]["per_grade"]
            rows.append([name] + ["%.2f%%" % (100 * pg[g]["share_of_z_axis"])
                                  if pg[g]["reachable"] else "UNREACHABLE" for g in GRADES])
        D.table(["decode rule"] + GRADES, rows,
                [0.19, 0.10, 0.12, 0.115, 0.11, 0.10], mono_from=1, hl={2})

        D.h2("The bound that makes argmax impossible")
        D.para("With one shared rank z and thresholds b_k, the interior classes are differences\n"
               "of sigmoids (Frank & Hall 2001):\n"
               "\n"
               "    P(Y=k) = sigmoid(z + b_{k-1}) - sigmoid(z + b_k)   <=   tanh((b_{k-1}-b_k)/4)\n"
               "\n"
               "so an interior class can never exceed that bound, while the two endpoint\n"
               "classes are unbounded and reach 1.0. At the learned gaps:", size=8.6, mono=False)
        rows = [[g, "%.3f" % v, "%.3f" % reach["max_attainable_class_probability"][g]]
                for g, v in reach["interior_bound_tanh_gap_over_4"].items()]
        D.table(["interior grade", "bound tanh(gap/4)", "max P observed on sweep"], rows,
                [0.22, 0.22, 0.26], mono_from=1)
        D.para("Every interior class is capped near 0.24 while grades 0 and 4 reach 1.0.\n"
               "That is why argmax emits only %s."
               % ", ".join(GRADES[i] for i in reach["rules"]["argmax"]["reachable_grades"]),
               size=8.6)

        # decode map figure
        ax = D.axes(0.115, left=0.165, width=0.75, top_gap=0.045, bottom_gap=0.055)
        grid = np.stack([reach["_pred_grid"][k] for k in
                         ("coral_native", "coral_roundtrip", "argmax")])
        zz = reach["_z"]
        cmap = matplotlib.colors.ListedColormap(GRADE_C)
        ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=4,
                  extent=[zz[0], zz[-1], 3, 0])
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(["CORAL native", "CORAL round-trip", "argmax"], fontsize=7.2)
        ax.set_xlabel("shared CORAL rank  z", fontsize=8, color=MUTED)
        ax.set_title("grade emitted as a function of z — argmax has no Mild or Severe band",
                     fontsize=8.5, color=INK, pad=6)
        ax.legend(handles=[Patch(facecolor=GRADE_C[i], label=GRADES[i]) for i in range(5)],
                  fontsize=6.4, frameon=False, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30))

        D.h2("A hypothesis this rules out")
        D.para("The two CORAL paths differ in implementation: training reads the head's own\n"
               "cum_probs, evaluation re-derives them from class_probs. Tested against each\n"
               "other on the same grid they agree exactly — disagreement %.4f%%. The\n"
               "round-trip is not lossy and is not the cause of anything in this document."
               % (100 * reach["native_vs_roundtrip_disagreement"]), size=8.6)
        D.close()

        # ---------------- page 5: the ladders -------------------------------
        D.new("The decision ladder never moved", "Objective 1 · learned CORAL thresholds")
        ini = lads["initialisation"]
        D.para("The head is initialised with b0 = 0 and deltas = 0.5, and the thresholds are\n"
               "b_0 = b0, b_k = b_{k-1} - softplus(delta_k). softplus(0.5) = %.5f, so the head\n"
               "is born with a UNIFORM ladder of that spacing."
               % float(F.softplus(torch.tensor(0.5))), size=8.8)

        D.h2("Learned thresholds, per checkpoint")
        rows = [["initialisation", "-", "-"]
                + ["%+.3f" % x for x in ini["thresholds"]] + ["0"]]
        hl = set()
        for i, l in enumerate(lads["learned"], start=1):
            rows.append([l["run"][:21], str(l["epoch"]), str(l["stage"] or "-")]
                        + ["%+.3f" % x for x in l["thresholds"]]
                        + ["%.1e" % l["max_abs_drift_from_init"]])
            if l["max_abs_drift_from_init"] < 0.01:
                hl.add(i)
        D.table(["run", "ep", "stage", "b_0", "b_1", "b_2", "b_3", "drift"], rows,
                [0.205, 0.038, 0.078, 0.072, 0.072, 0.072, 0.072, 0.07],
                mono_from=1, hl=hl, size=7.6)
        D.para("Bold = still at initialisation (max drift < 0.01). Two of three runs match the\n"
               "initialised ladder to four decimal places; the gradient reaching b0 and deltas\n"
               "is real but is three orders of magnitude smaller than the movement required.",
               size=8.2, color=MUTED)

        D.h2("Gap widths — the z-interval each interior grade owns")
        rows = [["initialisation"] + ["%.3f" % g for g in ini["gaps"]]]
        for l in lads["learned"]:
            rows.append([l["run"][:22]] + ["%.3f" % g for g in l["gaps"]])
        rows.append(["prior-matched (p6)"] + ["%.3f" % g for g in lads["prior_matched"]["gaps"]])
        D.table(["ladder", "Mild band", "Moderate band", "Severe band"], rows,
                [0.24, 0.15, 0.16, 0.15], mono_from=1)

        ax = D.axes(0.135, left=0.115, width=0.80, top_gap=0.045, bottom_gap=0.058)
        show = [("prior-matched", lads["prior_matched"]["thresholds"]),
                ("learned (ViT-L)", lads["learned"][0]["thresholds"]),
                ("initialisation", ini["thresholds"])]
        lo, hi = -6.0, 4.0
        for row, (lab, thr) in enumerate(show):
            edges = [lo] + list(-np.array(thr)) + [hi]
            for g in range(5):
                a, b = edges[g], edges[g + 1]
                ax.barh(row, b - a, left=a, height=0.55, color=GRADE_C[g],
                        edgecolor="white", lw=0.7)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels([s[0] for s in show], fontsize=7.4)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("shared CORAL rank  z   (band = the grade decoded at that z)",
                      fontsize=8, color=MUTED)
        ax.set_title("decision bands: uniform at init, prior-matched puts them where the data is",
                     fontsize=8.3, color=INK, pad=6)
        ax.legend(handles=[Patch(facecolor=GRADE_C[i], label=GRADES[i]) for i in range(5)],
                  fontsize=6.4, frameon=False, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.32))
        D.close()

        # ---------------- page 6: the remedy --------------------------------
        D.new("The ladder the data asks for", "Objective 1 · proposed remedy, not yet trained")
        pm = lads["prior_matched"]
        D.para("Logit adjustment by the log class prior is the standard correction for\n"
               "long-tailed decision boundaries (Menon 2021), and prior-based bias\n"
               "initialisation is the standard fix for minority collapse at the start of\n"
               "training (Lin 2017). In a CORAL head the b_k ARE the adjustable offsets, so\n"
               "both amount to one change: initialise b_k at logit P(Y>k) on the training\n"
               "distribution instead of at a data-independent constant.", size=8.8)

        D.h2("Computed from the live training counts")
        rows = []
        for k in range(4):
            rows.append(["P(Y > %d)" % k, "%.4f" % pm["p_y_gt_k"][k],
                         "%+.3f" % pm["thresholds"][k],
                         "%+.3f" % ini["thresholds"][k]])
        D.table(["cumulative", "prior", "prior-matched b_k", "current b_k"], rows,
                [0.16, 0.13, 0.19, 0.16], mono_from=1)
        D.para("train counts %s, n = %d" % (pm["train_counts"], pm["n_train"]),
               size=8.0, color=MUTED, mono=True)

        D.h2("What changes")
        rows = []
        for i, g in enumerate(GRADES[1:4]):
            cur, new = ini["gaps"][i], pm["gaps"][i]
            rows.append([g, "%.3f" % cur, "%.3f" % new, "%.2fx" % (new / cur)])
        D.table(["interior grade", "current band", "prior-matched band", "ratio"], rows,
                [0.20, 0.16, 0.20, 0.12], mono_from=1)
        D.para("Two errors in opposite directions: the Mild band is currently far too wide for\n"
               "its prior, and the Moderate band far too narrow. And b_0 = %+.3f means the\n"
               "model must reach z > %+.3f before it calls any DR at all, when the base rate\n"
               "puts that boundary at z = %+.3f."
               % (ini["thresholds"][0], -ini["thresholds"][0] + 0.0,
                  -pm["thresholds"][0] + 0.0), size=8.6)

        D.h2("Status of this remedy")
        D.para("PROPOSED, NOT TESTED. It has not been trained or evaluated. It is recorded here\n"
               "because it follows from the measurement, not because it is known to work. The\n"
               "companion change — giving ordinal.b0 and ordinal.deltas their own, higher\n"
               "learning-rate parameter group so the ladder can move during training — is\n"
               "equally untested.", size=8.6, color="#8a5a1b")
        D.close()

        # ---------------- page 7: open discrepancy --------------------------
        D.new("Unresolved", "Objective 1 · a discrepancy that is not yet explained")
        od = J["open_discrepancy"]
        D.para(od["what"], size=9)
        D.para("Run: %s" % od["runs"], size=8.4, color=MUTED)

        D.h2("Validation, per epoch (CORAL native decode, during training)")
        rows = []
        for h in J["results"]["validation_history"]:
            rows.append([str(h["epoch"]), h["stage"][:10], "%.3f" % h["val_accuracy"],
                         "%.3f" % h["val_qwk"]]
                        + ["%.2f" % x for x in h["val_per_grade_recall"]])
        D.table(["ep", "stage", "acc", "QWK"] + GRADES, rows,
                [0.04, 0.10, 0.065, 0.065, 0.065, 0.06, 0.075, 0.065, 0.055],
                mono_from=2, size=7.6)

        D.h2("Test, same checkpoint (CORAL round-trip decode, at evaluation)")
        cor = next((a for a in abl if a["decode"] == "coral"), None)
        if cor:
            D.table(["split", "acc", "QWK"] + GRADES,
                    [["test (926)", "%.3f" % cor["accuracy"],
                      "%.3f" % cor["quadratic_weighted_kappa"]]
                     + ["%.2f" % x for x in cor["per_grade_recall"]]],
                    [0.11, 0.065, 0.065, 0.065, 0.06, 0.075, 0.065, 0.055],
                    mono_from=1, size=7.6)

        D.h2("Ruled out")
        D.para(od["ruled_out"], size=8.6)
        D.h2("Still open")
        D.para(od["still_open"], size=8.6)
        D.h2("Consequence")
        D.para(od["consequence"], size=8.8, color="#8a2b2b")
        D.para("\nThe validation rows show the SAME failure as argmax — Mild and Severe at 0.00\n"
               "at every epoch — which is consistent with everything on pages 4 and 5. The test\n"
               "row is the outlier, and it is the row that would be quoted. It should not be,\n"
               "until the two are reconciled.", size=8.6)
        D.close()

        # ---------------- page 8: limitations -------------------------------
        D.new("Limitations and reproduction", "Objective 1 · what this document does not show")
        D.h2("Limitations")
        for i, l in enumerate(J["limitations"], 1):
            D.para("%d. %s" % (i, l), size=8.6, lead=0.0138)
            D.para("", size=3)
        D.h2("Reproduce")
        D.kv(list(J["reproduce"].items()), w=0.30, size=8.0)
        D.h2("Artefacts read")
        D.para("outputs/<run>/evaluation.json     per-grade confusion matrices\n"
               "outputs/<run>/history.json        per-epoch validation recalls\n"
               "outputs/<run>/best.pt             ordinal.b0, ordinal.deltas\n"
               "outputs/decode_ablation_*.json    argmax vs CORAL on one checkpoint\n"
               "data/cache/meta.csv               grade counts per split\n"
               "src/dr/config.py                  SamplingCfg, read live", size=7.8, mono=True)
        D.close()

        # ---------------- page 9+: literature -------------------------------
        lit = J["literature"]
        per_page = 5
        for start in range(0, len(lit), per_page):
            D.new("Literature" if start == 0 else "Literature (cont.)",
                  "Objective 1 · what each design choice is answerable to")
            for ref in lit[start:start + per_page]:
                D.para(ref["cite"], size=8.2)
                D.para("supports:  " + ref["supports"], size=7.8, color=MUTED, indent=0.012)
                if ref.get("value"):
                    D.para("value:     " + str(ref["value"]), size=7.8, color=MUTED,
                           indent=0.012, mono=True)
                D.para("used as:   " + ref["used_as"], size=7.8, color=INK, indent=0.012)
                D.para("", size=4)
            D.close()

        info = pdf.infodict()
        info["Title"] = "Objective 1 evidence pack - detection across all severity grades"
        info["Subject"] = doc["status"]


# ==========================================================================
def main() -> None:
    cfg = default_config()

    print("[read] checkpoints ...")
    ladders = read_ladders()
    if not ladders:
        raise SystemExit("no checkpoints with an ordinal head found under outputs/")
    print("[read] evaluation artefacts ...")
    runs = read_run_results()
    abl = read_decode_ablation()
    hist = read_val_history()
    counts = read_grade_counts()

    # reachability is a property of one ladder; use the run the ablation was run on
    ref_run = "vits_objectives"
    lad = next((l for l in ladders if l["run"] == ref_run), ladders[0])
    print("[sweep] decode reachability on %s ..." % lad["run"])
    reach = decode_reachability(np.array(lad["thresholds"]))
    reach["_run"] = lad["run"]

    prior = prior_ladder(counts["train"])

    J = build_json(cfg, ladders, runs, abl, hist, counts, reach, prior)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR / "objective1_evidence.json"
    json.dump(J, open(jpath, "w"), indent=2)
    save_csv_alongside(J, jpath)
    print(f"[write] {jpath}")

    ppath = OUT_DIR / "objective1_evidence.pdf"
    build_pdf(J, reach, ppath)
    print(f"[write] {ppath}")

    print("\nstatus: %s" % J["document"]["status"].split(".")[0])
    for r in runs:
        print("  %-26s per-grade recall %s"
              % (r["run"], ["%.3f" % x for x in r["per_grade_recall"]]))


if __name__ == "__main__":
    main()
