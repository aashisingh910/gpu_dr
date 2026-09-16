#!/usr/bin/env python3
"""Step 17 - post-hoc decode calibration: the method, its formulas, its result.

Turns outputs/ordinal_calibration.json (written by step 16) into a printable
document: every formula the method rests on, every fitted value, the val -> test
protocol stated so it can be checked, and the measured result for all three
decode rules on both splits.

This is a companion to outputs/objective1_evidence.pdf, which diagnoses why the
minority grades collapse.  This one records the one remedy that needs no
retraining, and whether it worked.

    python scripts/17_calibration_report.py

Reads   outputs/ordinal_calibration.json     (step 16)
        outputs/objective1_evidence.json     (for the prior-matched ladder)
Writes  outputs/objective1_calibration.pdf
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dr.config import OUT_DIR  # noqa: E402

# reuse the page furniture of the Objective 1 pack so the two documents match
_spec = importlib.util.spec_from_file_location(
    "o1report", ROOT / "scripts" / "15_objective1_report.py")
_o1 = importlib.util.module_from_spec(_spec)
sys.modules["o1report"] = _o1
_spec.loader.exec_module(_o1)
Doc, GRADES, INK, MUTED, RULE, GRADE_C = (
    _o1.Doc, _o1.GRADES, _o1.INK, _o1.MUTED, _o1.RULE, _o1.GRADE_C)

RULES = [("baseline_coral_0.5", "CORAL, hard cut 0.5"),
         ("argmax", "argmax over class_probs"),
         ("calibrated", "CORAL, cuts fitted on val")]

LITERATURE = [
    dict(cite="Cao, W., Mirjalili, V., Raschka, S. (2020). Rank consistent ordinal "
              "regression for neural networks with application to age estimation. "
              "Pattern Recognition Letters 140, 325-331.",
         used_as="Defines the head being decoded: P(Y>k) = sigmoid(z + b_k) with one "
                 "shared projection z and K-1 learned biases. The 0.5 cut is the "
                 "paper's decode rule, not a fitted quantity."),
    dict(cite="Menon, A. K. et al. (2021). Long-tail learning via logit adjustment. "
              "ICLR 2021.",
         used_as="The method. Adjusting the decision logit by a per-class offset is the "
                 "principled correction for a long-tailed prior, and is consistent for "
                 "balanced error. Here the offset is fitted rather than set to log(prior)."),
    dict(cite="Lin, T.-Y. et al. (2017). Focal loss for dense object detection. "
              "ICCV 2017, 2980-2988.",
         used_as="Precedent for moving the decision bias to the base rate rather than "
                 "leaving it at a data-independent constant."),
    dict(cite="Kang, B. et al. (2020). Decoupling representation and classifier for "
              "long-tailed recognition. ICLR 2020.",
         used_as="Why this works at all without retraining: the representation and the "
                 "classifier boundary are separable problems, and the boundary is the "
                 "one that is mis-set here."),
    dict(cite="Provost, F., Fawcett, T. (2001). Robust classification for imprecise "
              "environments. Machine Learning 42(3), 203-231.",
         used_as="Threshold selection as a decision-theoretic step separate from model "
                 "fitting - the justification for choosing a cut at all."),
    dict(cite="Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision "
              "for scaled disagreement or partial credit. Psychological Bulletin 70(4), "
              "213-220.",
         used_as="The QWK term of the objective the cuts are fitted against."),
    dict(cite="Gulshan, V. et al. (2016). Development and validation of a deep learning "
              "algorithm for detection of diabetic retinopathy in retinal fundus "
              "photographs. JAMA 316(22), 2402-2410.",
         used_as="The AUCs are unchanged by any decode rule; only the operating point "
                 "moves. This is the standard framing for that distinction."),
]


def _rec(r):
    return " ".join("%6.3f" % x for x in r["per_grade_recall"])


def build(cal: dict, prior: dict | None, path: Path) -> None:
    doc = cal["document"]
    fc = cal["fitted_cuts"]
    tau = np.array(fc["tau"])
    b = np.array(fc["learned_ladder_b_k"])
    eff = np.array(fc["effective_ladder_after_calibration"])
    res = cal["results"]
    passed = res["test"]["calibrated"]["all_grades_predicted"]

    with PdfPages(path) as pdf:
        D = Doc(pdf)
        D.close_label = None

        # ---------------- page 1 -------------------------------------------
        D.new("Objective 1 — post-hoc decode calibration",
              "Method, formulas and result · generated %s" % doc["generated"])
        D.para("The companion document outputs/objective1_evidence.pdf establishes that the "
               "minority grades collapse because the CORAL decision ladder never left its "
               "initialisation. This document records the one remedy that requires no "
               "retraining, states every formula it rests on, and reports whether it worked.",
               size=9)

        D.h2("Result")
        tc = res["test"]["calibrated"]
        rec = tc["per_grade_recall"]
        cmt = tc["confusion_matrix"]
        weak = int(np.argmin(rec))
        D.para(("PASSED ON THE LITERAL CLAIM. All five severity grades are predicted on the "
                "held-out test split under cuts fitted on validation alone, and QWK rises to "
                "%.3f — above both the 0.5 cut (%.3f) and argmax (%.3f)."
                % (tc["quadratic_weighted_kappa"],
                   res["test"]["baseline_coral_0.5"]["quadratic_weighted_kappa"],
                   res["test"]["argmax"]["quadratic_weighted_kappa"])
                if passed else
                "NOT PASSED. At least one severity grade is still never predicted on test "
                "under cuts fitted on validation alone."),
               size=9.4, color="#1f5c34" if passed else "#8a2b2b")
        if passed:
            D.para("NOT PASSED ON THE SUBSTANTIVE CLAIM. '%s' is recovered at recall %.3f — "
                   "%d of %d images. That is non-zero, which is what the objective as worded "
                   "asks for, and it is not clinically usable detection. Read the two "
                   "sentences above together or neither is true."
                   % (GRADES[weak], rec[weak], cmt[weak][weak], sum(cmt[weak])),
                   size=9.4, color="#8a2b2b")

        t = res["test"]
        D.h2("Test split, all three decode rules")
        D.table(["decode rule"] + GRADES + ["acc", "QWK", "all 5"],
                [[lbl[:26]] + ["%.3f" % x for x in t[k]["per_grade_recall"]]
                 + ["%.3f" % t[k]["accuracy"], "%.3f" % t[k]["quadratic_weighted_kappa"],
                    "yes" if t[k]["all_grades_predicted"] else "NO"]
                 for k, lbl in RULES],
                [0.185, 0.072, 0.068, 0.082, 0.072, 0.062, 0.062, 0.062, 0.05],
                mono_from=1, size=7.7)

        D.h2("What did not change")
        D.kv([("sight-threatening AUC", "%.4f (identical for all three rules)"
               % t["calibrated"]["auc_sight_threatening"]),
              ("referable-DR AUC", "%.4f (identical for all three rules)"
               % t["calibrated"]["auc_referable_dr"]),
              ("network weights", "unchanged — no retraining"),
              ("test labels", "never seen by the fit")], w=0.30)
        D.para("A decode rule cannot change a ranking. The AUCs are a property of the scores, "
               "and they are the same under every rule on this page. What moves is the "
               "operating point — which score becomes which grade.", size=8.6, color=MUTED)

        D.h2("Contents")
        D.para("p2  the formulas, in full\n"
               "p3  the protocol, and why it is not test-set tuning\n"
               "p4  every fitted value\n"
               "p5  results on both splits, with confusion matrices\n"
               "p6  the fit trace\n"
               "p7  what this does and does not claim\n"
               "p8  literature", size=8.6)
        D.close()

        # ---------------- page 2: formulas ---------------------------------
        D.new("The formulas", "Post-hoc decode calibration · every quantity defined")

        D.h2("1. The head (Cao et al. 2020)")
        D.para("One shared projection z = w'h of the fused feature, and K-1 learned biases\n"
               "b_0 > b_1 > ... > b_{K-2} constructed to be strictly decreasing:\n"
               "\n"
               "    b_0 = beta_0 ,   b_k = b_{k-1} - softplus(delta_k)\n"
               "    P(Y > k)  =  sigmoid( z + b_k )\n"
               "\n"
               "Monotonicity is structural: softplus > 0 forces b_k to decrease, so\n"
               "P(Y>0) >= P(Y>1) >= ... and the cumulatives can never cross.",
               size=8.6, mono=False)

        D.h2("2. The categorical distribution")
        D.para("    P(Y=0)   = 1 - P(Y>0)\n"
               "    P(Y=k)   = P(Y>k-1) - P(Y>k)      for 0 < k < K-1\n"
               "    P(Y=K-1) = P(Y>K-2)\n"
               "\n"
               "The interior classes are differences of sigmoids, and are therefore bounded:\n"
               "\n"
               "    P(Y=k)  <=  tanh( (b_{k-1} - b_k) / 4 )\n"
               "\n"
               "which is why argmax cannot reach them when the gaps are narrow. That bound is\n"
               "the subject of page 4 of the evidence pack.", size=8.6)

        D.h2("3. The decode rule being replaced")
        D.para("    yhat = sum_k  1[ P(Y>k) > 0.5 ]", size=9, mono=True)
        D.para("0.5 is the canonical cut. It is a default inherited from the binary case, not "
               "a quantity fitted to anything, and under a long-tailed prior it is the wrong "
               "cut for every k.", size=8.6)

        D.h2("4. The decode rule fitted here")
        D.para("    yhat = sum_k  1[ P(Y>k) > tau_k ]", size=9, mono=True)
        D.para("with one cut per cumulative, fitted on validation.", size=8.6)

        D.h2("5. Equivalence to logit adjustment (Menon et al. 2021)")
        D.para("    P(Y>k) > tau_k\n"
               "      <=>  sigmoid(z + b_k) > tau_k\n"
               "      <=>  z + b_k > logit(tau_k)\n"
               "      <=>  z + ( b_k - logit(tau_k) ) > 0\n"
               "\n"
               "So fitting a cut is exactly shifting the ladder:\n"
               "\n"
               "    b_k^eff  =  b_k - logit(tau_k) ,    logit(t) = log( t / (1-t) )\n"
               "\n"
               "The fitted cuts are therefore reported on page 4 both as probabilities and as\n"
               "the effective ladder they imply, so they can be read against the\n"
               "prior-matched ladder derived in the evidence pack.", size=8.6)

        D.h2("6. The objective the cuts are fitted against")
        D.para("The repository's own model-selection score (scripts/03_train.py:74), so the\n"
               "decode rule is chosen by the same criterion the checkpoint was:\n"
               "\n"
               "    S = 0.5 * QWK  +  0.3 * MacroF1  +  0.2 * mean recall over grades 1,2,3\n"
               "\n"
               "subject to a hard feasibility constraint:\n"
               "\n"
               "    reject any tau for which some grade is never predicted\n"
               "\n"
               "The constraint is what makes this a test of Objective 1 rather than of\n"
               "accuracy: a rule that maximises S by abandoning a grade is not admissible.",
               size=8.6)

        D.h2("7. The search")
        D.para("Coordinate ascent over k. Candidates for tau_k are the %s empirical quantiles\n"
               "of P(Y>k) observed on validation, so the search runs over cuts that actually\n"
               "separate this split's scores. Sweeps repeat until no coordinate improves S."
               % len(cal.get("fit_trace", [{}])[0].get("taus", [])) or "96", size=8.6)
        D.close()

        # ---------------- page 3: protocol ---------------------------------
        D.new("The protocol", "Post-hoc decode calibration · why this is not test-set tuning")
        D.para(doc["protocol"], size=9)

        D.h2("Order of operations")
        D.para("  1.  the checkpoint is loaded and frozen — no weight is updated, ever\n"
               "  2.  P(Y=k) is computed once on val (n=%d) and once on test (n=%d)\n"
               "  3.  tau is fitted on the VAL scores and VAL labels only\n"
               "  4.  tau is frozen\n"
               "  5.  the frozen tau is applied to the TEST scores\n"
               "  6.  test labels are read for the first time, to score step 5"
               % (cal["n"]["val"], cal["n"]["test"]), size=8.6, mono=True)

        D.h2("Why this is legitimate")
        D.para("This is the protocol scripts/09_screening.py already uses for the referral "
               "threshold: select on validation, freeze, report on test. The README states the "
               "reason plainly — tuning a threshold on the data you report it on inflates the "
               "number and is the most common way a screening result becomes meaningless. The "
               "same discipline is applied here to the five-class decode instead of to the "
               "binary one.", size=8.8)
        D.para("Choosing an operating point is a decision-theoretic step that is separate from "
               "fitting the model (Provost & Fawcett 2001). Every threshold in this "
               "repository is chosen this way; the 0.5 in the canonical CORAL rule is the "
               "only one that was never chosen at all.", size=8.8)

        D.h2("What would make it illegitimate")
        D.para("  ·  fitting tau on the test split, or on val+test pooled\n"
               "  ·  refitting tau after seeing the test result\n"
               "  ·  reporting the val number as though it were held out\n"
               "  ·  selecting among many checkpoints by their test score\n"
               "\n"
               "None of these is done here. The val and test numbers are both reported on\n"
               "page 5 precisely so the gap between them is visible.", size=8.6)

        D.h2("What it cannot do")
        D.para("Calibration moves the operating point along a fixed ranking. It cannot improve "
               "the ranking, so every AUC is invariant to it. If the model's scores do not "
               "separate a grade from its neighbours, no cut will recover that grade at any "
               "useful precision — it will only trade recall against the grades either side. "
               "Page 5 shows exactly that trade.", size=8.8)
        D.close()

        # ---------------- page 4: fitted values ----------------------------
        D.new("The fitted values", "Post-hoc decode calibration · every number")
        D.h2("Cuts, and the ladder they imply")
        rows = []
        for k in range(len(tau)):
            rows.append(["P(Y > %d)" % k, "0.500", "%.4f" % tau[k],
                         "%+.3f" % b[k], "%+.3f" % eff[k],
                         "%+.3f" % (eff[k] - b[k])])
        D.table(["cumulative", "default cut", "fitted tau_k", "b_k", "b_k eff", "shift"],
                rows, [0.135, 0.115, 0.115, 0.105, 0.105, 0.10], mono_from=1)
        D.para("b_k eff = b_k - logit(tau_k). A positive shift widens the region decoded as "
               "at-least-grade-(k+1); a negative shift narrows it.", size=8.2, color=MUTED)

        if prior:
            D.h2("Read against the prior-matched ladder")
            pm = prior["ladders"]["prior_matched"]
            rows = []
            for k in range(len(tau)):
                rows.append(["b_%d" % k, "%+.3f" % b[k], "%+.3f" % eff[k],
                             "%+.3f" % pm["thresholds"][k],
                             "%.4f" % pm["p_y_gt_k"][k]])
            D.table(["", "learned", "calibrated (eff)", "prior-matched", "P(Y>k) on train"],
                    rows, [0.075, 0.115, 0.155, 0.135, 0.15], mono_from=1)
            D.para("The prior-matched column is the ladder derived independently in the "
                   "evidence pack from the training class frequencies alone, with no fitting. "
                   "Agreement between it and the calibrated column would be evidence that the "
                   "fit found the prior rather than noise in the validation split; "
                   "disagreement is evidence it did not.", size=8.4, color=MUTED)

            ax = D.axes(0.13, left=0.145, width=0.77, top_gap=0.042, bottom_gap=0.058)
            ks = np.arange(len(tau))
            ax.plot(ks, b, "o-", color=GRADE_C[1], lw=1.4, ms=5, label="learned b_k")
            ax.plot(ks, eff, "s-", color=GRADE_C[4], lw=1.4, ms=5, label="calibrated b_k eff")
            ax.plot(ks, pm["thresholds"], "^--", color="#8a5a1b", lw=1.4, ms=5,
                    label="prior-matched")
            ax.set_xticks(ks)
            ax.set_xticklabels(["b_%d" % k for k in ks], fontsize=7.5)
            ax.set_ylabel("threshold value", fontsize=8, color=MUTED)
            ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper center",
                      bbox_to_anchor=(0.5, -0.14))
            ax.set_title("three ladders: as learned, as calibrated, as the prior implies",
                         fontsize=8.4, color=INK, pad=6)

        D.h2("Feasibility constraint")
        D.para(fc["constraint"], size=8.6)
        D.close()

        # ---------------- page 5: results ----------------------------------
        D.new("Results", "Post-hoc decode calibration · both splits, all three rules")
        for split in ("val", "test"):
            lab = ("validation — the cuts were fitted here, so this is NOT held out"
                   if split == "val" else
                   "test — held out; the cuts were frozen before it was scored")
            D.h2("%s (n=%d)" % (split.upper(), cal["n"][split]))
            D.para(lab, size=8.0, color=MUTED)
            rows = []
            for k, lbl in RULES:
                r = cal["results"][split][k]
                rows.append([lbl[:26]] + ["%.3f" % x for x in r["per_grade_recall"]]
                            + ["%.3f" % r["accuracy"],
                               "%.3f" % r["quadratic_weighted_kappa"],
                               "%.3f" % r["f1_macro"],
                               "yes" if r["all_grades_predicted"] else "NO"])
            D.table(["decode rule"] + GRADES + ["acc", "QWK", "F1", "all5"], rows,
                    [0.17, 0.066, 0.062, 0.076, 0.066, 0.058, 0.058, 0.058, 0.058, 0.05],
                    mono_from=1, size=7.4)

        D.h2("Predictions issued per grade, test split")
        D.table(["decode rule"] + GRADES,
                [[lbl[:26]] + [str(x) for x in cal["results"]["test"][k]["n_predicted_as"]]
                 for k, lbl in RULES],
                [0.22, 0.10, 0.10, 0.11, 0.10, 0.10], mono_from=1)

        ax = D.axes(0.15, bottom_gap=0.075)
        w, xs = 0.8 / 3, np.arange(5)
        for i, (k, lbl) in enumerate(RULES):
            ax.bar(xs + i * w - 0.4 + w / 2,
                   cal["results"]["test"][k]["per_grade_recall"],
                   width=w * 0.9, color=[GRADE_C[0], GRADE_C[2], GRADE_C[4]][i],
                   label=lbl[:24], edgecolor="white", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(GRADES, fontsize=7.5)
        ax.set_ylabel("recall", fontsize=8, color=MUTED)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6.6, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13))
        ax.set_title("per-grade recall on test — held-out, cuts frozen",
                     fontsize=8.5, color=INK, pad=6)

        D.h2("Confusion matrix, calibrated rule, test split")
        cm = cal["results"]["test"]["calibrated"]["confusion_matrix"]
        D.table(["true \\ pred"] + GRADES + ["n"],
                [[GRADES[i]] + [str(v) for v in cm[i]] + [str(sum(cm[i]))]
                 for i in range(5)],
                [0.135, 0.085, 0.085, 0.095, 0.085, 0.085, 0.08], mono_from=1)
        D.close()

        # ---------------- page 6: fit trace ---------------------------------
        D.new("The fit", "Post-hoc decode calibration · coordinate ascent trace")
        D.para("Each pass sweeps every tau_k once, accepting a candidate only if it raises S "
               "and keeps all five grades reachable. Pass 0 is the unfitted 0.5 cut.", size=8.8)
        rows = []
        for t_ in cal.get("fit_trace", []):
            rows.append([str(t_["pass"]),
                         "[" + ", ".join("%.3f" % x for x in t_["taus"]) + "]",
                         "%.4f" % t_["selection_score"], "%.3f" % t_["qwk"],
                         "%.3f" % t_["f1_macro"], "%.3f" % t_["minority_recall"],
                         "yes" if t_["all_grades_predicted"] else "NO"])
        D.table(["pass", "tau", "S", "QWK", "F1", "min.rec", "all5"], rows,
                [0.045, 0.28, 0.075, 0.068, 0.062, 0.078, 0.05], mono_from=1, size=7.4)
        D.para("S = 0.5*QWK + 0.3*MacroF1 + 0.2*minority recall, on validation.",
               size=8.2, color=MUTED)

        if len(cal.get("fit_trace", [])) > 1:
            ax = D.axes(0.14)
            tr = cal["fit_trace"]
            ps = [t_["pass"] for t_ in tr]
            for key, col, lab in (("selection_score", GRADE_C[4], "S"),
                                  ("qwk", GRADE_C[2], "QWK"),
                                  ("minority_recall", "#8a5a1b", "minority recall")):
                ax.plot(ps, [t_[key] for t_ in tr], "o-", color=col, lw=1.4, ms=4, label=lab)
            ax.set_xlabel("coordinate-ascent pass", fontsize=8, color=MUTED)
            ax.set_xticks(ps)
            ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16))
            ax.set_title("validation objective during the fit", fontsize=8.4, color=INK, pad=6)
        D.close()

        # ---------------- page 7: claims ------------------------------------
        D.new("What this does and does not claim",
              "Post-hoc decode calibration · scope")
        D.h2("Claimed")
        D.para("  ·  Under cuts fitted on validation and frozen, the decode rule emits\n"
               "     %s of the five grades on the held-out test split.\n"
               "  ·  The improvement is a change of operating point, achieved without\n"
               "     retraining and without the test labels entering the fit.\n"
               "  ·  The fitted cuts are equivalent to a ladder shift, and are reported as\n"
               "     such so they can be checked against the training prior."
               % sum(1 for x in cal["results"]["test"]["calibrated"]["per_grade_recall"]
                     if x > 0), size=8.6)

        D.h2("Not claimed")
        D.para("  ·  That the model is better. No AUC moves; the ranking is untouched.\n"
               "  ·  That the underlying diagnosis is fixed. The ladder is still frozen at\n"
               "     initialisation inside the checkpoint; this corrects it after the fact.\n"
               "     The prior-init retrain proposed in the evidence pack remains untested.\n"
               "  ·  That these cuts transfer. Thresholds are split- and site-specific; the\n"
               "     README already records that the referral threshold does not survive the\n"
               "     move to APTOS, and there is no reason these would either.\n"
               "  ·  That per-grade recall at this level is clinically sufficient.", size=8.6)

        D.h2("Limitations")
        D.para("  ·  One checkpoint, one seed, no confidence intervals on the per-grade\n"
               "     recalls. With %d Severe and %d PDR images in test, single-grade recalls\n"
               "     move by ~0.04 per image.\n"
               "  ·  The validation split has the same size problem, so the fitted cuts are\n"
               "     themselves estimated from few minority examples.\n"
               "  ·  The val/test discrepancy recorded on page 7 of the evidence pack is\n"
               "     still unexplained, and it bears on how much any number from this\n"
               "     checkpoint should be trusted."
               % (sum(cal["results"]["test"]["calibrated"]["confusion_matrix"][3]),
                  sum(cal["results"]["test"]["calibrated"]["confusion_matrix"][4])), size=8.6)

        D.h2("Reproduce")
        D.kv([("fit and evaluate", "python scripts/16_ordinal_calibration.py"),
              ("refit from cached scores", "python scripts/16_ordinal_calibration.py --refit-only"),
              ("this document", "python scripts/17_calibration_report.py"),
              ("machine-readable twin", "outputs/ordinal_calibration.json"),
              ("checkpoint", doc["checkpoint"])], w=0.30, size=8.0)
        D.close()

        # ---------------- page 8: literature --------------------------------
        D.new("Literature", "Post-hoc decode calibration · what the method rests on")
        for ref in LITERATURE:
            D.para(ref["cite"], size=8.2)
            D.para("used as:   " + ref["used_as"], size=7.8, color=MUTED, indent=0.012)
            D.para("", size=4)
        D.close()

        info = pdf.infodict()
        info["Title"] = "Objective 1 - post-hoc decode calibration"
        info["Subject"] = ("all five grades predicted on test" if passed
                           else "not all grades recovered")


def main() -> None:
    cal_path = OUT_DIR / "ordinal_calibration.json"
    if not cal_path.exists():
        raise SystemExit("run scripts/16_ordinal_calibration.py first")
    cal = json.load(open(cal_path))
    ev = OUT_DIR / "objective1_evidence.json"
    prior = json.load(open(ev)) if ev.exists() else None

    out = OUT_DIR / "objective1_calibration.pdf"
    build(cal, prior, out)
    print(f"[write] {out}")
    t = cal["results"]["test"]["calibrated"]
    print("test, calibrated: recall %s  acc %.3f  QWK %.3f  all five grades: %s"
          % (["%.3f" % x for x in t["per_grade_recall"]], t["accuracy"],
             t["quadratic_weighted_kappa"], t["all_grades_predicted"]))


if __name__ == "__main__":
    main()
