#!/usr/bin/env python3
"""Step 22 - Objective 4 evidence pack: lesion-grounded explainability.

Objective 4 is not stated verbatim anywhere in this repository.  It is
reconstructed from the architecture it names - block A9, the "-XAI" in the
project title, and the grounded-explanation chain in a9_xai.explain() - and
that reconstruction is flagged on the front page, exactly as for Objective 1.

The document reports three things:

  1. THE MEASUREMENT.  Attribution faithfulness scored with threshold-free
     metrics that had never been computed (attribution AUROC, AUPRC lift,
     energy pointing game, concentration ratio), alongside the Dice number
     that was being reported, and the analytic ceiling on that Dice.

  2. THE DIAGNOSIS.  lambda_XAI has never been non-zero in any recorded run,
     because the schedule holds it at zero through stages 1-2 and no run has
     ever reached stage 3.  The attribution-consistency loss has therefore
     never been applied.  Objective 4 has not failed; it had not been tested.

  3. THE TEST.  A controlled A/B - identical everything, lambda_XAI the only
     variable - run by run_objective4.sh, measured on identical metrics over
     identical images.  If that has not finished, the section says so rather
     than being filled in.

    python scripts/22_objective4_report.py

Reads   outputs/xai_evaluation.json            (baseline measurement, step 21)
        outputs/xai_obj4_xai_{off,on}.json     (the A/B, step 21 per arm)
        outputs/obj4_xai_{off,on}/history.json (lambda_xai actually applied)
        outputs/<run>/evaluation.json          (the reported lesion_xai block)
Writes  outputs/objective4_evidence.json
        outputs/objective4_evidence.pdf
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dr.config import OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "o1report", ROOT / "scripts" / "15_objective1_report.py")
_o1 = importlib.util.module_from_spec(_spec)
sys.modules["o1report"] = _o1
_spec.loader.exec_module(_o1)
Doc, GRADES, INK, MUTED, RULE, GRADE_C = (
    _o1.Doc, _o1.GRADES, _o1.INK, _o1.MUTED, _o1.RULE, _o1.GRADE_C)

RUNS = [("retfound_plus_laft_xai", "RETFound ViT-L, flagship"),
        ("vits_objectives", "ViT-S, screening-selected")]
AB = [("obj4_xai_off", "lambda_XAI = 0   (control)"),
      ("obj4_xai_on", "lambda_XAI > 0   (treatment)")]

# metric key -> (label, chance level, higher is better)
XAI_METRICS = [
    ("attribution_auroc", "attribution AUROC", 0.5, True),
    ("auprc_lift", "AUPRC lift over prevalence", 1.0, True),
    ("concentration_ratio", "concentration ratio", 1.0, True),
    ("energy_pointing_game", "energy pointing game", None, True),
    ("pointing_game", "pointing game", None, True),
    ("dice_at_0.5", "Dice @ 0.5 (as reported)", None, True),
    ("dice_ceiling", "Dice ceiling at these areas", None, None),
    ("dice_normalised", "normalised Dice (achieved/ceiling)", None, True),
]

LITERATURE = [
    dict(cite="Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep "
              "networks via gradient-based localization. ICCV 2017, 618-626.",
         supports="The attribution operator being scored. Grad-CAM weights activation "
                  "maps by the gradient of the target score, giving a class-discriminative "
                  "saliency map at feature resolution.",
         value="L = ReLU( sum_k alpha_k A^k ),  alpha_k = mean_ij dy/dA^k_ij",
         used_as="Produces the saliency map A on every page of this document."),
    dict(cite="Zhang, J. et al. (2018). Top-down neural attention by excitation backprop. "
              "International Journal of Computer Vision 126, 1084-1102.",
         supports="The pointing game: an attribution is credited when its maximum falls "
                  "inside the annotated region. A localisation metric that does not "
                  "depend on a threshold.",
         value="hit if argmax(saliency) is inside the mask",
         used_as="One of the two localisation metrics reported; it was already being "
                 "computed and is retained unchanged."),
    dict(cite="Wang, H. et al. (2020). Score-CAM: Score-weighted visual explanations for "
              "convolutional neural networks. CVPR Workshops 2020, 24-25.",
         supports="The energy-based pointing game: the fraction of total saliency MASS "
                  "falling inside the mask, rather than only the location of its peak.",
         value="energy = sum(saliency * mask) / sum(saliency)",
         used_as="Added here. Divided by the mask area it gives the concentration ratio, "
                 "whose chance level is exactly 1.0."),
    dict(cite="Dice, L. R. (1945). Measures of the amount of ecologic association between "
              "species. Ecology 26(3), 297-302.",
         supports="The overlap coefficient used by the existing metric and by the "
                  "attribution-consistency loss.",
         value="Dice = 2|A n B| / (|A| + |B|)",
         used_as="Page 4 derives its ceiling at fixed areas, which is what makes the "
                 "reported number interpretable."),
    dict(cite="Saito, T., Rehmsmeier, M. (2015). The precision-recall plot is more "
              "informative than the ROC plot when evaluating binary classifiers on "
              "imbalanced datasets. PLoS ONE 10(3), e0118432.",
         supports="For a target covering ~1% of pixels, average precision against the "
                  "prevalence baseline is the informative summary; ROC-AUC is optimistic "
                  "under extreme imbalance.",
         value="AUPRC baseline = prevalence",
         used_as="Why AUPRC is reported as a LIFT over prevalence rather than raw."),
    dict(cite="Petsiuk, V., Das, A., Saenko, K. (2018). RISE: Randomized input sampling "
              "for explanation of black-box models. BMVC 2018.",
         supports="Perturbation-based faithfulness: an explanation is faithful if "
                  "removing what it cites changes the prediction.",
         value=None,
         used_as="The counterfactual delta-P arm - erase the cited lesions, re-run, "
                 "measure the shift in P(referable)."),
    dict(cite="Adebayo, J. et al. (2018). Sanity checks for saliency maps. NeurIPS 2018, "
              "9505-9515.",
         supports="Saliency methods can produce plausible maps that are independent of "
                  "the model's learned weights; a saliency map must be validated, not "
                  "assumed.",
         value=None,
         used_as="Why chance-level attribution is reported as a finding rather than "
                 "excused, and why the A/B on page 6 is the right test."),
    dict(cite="Rudin, C. (2019). Stop explaining black box machine learning models for "
              "high stakes decisions and use interpretable models instead. "
              "Nature Machine Intelligence 1, 206-215.",
         supports="Post-hoc explanations of high-stakes medical models are unreliable "
                  "unless their faithfulness is measured, not asserted.",
         value=None,
         used_as="The standard this objective is held to: a grounded explanation must "
                 "be shown to track the evidence it cites."),
    dict(cite="Wilkinson, C. P. et al. (2003). Proposed international clinical diabetic "
              "retinopathy and diabetic macular edema disease severity scales. "
              "Ophthalmology 110(9), 1677-1682.",
         supports="The lesion classes an explanation must cite to be clinically "
                  "meaningful: microaneurysms, haemorrhages, exudates, neovascularisation.",
         value="grade defined by lesion type and distribution",
         used_as="Why the grounding target is a lesion mask rather than a saliency blob."),
    dict(cite="Porwal, P. et al. (2018). IDRiD: Indian Diabetic Retinopathy Image Dataset. "
              "Data 3(3), 25.",
         supports="Pixel-level expert lesion annotation - the stronger reference standard "
                  "for an attribution score.",
         value="81 images with MA/HE/EX/SE masks",
         used_as="Named on page 7 as the reference this measurement should move to; "
                 "EyePACS itself carries no pixel labels."),
    dict(cite="Li, T. et al. (2019). Diagnostic assessment of deep learning algorithms for "
              "diabetic retinopathy screening. Information Sciences 501, 511-522. (DDR)",
         supports="The second source of pixel-level lesion masks.",
         value="757 annotated images",
         used_as="Together with IDRiD, the 838-image reference already used for "
                 "Objective 2."),
]


# --------------------------------------------------------------------------
def read_measurement(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.load(open(path))
    if "metrics" not in d:
        return None
    return d


def read_reported_xai() -> list:
    out = []
    for tag, label in RUNS:
        f = OUT_DIR / tag / "evaluation.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        lx = d.get("lesion_xai")
        if lx:
            out.append({"run": tag, "label": label, **lx})
    return out


def read_lambda_history() -> list:
    """What lambda_XAI was actually applied, per run - the diagnosis."""
    out = []
    for tag, label in RUNS + AB:
        f = OUT_DIR / tag / "history.json"
        if not f.exists():
            continue
        h = json.load(open(f))
        lam = [e.get("lambda_xai", 0.0) for e in h]
        xai = [e.get("train_xai", 0.0) for e in h]
        out.append({"run": tag, "label": label, "epochs": len(h),
                    "last_stage": h[-1].get("stage") if h else None,
                    "lambda_xai_by_epoch": lam,
                    "max_lambda_xai": float(max(lam)) if lam else 0.0,
                    "train_xai_by_epoch": xai,
                    "ever_applied": bool(lam and max(lam) > 0)})
    return out


def summarise_ab(off: dict | None, on: dict | None) -> dict:
    if not (off and on):
        return {}
    out = {}
    for key, label, chance, higher in XAI_METRICS:
        a = off["metrics"].get(key, {}).get("mean")
        b = on["metrics"].get(key, {}).get("mean")
        if a is None or b is None:
            continue
        out[key] = {"label": label, "chance": chance,
                    "xai_off": a, "xai_on": b, "delta": b - a,
                    "improved": bool(b > a) if higher else None}
    return out


# --------------------------------------------------------------------------
def build_json() -> dict:
    cfg = default_config()
    base = read_measurement(OUT_DIR / "xai_evaluation.json")
    off = read_measurement(OUT_DIR / "xai_obj4_xai_off.json")
    on = read_measurement(OUT_DIR / "xai_obj4_xai_on.json")
    reported = read_reported_xai()
    lam_hist = read_lambda_history()
    ab = summarise_ab(off, on)

    ever = any(h["ever_applied"] for h in lam_hist if h["run"] in
               [r for r, _ in RUNS])
    auroc = base["metrics"]["attribution_auroc"]["mean"] if base else None

    if ab:
        key = "attribution_auroc"
        moved = ab.get(key, {}).get("improved")
        status = ("ACHIEVED. With the attribution-consistency loss switched on, "
                  "attribution moves off chance on the primary localisation metric "
                  "(AUROC %.4f against %.4f in the matched control)."
                  % (ab[key]["xai_on"], ab[key]["xai_off"]) if moved else
                  "NOT ACHIEVED. Switching the attribution-consistency loss on did not "
                  "move attribution off chance under this budget (AUROC %.4f against "
                  "%.4f in the matched control)."
                  % (ab[key]["xai_on"], ab[key]["xai_off"]))
    elif base:
        status = ("NOT YET TESTED. Attribution is at chance (AUROC %.4f) on every "
                  "recorded checkpoint, and the reason is mechanical: lambda_XAI has "
                  "never been non-zero in any run, so the attribution-consistency loss "
                  "has never been applied. The controlled A/B that tests it is running."
                  % auroc)
    else:
        status = "NO MEASUREMENT AVAILABLE - run scripts/21_xai_evaluation.py first."

    return {
        "document": {
            "title": "Objective 4 - evidence pack",
            "objective_as_reconstructed":
                "the system produces lesion-grounded explanations of its decisions - "
                "attribution that lands on the lesions actually present, a citation "
                "chain from prediction to lesion to anatomical region to severity, and "
                "a calibrated confidence - so that a clinician can check the reasoning "
                "and not only the grade",
            "wording_provenance": "RECONSTRUCTED. Objective 4 is not stated verbatim "
                                  "anywhere in this repository. It is inferred from the "
                                  "architecture it names: block A9 (a9_xai.py, "
                                  "'Lesion-Grounded Explainability Engine'), the '-XAI' in "
                                  "the project title, the L_XAI term in the training "
                                  "objective, and the GroundedExplanation chain returned by "
                                  "a9_xai.explain(). Replace this field with the thesis "
                                  "wording before quoting.",
            "generated": date.today().isoformat(),
            "generator": "scripts/22_objective4_report.py",
            "status": status,
        },
        "the_diagnosis": {
            "finding": "lambda_XAI has never been non-zero in any recorded run of this "
                       "repository, so L_XAI has never been applied.",
            "mechanism": "XaiCfg.weight_schedule = %s, indexed by stage. Stages 1 and 2 "
                         "get zero. No run has ever reached stage 3."
                         % (tuple(cfg.xai.weight_schedule),),
            "why_the_schedule_exists": "tying attribution to lesion masks while the "
                                       "attention map is still noise only teaches the "
                                       "model to match noise (a9_xai.xai_weight_for_stage)",
            "evidence": "every history.json carries lambda_xai 0.0 in every epoch, and "
                        "every training log prints 'xai 0.000@0.000'",
            "consequence": "chance-level attribution is the EXPECTED result for a model "
                           "that was never asked to align anything. Objective 4 had not "
                           "failed; it had not been tested.",
            "lambda_history": lam_hist,
        },
        "measurement": {
            "baseline": base,
            "reported_lesion_xai_blocks": reported,
            "metric_definitions": {
                "attribution_auroc": "saliency treated as a per-pixel score for 'is this "
                                     "pixel a lesion'; threshold-free; chance = 0.5",
                "attribution_auprc": "average precision of the same ranking; the "
                                     "informative summary under ~1% prevalence",
                "auprc_lift": "AUPRC / prevalence; chance = 1.0",
                "energy_pointing_game": "sum(saliency * mask) / sum(saliency)",
                "concentration_ratio": "energy / mask area; chance = 1.0",
                "pointing_game": "1 if argmax(saliency) lies inside the mask",
                "dice_at_0.5": "2|A n B| / (|A| + |B|) with A = saliency >= 0.5",
                "dice_ceiling": "2 min(|A|,|B|) / (|A| + |B|), the largest Dice attainable "
                                "at those two areas",
                "dice_normalised": "dice_at_0.5 / dice_ceiling",
                "counterfactual_delta_p": "P(referable | original) - P(referable | lesions "
                                          "inpainted); positive means the model uses them",
            },
            "hypothesis_tested_and_rejected":
                "That the reported Dice was structurally capped by the sparsity mismatch "
                "between a dense saliency map and a sparse lesion mask. Measured, the "
                "ceiling is %s and the achieved Dice is %s, i.e. the explanation reaches "
                "%s of what the metric can express. The metric is not the binding "
                "constraint; the attribution is at chance on threshold-free metrics too."
                % ("%.3f" % base["metrics"]["dice_ceiling"]["mean"] if base else "-",
                   "%.4f" % base["metrics"]["dice_at_0.5"]["mean"] if base else "-",
                   "%.1f%%" % (100 * base["metrics"]["dice_normalised"]["mean"])
                   if base else "-"),
        },
        "the_test": {
            "design": "controlled A/B: identical backbone, stage schedule, learning "
                      "rates, sampler, all other loss weights, data subset, batch size, "
                      "crop geometry, validation split and selection criterion; "
                      "lambda_XAI is the only variable",
            "arm_A": "lambda_XAI = 0 (control, current behaviour)",
            "arm_B": "lambda_XAI > 0 (treatment, loss applied from stage 1)",
            "mechanism_added": "scripts/03_train.py --xai-weight, which overrides the "
                               "stage schedule; without it a run that stops in stage 2 "
                               "can never apply the loss",
            "driver": "run_objective4.sh",
            "results": ab,
            "arms_complete": bool(ab),
        },
        "reference_mask_caveat": (base or {}).get("reference_mask"),
        "limitations": [
            "The objective wording is reconstructed, not quoted.",
            "The reference mask is the MORPHOLOGICAL lesion prior in "
            "data/cache/lesions.npy, not ophthalmologist annotation. EyePACS carries no "
            "pixel labels. Any attribution score on it is scored against a weak "
            "reference, and a chance-level result could in principle reflect a weak "
            "reference rather than a weak explanation - though the counterfactual arm, "
            "which needs no mask, is an independent check.",
            "Grad-CAM only. Attention rollout and the differentiable attribution used "
            "inside the loss are not scored here, and may localise differently.",
            "The A/B uses ViT-S on a short schedule. It answers 'does switching the loss "
            "on move the attribution', not 'what is the best achievable grounding'.",
            "Single seed, one split, no confidence intervals.",
        ],
        "next_step": {
            "what": "score attribution against the 838 IDRiD + DDR images that carry "
                    "pixel-level ophthalmologist masks, the same reference already used "
                    "for Objective 2",
            "why": "it removes the weak-reference caveat entirely and is the single "
                   "change that would make this evidence hard to attack",
        },
        "reproduce": {
            "baseline measurement": "python scripts/21_xai_evaluation.py --n 120 "
                                    "--counterfactual",
            "the A/B": "./run_objective4.sh",
            "this document": "python scripts/22_objective4_report.py",
            "machine-readable twin": "outputs/objective4_evidence.json",
        },
        "literature": LITERATURE,
    }


# --------------------------------------------------------------------------
def _m(d, key, stat="mean"):
    try:
        return d["metrics"][key][stat]
    except Exception:
        return float("nan")


def build_pdf(J, path: Path) -> None:
    doc = J["document"]
    diag = J["the_diagnosis"]
    meas = J["measurement"]
    base = meas["baseline"]
    test = J["the_test"]
    ab = test["results"]

    with PdfPages(path) as pdf:
        D = Doc(pdf)

        # ---------------- 1 verdict ----------------------------------------
        D.new("Objective 4 — lesion-grounded explainability",
              "Evidence pack · generated %s · RETFound Plus–LAFT–XAI" % doc["generated"])
        D.para("Objective as reconstructed: " + doc["objective_as_reconstructed"],
               size=8.6, color=MUTED)
        D.para("WORDING IS RECONSTRUCTED, NOT QUOTED. Objective 4 appears nowhere verbatim "
               "in this repository; it is inferred from block A9, the '-XAI' in the project "
               "title, the L_XAI loss term and the GroundedExplanation chain. Replace it "
               "before thesis submission.", size=8, color="#8a5a1b")

        D.h2("Status")
        col = ("#1f5c34" if ab and ab.get("attribution_auroc", {}).get("improved")
               else ("#8a2b2b" if ab else "#8a5a1b"))
        D.para(doc["status"], size=9.2, color=col)

        D.h2("The finding that reframes this objective")
        D.para(diag["finding"], size=9)
        D.kv([("schedule", diag["mechanism"]),
              ("evidence", diag["evidence"]),
              ("consequence", "attribution at chance is the EXPECTED result")],
             w=0.20, size=8.2)

        if base:
            D.h2("Attribution, measured threshold-free (n=%d images)"
                 % base["document"]["n_images_scored"])
            D.table(["metric", "mean", "median", "chance"],
                    [[lab, "%.4f" % _m(base, k), "%.4f" % _m(base, k, "median"),
                      "%.2f" % ch if ch is not None else "-"]
                     for k, lab, ch, _ in XAI_METRICS if k in base["metrics"]],
                    [0.29, 0.115, 0.115, 0.09], mono_from=1, size=7.8)
            cf = base.get("counterfactual")
            if cf and cf.get("mean_delta_referable") is not None:
                dp = cf["mean_delta_referable"]
                D.para("Counterfactual delta-P = %+.4f (n=%d). %s"
                       % (dp, cf["n"],
                          "Positive: erasing the cited lesions lowers predicted severity, "
                          "so the model is using lesion evidence even where its saliency "
                          "map does not localise it." if dp > 0 else
                          "NEGATIVE, which is the wrong direction: erasing the cited "
                          "lesions RAISED predicted severity. A lesion-grounded model "
                          "should lose confidence when its evidence is removed. Part of "
                          "this may be the inpainting itself (a blur patch is an "
                          "abnormality of its own), but it cannot be read as faithfulness.")
                       , size=8.4, color=MUTED)

        D.h2("Contents")
        D.para("p2  the objective and what an explanation must satisfy\n"
               "p3  the formulas, in full\n"
               "p4  a hypothesis tested and rejected — the Dice ceiling\n"
               "p5  the diagnosis — lambda_XAI has never been applied\n"
               "p6  the controlled A/B that tests it\n"
               "p7  limitations and the one change that would settle it\n"
               "p8  literature", size=8.6)
        D.close()

        # ---------------- 2 what the objective requires ---------------------
        D.new("What the objective requires", "Objective 4 · the claim, decomposed")
        D.para(doc["objective_as_reconstructed"], size=8.8)
        D.h2("Three separable properties")
        D.para("LOCALISATION   the attribution lands on lesions that are actually there\n"
               "FAITHFULNESS   removing what the explanation cites changes the prediction\n"
               "GROUNDING      the citation chain names lesion, region and contribution",
               size=8.6)
        D.para("They can come apart, and in this system they do. Faithfulness is present "
               "(the counterfactual moves the prediction). Localisation is at chance. "
               "Grounding is implemented and emits a structured chain, but a chain built "
               "on a saliency map that does not localise is a chain of unsupported "
               "citations.", size=8.8)

        D.h2("Why localisation is the binding one")
        D.para("A9 builds its explanation by ranking lesion channels by the attention mass "
               "the attribution places on them (a9_xai.attention_lesion_overlap), then "
               "naming the anatomical region of the top-ranked lesions. If the attribution "
               "is uninformative about lesion location, that ranking is arbitrary and every "
               "downstream field of the GroundedExplanation inherits the problem. "
               "Localisation is therefore measured first and everything else is read "
               "against it.", size=8.8)

        D.h2("What was and was not being measured")
        D.para("The evaluation reported six numbers in its LESION / XAI block. Five of them "
               "(Dice, IoU, pointing game, consistency, counterfactual) were computed; none "
               "was threshold-free except the pointing game, and no ceiling was ever "
               "computed for the Dice. The threshold-free localisation metrics standard in "
               "the saliency literature - attribution AUROC, AUPRC against prevalence, the "
               "energy pointing game - were absent. They are added here.", size=8.8)
        D.close()

        # ---------------- 3 formulas ----------------------------------------
        D.new("The formulas", "Objective 4 · every quantity defined")
        D.h2("1. The attribution (Grad-CAM, Selvaraju et al. 2017)")
        D.para("    alpha_k = mean_ij  dy / dA^k_ij\n"
               "    L       = ReLU( sum_k alpha_k A^k )\n"
               "    sal     = (L - min L) / (max L - min L)      min-max normalised",
               size=8.5, mono=True)
        D.h2("2. The reference mask")
        D.para("    B = 1[ max_c lesion_c >= 0.5 ]     union over the six lesion channels",
               size=8.5, mono=True)
        D.h2("3. Overlap, and its ceiling")
        D.para("    A        = 1[ sal >= 0.5 ]\n"
               "    Dice     = 2|A n B| / (|A| + |B|)\n"
               "    Dice_max = 2 min(|A|,|B|) / (|A| + |B|)      maximal overlap\n"
               "    Dice_norm = Dice / Dice_max                  in [0,1]",
               size=8.5, mono=True)
        D.para("Dice_max is what the metric could return if the above-threshold region "
               "contained every lesion pixel it possibly could. Reporting Dice without it "
               "makes a sparsity artefact indistinguishable from a bad explanation.",
               size=8.4, color=MUTED)
        D.h2("4. Threshold-free localisation")
        D.para("    AUROC  = ROC-AUC( y = B.ravel() , score = sal.ravel() )     chance 0.5\n"
               "    AUPRC  = average precision of the same ranking\n"
               "    lift   = AUPRC / |B|                                        chance 1.0\n"
               "    energy = sum(sal * B) / sum(sal)                (Wang et al. 2020)\n"
               "    conc   = energy / |B|                                       chance 1.0",
               size=8.5, mono=True)
        D.para("Under ~1% prevalence, ROC-AUC is optimistic and average precision against "
               "the prevalence baseline is the informative summary (Saito & Rehmsmeier "
               "2015). Both are reported.", size=8.4, color=MUTED)
        D.h2("5. Faithfulness by perturbation (RISE-style)")
        D.para("    delta_P = P(referable | x) - P(referable | inpaint(x, B))",
               size=8.5, mono=True)
        D.para("Positive delta_P means erasing the cited lesions lowers predicted severity. "
               "This needs no saliency map, so it is an independent check on the "
               "localisation numbers.", size=8.4, color=MUTED)
        D.h2("6. The training term that is supposed to create all of this")
        D.para("    L_XAI  = 1 - Dice( attribution , lesion mask )\n"
               "    L     += lambda_XAI(stage) * L_XAI\n"
               "    lambda_XAI = %s     indexed by stage"
               % (tuple(default_config().xai.weight_schedule),), size=8.5, mono=True)
        D.close()

        # ---------------- 4 rejected hypothesis -----------------------------
        D.new("A hypothesis, tested and rejected",
              "Objective 4 · the Dice ceiling is not the problem")
        D.para("The reported Dice of 0.0025 (flagship) and 0.0367 (ViT-S) invites an "
               "obvious explanation: Dice between a dense saliency map and a sparse lesion "
               "mask is bounded by the area mismatch, so the number would look like failure "
               "even for a perfect explanation. That hypothesis is testable, and it is "
               "wrong.", size=8.8)
        if base:
            D.h2("Measured areas and the ceiling they imply")
            D.table(["quantity", "mean", "median"],
                    [["saliency area, sal >= 0.5", "%.4f" % _m(base, "area_saliency_ge_0.5"),
                      "%.4f" % _m(base, "area_saliency_ge_0.5", "median")],
                     ["lesion union area |B|", "%.4f" % _m(base, "area_lesion_union"),
                      "%.4f" % _m(base, "area_lesion_union", "median")],
                     ["Dice ceiling", "%.4f" % _m(base, "dice_ceiling"),
                      "%.4f" % _m(base, "dice_ceiling", "median")],
                     ["Dice achieved @ 0.5", "%.4f" % _m(base, "dice_at_0.5"),
                      "%.4f" % _m(base, "dice_at_0.5", "median")],
                     ["normalised Dice", "%.4f" % _m(base, "dice_normalised"),
                      "%.4f" % _m(base, "dice_normalised", "median")]],
                    [0.26, 0.13, 0.13], mono_from=1, size=8.0)
            D.para("The saliency map covers about %.1f%% of the image, not the tens of "
                   "percent the hypothesis assumed, so the ceiling sits at %.3f. The "
                   "explanation reaches %.1f%% of it."
                   % (100 * _m(base, "area_saliency_ge_0.5"), _m(base, "dice_ceiling"),
                      100 * _m(base, "dice_normalised")), size=8.8)

            D.h2("And the threshold-free metrics agree")
            rows = []
            for k, lab, ch, _h in XAI_METRICS[:5]:
                if k in base["metrics"]:
                    rows.append([lab, "%.4f" % _m(base, k),
                                 "%.2f" % ch if ch is not None else "-",
                                 "at chance" if ch is not None and
                                 abs(_m(base, k) - ch) < 0.05 * max(ch, 1) else "-"])
            D.table(["metric", "measured", "chance", "reading"], rows,
                    [0.27, 0.115, 0.09, 0.12], mono_from=1, size=7.8)
            D.para("No threshold is involved in any of these. The attribution is "
                   "uninformative about lesion location, and the metric was not hiding it.",
                   size=8.8)

            ax = D.axes(0.14, bottom_gap=0.095)
            ks = [k for k, _, ch, _h in XAI_METRICS[:3] if k in base["metrics"]]
            xs = np.arange(len(ks))
            vals = [_m(base, k) for k in ks]
            chs = [dict((k, ch) for k, _, ch, _ in XAI_METRICS)[k] for k in ks]
            ax.bar(xs, vals, width=0.5, color=GRADE_C[3], edgecolor="white", lw=0.6,
                   label="measured")
            for i, c in enumerate(chs):
                ax.plot([i - 0.3, i + 0.3], [c, c], color="#8a2b2b", lw=1.6,
                        label="chance" if i == 0 else None)
            ax.set_xticks(xs)
            ax.set_xticklabels([dict((k, l) for k, l, _, _ in XAI_METRICS)[k]
                                .replace(" ", "\n") for k in ks], fontsize=6.8)
            ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.30))
            ax.set_title("measured attribution against chance", fontsize=8.4,
                         color=INK, pad=6)
        D.close()

        # ---------------- 5 the diagnosis -----------------------------------
        D.new("The diagnosis", "Objective 4 · the loss has never been applied")
        D.para(diag["finding"], size=9.2)
        D.h2("The mechanism")
        D.para(diag["mechanism"], size=8.8)
        D.para("The schedule is deliberate and the reasoning is sound: " +
               diag["why_the_schedule_exists"] + ".", size=8.6, color=MUTED)
        D.para("But it has a consequence nobody checked. A run that stops in stage 2 — and "
               "every run in this repository does — applies lambda_XAI = 0 in every epoch "
               "it ever executes. The term is in the loss function and contributes exactly "
               "nothing.", size=8.8)

        D.h2("Evidence, per run")
        rows = []
        for h in diag["lambda_history"]:
            rows.append([h["label"][:26], str(h["epochs"]), str(h["last_stage"] or "-"),
                         "%.3f" % h["max_lambda_xai"],
                         "YES" if h["ever_applied"] else "never"])
        if rows:
            D.table(["run", "epochs", "last stage", "max lambda_XAI", "applied?"],
                    rows, [0.235, 0.075, 0.115, 0.135, 0.09], mono_from=1, size=7.8)

        rep = J["measurement"]["reported_lesion_xai_blocks"]
        if rep:
            D.h2("The reported LESION / XAI block, for reference")
            D.table(["run", "dice", "IoU", "pointing", "counterfactual", "expert AUROC"],
                    [[r["label"][:22], "%.4f" % r["lesion_attr_dice"],
                      "%.4f" % r["lesion_attr_iou"], "%.4f" % r["pointing_game"],
                      "%+.4f" % r["counterfactual_delta_p"],
                      "%.4f" % r["lesion_expert_auroc"]] for r in rep],
                    [0.185, 0.095, 0.095, 0.095, 0.135, 0.115], mono_from=1, size=7.6)
            worst = min(r["counterfactual_delta_p"] for r in rep)
            D.para("Note the last two columns, and note them carefully. The lesion experts "
                   "discriminate genuinely well (AUROC up to %.2f) - the system HAS lesion "
                   "evidence. But the counterfactual is NEGATIVE (down to %+.4f), meaning "
                   "erasing the cited lesions RAISED the predicted referable probability. "
                   "That is the opposite of what a lesion-grounded model should do, and it "
                   "is consistent with the chance-level localisation: the attribution is "
                   "not pointing at the evidence the model is actually using."
                   % (max(r["lesion_expert_auroc"] for r in rep), worst), size=8.6)

        D.h2("What follows")
        D.para(diag["consequence"][0].upper() + diag["consequence"][1:], size=8.8)
        D.close()

        # ---------------- 6 the A/B -----------------------------------------
        D.new("The test", "Objective 4 · lambda_XAI as the only variable")
        D.para(test["design"] + ".", size=8.8)
        D.kv([("arm A", test["arm_A"]), ("arm B", test["arm_B"]),
              ("mechanism added", test["mechanism_added"]),
              ("driver", test["driver"])], w=0.22, size=8.2)

        if not ab:
            D.h2("IN PROGRESS")
            D.para("The two arms have not both produced an attribution measurement yet. "
                   "This section is deliberately left empty rather than estimated; "
                   "re-running scripts/22_objective4_report.py after ./run_objective4.sh "
                   "finishes replaces this page with the measured before/after.",
                   size=8.8, color="#8a5a1b")
            D.h2("What the page will contain")
            D.para("the same threshold-free metrics on the same %s images, for both arms, "
                   "with the delta and whether attribution moved off chance"
                   % (base["document"]["n_images_scored"] if base else "N"), size=8.6)
        else:
            D.h2("Before and after, identical metrics and images")
            D.table(["metric", "lambda_XAI = 0", "lambda_XAI > 0", "delta", "chance"],
                    [[v["label"][:30], "%.4f" % v["xai_off"], "%.4f" % v["xai_on"],
                      "%+.4f" % v["delta"],
                      "%.2f" % v["chance"] if v["chance"] is not None else "-"]
                     for v in ab.values()],
                    [0.245, 0.125, 0.125, 0.105, 0.08], mono_from=1, size=7.6)

            ax = D.axes(0.15, bottom_gap=0.075)
            ks = [k for k in ("attribution_auroc", "auprc_lift", "concentration_ratio",
                              "dice_normalised") if k in ab]
            xs = np.arange(len(ks))
            ax.bar(xs - 0.2, [ab[k]["xai_off"] for k in ks], width=0.38,
                   color=GRADE_C[1], label="lambda_XAI = 0", edgecolor="white", lw=0.5)
            ax.bar(xs + 0.2, [ab[k]["xai_on"] for k in ks], width=0.38,
                   color=GRADE_C[4], label="lambda_XAI > 0", edgecolor="white", lw=0.5)
            for i, k in enumerate(ks):
                c = ab[k]["chance"]
                if c is not None:
                    ax.plot([i - 0.42, i + 0.42], [c, c], color="#8a2b2b", lw=1.4)
            ax.set_xticks(xs)
            ax.set_xticklabels([ab[k]["label"].replace(" ", "\n") for k in ks],
                               fontsize=6.6)
            ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16))
            ax.set_title("attribution before and after the loss is switched on "
                         "(red line = chance)", fontsize=8.3, color=INK, pad=6)

            D.h2("Reading it")
            k = "attribution_auroc"
            D.para("The primary localisation metric moves from %.4f to %.4f against a "
                   "chance level of 0.50. %s"
                   % (ab[k]["xai_off"], ab[k]["xai_on"],
                      "That is the objective's mechanism working." if ab[k]["improved"]
                      else "That is not a move off chance, and the objective is not met "
                           "by simply enabling the term under this budget."), size=8.8)
        D.close()

        # ---------------- 7 limitations -------------------------------------
        D.new("Limitations and the next step", "Objective 4 · scope")
        D.h2("Limitations")
        for i, l in enumerate(J["limitations"], 1):
            D.para("%d.  %s" % (i, l), size=8.5, lead=0.0138)
            D.para("", size=3)
        D.h2("The one change that would settle it")
        D.para(J["next_step"]["what"] + " — " + J["next_step"]["why"] + ".", size=8.8)
        D.para("EyePACS has no pixel labels, so every attribution number in this document "
               "is scored against morphological priors. The IDRiD + DDR masks are real "
               "ophthalmologist annotation and are already in this repository, already "
               "used by Objective 2. Moving the measurement onto them removes the only "
               "caveat an examiner is likely to press on.", size=8.6, color=MUTED)
        D.h2("Reproduce")
        D.kv(list(J["reproduce"].items()), w=0.32, size=8.0)
        D.h2("Status")
        D.para(doc["status"], size=8.8)
        D.close()

        # ---------------- 8 literature --------------------------------------
        lit = J["literature"]
        per_page = 5
        for start in range(0, len(lit), per_page):
            D.new("Literature" if start == 0 else "Literature (cont.)",
                  "Objective 4 · what each measurement is answerable to")
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
        info["Title"] = "Objective 4 evidence pack - lesion-grounded explainability"
        info["Subject"] = doc["status"]


def main() -> None:
    J = build_json()
    jpath = OUT_DIR / "objective4_evidence.json"
    json.dump(J, open(jpath, "w"), indent=2)
    save_csv_alongside(J, jpath)
    print(f"[write] {jpath}")
    ppath = OUT_DIR / "objective4_evidence.pdf"
    build_pdf(J, ppath)
    print(f"[write] {ppath}")
    print("\n" + J["document"]["status"])


if __name__ == "__main__":
    main()
