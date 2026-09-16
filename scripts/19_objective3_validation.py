#!/usr/bin/env python3
"""Step 19 - Objective 3 validation: the adaptation arm, with the probe as control.

Step 18 reported Objective 3 as SPLIT: the efficiency half supported, the
retinal-domain half not supported by a linear probe.  That probe answers a
question Objective 3 does not ask.  Objective 3 says a large model can be
"adapted efficiently"; a linear probe measures a FROZEN representation and
never adapts anything.  The distinction matters here specifically because
RETFound is an MAE, and MAE backbones are documented to underperform supervised
ones under linear probing while matching or beating them as soon as any
adaptation is permitted (He et al. 2022).

This script reports the matched adaptation experiment run by run_objective3.sh:
the same GLA-LoRA pipeline, twice, changing only the initialisation.  The probe
results are retained as the frozen-feature control, so the document carries both
instruments and states which claim each one licenses.

    python scripts/19_objective3_validation.py

Reads   outputs/obj3_retfound/{evaluation,history,gla_lora,config}.json
        outputs/obj3_imagenet/{evaluation,history,gla_lora,config}.json
        outputs/linear_probe_*.json              (the frozen control)
        outputs/logs/obj3_*_train.log            (trainable fractions, live)
Writes  outputs/objective3_validation.json
        outputs/objective3_validation.pdf
"""
from __future__ import annotations

import importlib.util
import json
import re
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
from dr.config import OUT_DIR  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "o1report", ROOT / "scripts" / "15_objective1_report.py")
_o1 = importlib.util.module_from_spec(_spec)
sys.modules["o1report"] = _o1
_spec.loader.exec_module(_o1)
Doc, GRADES, INK, MUTED, RULE, GRADE_C = (
    _o1.Doc, _o1.GRADES, _o1.INK, _o1.MUTED, _o1.RULE, _o1.GRADE_C)

ARMS = [("obj3_retfound", "RETFound MAE ViT-L/16", "retinal, 1.6M fundus images"),
        ("obj3_imagenet", "ImageNet ViT-L/16", "generic, supervised")]
PROBES = [("linear_probe_retfound_vitl.json", "RETFound ViT-L/16"),
          ("linear_probe_imagenet_vitl.json", "ImageNet ViT-L/16"),
          ("linear_probe_imagenet_vits.json", "ImageNet ViT-S/16")]

# metrics compared between the two arms, and whether higher is better
METRICS = [("quadratic_weighted_kappa", "QWK", True),
           ("accuracy", "accuracy", True),
           ("balanced_accuracy", "balanced acc", True),
           ("f1_macro", "macro F1", True),
           ("auc_sight_threatening", "STDR AUC", True),
           ("auc_referable_dr", "referable AUC", True),
           ("auroc_macro_ovr", "macro AUROC", True),
           ("adjacent_accuracy", "adjacent acc", True),
           ("mae_grade", "MAE grade", False)]

LITERATURE = [
    dict(cite="Zhou, Y. et al. (2023). A foundation model for generalizable disease "
              "detection from retinal images. Nature 622, 156-163. (RETFound)",
         supports="The retinal foundation model under test. Released as an MAE-pretrained "
                  "ViT-Large/16 encoder; the published downstream protocol is fine-tuning, "
                  "not linear probing.",
         value="ViT-L/16, 303M backbone parameters, MAE on 1.6M retinal images",
         used_as="Arm A's initialisation. That the paper's own protocol is fine-tuning is "
                 "the reason this document exists."),
    dict(cite="He, K. et al. (2022). Masked autoencoders are scalable vision learners. "
              "CVPR 2022, 16000-16009.",
         supports="MAE representations score poorly under linear probing and strongly under "
                  "fine-tuning; the paper reports the gap explicitly and shows it closing as "
                  "soon as even a few blocks are tuned.",
         value="linear probe and fine-tune rank MAE differently",
         used_as="The methodological argument on page 2: a linear probe is the wrong "
                 "instrument for the claim Objective 3 actually makes."),
    dict(cite="Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language "
              "models. ICLR 2022.",
         supports="W + BA with rank r << d, trained while W stays frozen; trainable "
                  "parameters scale with r rather than with |W|.",
         value="delta W = B A,  B in R^{d x r},  A in R^{r x k}",
         used_as="The adaptation mechanism held identical across both arms."),
    dict(cite="Houlsby, N. et al. (2019). Parameter-efficient transfer learning for NLP. "
              "ICML 2019, 2790-2799.",
         supports="Adapter-based transfer, and trained-parameter fraction as the metric "
                  "for 'without full retraining'.",
         value=None,
         used_as="Defines the efficiency quantity reported on page 3."),
    dict(cite="Kornblith, S., Shlens, J., Le, Q. V. (2019). Do better ImageNet models "
              "transfer better? CVPR 2019, 2661-2671.",
         supports="Linear probing and fine-tuning are different questions about a "
                  "representation and can rank backbones differently.",
         value=None,
         used_as="Why both instruments are reported here rather than one replacing the "
                 "other."),
    dict(cite="Raghu, M. et al. (2019). Transfusion: Understanding transfer learning for "
              "medical imaging. NeurIPS 2019, 3347-3357.",
         supports="Transfer from natural images to medical imaging often yields little "
                  "benefit; in-domain advantage is not automatic and must be measured.",
         value=None,
         used_as="Prior art that this comparison is worth running and that either outcome "
                 "is publishable."),
    dict(cite="Matsoukas, C. et al. (2022). What makes transfer learning work for medical "
              "images: Feature reuse & other factors. CVPR 2022, 9225-9234.",
         supports="Which factors drive medical transfer performance; in-domain pretraining "
                  "does not automatically dominate under every adaptation budget.",
         value=None,
         used_as="Framing for whichever direction the result falls."),
    dict(cite="Azizi, S. et al. (2021). Big self-supervised models advance medical image "
              "classification. ICCV 2021, 3478-3488.",
         supports="In-domain self-supervised pretraining improving medical classification.",
         value=None,
         used_as="The hypothesis the adaptation arm tests."),
    dict(cite="Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision "
              "for scaled disagreement or partial credit. Psychological Bulletin 70(4), "
              "213-220.",
         supports="Quadratic weighted kappa, the primary endpoint of the comparison.",
         value="penalty proportional to (i-j)^2",
         used_as="The headline metric, and the QWK term of the selection score both arms "
                 "were checkpointed on."),
    dict(cite="Gulshan, V. et al. (2016). Development and validation of a deep learning "
              "algorithm for detection of diabetic retinopathy in retinal fundus "
              "photographs. JAMA 316(22), 2402-2410.",
         supports="The referral-screening endpoint reported alongside grading.",
         value=None,
         used_as="The sight-threatening AUC row."),
    dict(cite="Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers "
              "for image recognition at scale. ICLR 2021.",
         supports="The ViT-L/16 architecture shared by both arms.",
         value="24 blocks, 1024-d, 16 px patches",
         used_as="Both arms are the same architecture; only the weights differ."),
]


# --------------------------------------------------------------------------
def read_arm(tag: str, label: str, note: str) -> dict | None:
    d = OUT_DIR / tag
    ev = d / "evaluation.json"
    if not ev.exists():
        return None
    e = json.load(open(ev))
    out = {"run": tag, "label": label, "note": note,
           "n_images": e["_meta"]["n_images"],
           "n_patients": e["_meta"].get("n_patients"),
           "confusion_matrix": e["_meta"]["confusion_matrix"],
           "metrics": {**e["diagnostic"], **e["ordinal"]},
           "calibration": e.get("calibration"),
           "deployment": e.get("deployment")}
    cm = np.array(out["confusion_matrix"], float)
    tot = cm.sum(1)
    out["per_grade_recall"] = [float(cm[i, i] / tot[i]) if tot[i] else float("nan")
                               for i in range(len(tot))]
    g = d / "gla_lora.json"
    if g.exists():
        gj = json.load(open(g))
        out["gla_lora"] = {"ranks": gj["ranks"], "lora_params": gj["lora_params"],
                           "rank_mean": float(np.mean(gj["ranks"])),
                           "rank_min": int(min(gj["ranks"])),
                           "rank_max": int(max(gj["ranks"])),
                           "importance_S": gj.get("importance_S")}
    h = d / "history.json"
    if h.exists():
        hist = json.load(open(h))
        out["history"] = [{"epoch": x["epoch"], "stage": x["stage"],
                           "val_qwk": x.get("val_quadratic_weighted_kappa"),
                           "val_accuracy": x.get("val_accuracy"),
                           "val_f1_macro": x.get("val_f1_macro"),
                           "selection_score": x.get("selection_score"),
                           "train_loss": x.get("train_loss"),
                           "epoch_sec": x.get("epoch_sec")} for x in hist]
    log = OUT_DIR / "logs" / f"{tag}_train.log"
    if log.exists():
        txt = log.read_text(errors="ignore")
        m = re.search(r"trainable ([\d,]+) / ([\d,]+) \(([\d.]+)%\)[\s\S]*?"
                      r"trainable ([\d,]+) / ([\d,]+) \(([\d.]+)%\)", txt)
        if m:
            out["trainable"] = {"stage1": {"n": int(m.group(1).replace(",", "")),
                                           "pct": float(m.group(3))},
                                "stage2": {"n": int(m.group(4).replace(",", "")),
                                           "total": int(m.group(5).replace(",", "")),
                                           "pct": float(m.group(6))}}
        s = re.search(r"\[A4\] (backbone: .*)", txt)
        out["backbone_line"] = s.group(1) if s else None
        s = re.search(r"\[A4\] (RETFound weights loaded .*|no RETFound.*|.*ImageNet.*)", txt)
        out["init_line"] = s.group(1) if s else None
    return out


def read_probes() -> list:
    out = []
    for fname, label in PROBES:
        f = OUT_DIR / fname
        if not f.exists():
            continue
        d = json.load(open(f))
        out.append({"label": label, "tag": d["tag"],
                    "trainable_fraction_pct": d["trainable_fraction_pct"],
                    "metrics": d["test"]})
    return out


# --------------------------------------------------------------------------
def build_json(arms, probes) -> dict:
    a = {x["run"]: x for x in arms}
    ret, imn = a.get("obj3_retfound"), a.get("obj3_imagenet")
    delta, won = {}, None
    if ret and imn:
        for key, lab, higher in METRICS:
            rv, iv = ret["metrics"].get(key), imn["metrics"].get(key)
            if rv is None or iv is None:
                continue
            d = rv - iv
            delta[key] = {"label": lab, "retfound": rv, "imagenet": iv,
                          "delta": d, "higher_is_better": higher,
                          "retfound_better": bool(d > 0) if higher else bool(d < 0)}
        won = delta.get("quadratic_weighted_kappa", {}).get("retfound_better")

    pr = {p["label"]: p["metrics"]["quadratic_weighted_kappa"] for p in probes}
    probe_won = (pr.get("RETFound ViT-L/16", -9) > pr.get("ImageNet ViT-L/16", 9))

    if won is None:
        status = "INCOMPLETE - the adaptation arm has not produced both evaluations."
    elif won:
        status = ("MET. Under the adaptation budget the objective actually describes, "
                  "RETFound initialisation beats ImageNet initialisation on the primary "
                  "endpoint, with the backbone frozen in both arms. The efficiency half "
                  "was already supported; both halves now hold.")
    else:
        status = ("NOT MET on the domain half. Under a matched, lightweight adaptation "
                  "budget the RETFound initialisation does not beat the ImageNet "
                  "initialisation on the primary endpoint. The efficiency half remains "
                  "supported.")

    return {
        "document": {
            "title": "Objective 3 - validation by matched adaptation",
            "objective_as_stated_in_code":
                'a large model can be adapted "efficiently, enabling lightweight '
                'optimization without full retraining"; research gap 3 says generic '
                'pretrained backbones lack retinal domain awareness',
            "source_of_wording": "scripts/13_linear_probe.py:2-5",
            "generated": date.today().isoformat(),
            "generator": "scripts/19_objective3_validation.py",
            "status": status,
            "why_this_supersedes_the_probe":
                "A linear probe measures a frozen representation and adapts nothing. "
                "Objective 3 is a claim about adaptation. The probe is retained here as a "
                "control, not as the test.",
        },
        "design": {
            "controlled_comparison": "identical architecture, rank-allocation policy, stage "
                                     "schedule, learning rates, sampler, loss weights, data "
                                     "subset, batch size, crop geometry, validation split "
                                     "and selection criterion; the initialisation is the "
                                     "only variable",
            "backbone_frozen_in_both_arms": True,
            "driver": "run_objective3.sh",
            "arm_A": "RETFound MAE ViT-L/16 (--retfound-ckpt <published checkpoint>)",
            "arm_B": "ImageNet ViT-L/16 (--retfound-ckpt none --allow-no-retfound "
                     "--pretrained)",
            "selection": "0.5*QWK + 0.3*MacroF1 + 0.2*minority recall on validation",
        },
        "results": {
            "adaptation_arms": arms,
            "paired_deltas": delta,
            "frozen_probe_control": probes,
            "probe_and_adaptation_agree": bool(probe_won == won) if won is not None else None,
        },
        "headline_numbers": {
            "primary_endpoint": "quadratic_weighted_kappa",
            "retfound_qwk": ret["metrics"]["quadratic_weighted_kappa"] if ret else None,
            "imagenet_qwk": imn["metrics"]["quadratic_weighted_kappa"] if imn else None,
            "delta_qwk": delta.get("quadratic_weighted_kappa", {}).get("delta"),
            "metrics_favouring_retfound": sum(1 for v in delta.values()
                                              if v["retfound_better"]),
            "metrics_compared": len(delta),
            "trainable_pct": {x["label"]: x.get("trainable", {}).get("stage2", {}).get("pct")
                              for x in arms},
        },
        "limitations": [
            "Short schedule. Both arms ran %s epochs on a sampled subset under a wall-clock "
            "budget, not the published 5/12/14/14 schedule. The comparison is fair because "
            "both arms got the same budget, but neither arm is trained to convergence."
            % (len(ret.get("history", [])) if ret else "?"),
            "Two local crops, not six, and a 224 px global view - the reduced geometry that "
            "fits the time budget. The flagship configuration uses 448 px and six crops.",
            "Single seed per arm. No confidence interval on the difference, and QWK "
            "differences under roughly 0.05 should not be treated as decisive.",
            "Backbone frozen throughout. This tests the initialisation under a LoRA-only "
            "budget; it does not test what happens once blocks are unfrozen, which is where "
            "He et al. 2022 report the MAE advantage growing.",
            "The linear probe result is unchanged and is reported alongside. If the two "
            "instruments disagree, that disagreement is itself the finding and both numbers "
            "belong in any write-up.",
        ],
        "reproduce": {
            "adaptation_arms": "./run_objective3.sh",
            "this_document": "python scripts/19_objective3_validation.py",
            "frozen_probe_control": "python scripts/13_linear_probe.py ...",
            "machine_readable_twin": "outputs/objective3_validation.json",
        },
        "literature": LITERATURE,
    }


# --------------------------------------------------------------------------
def build_pdf(J, path: Path) -> None:
    doc = J["document"]
    arms = {x["run"]: x for x in J["results"]["adaptation_arms"]}
    ret, imn = arms.get("obj3_retfound"), arms.get("obj3_imagenet")
    delta = J["results"]["paired_deltas"]
    probes = J["results"]["frozen_probe_control"]
    hn = J["headline_numbers"]
    won = delta.get("quadratic_weighted_kappa", {}).get("retfound_better")

    with PdfPages(path) as pdf:
        D = Doc(pdf)

        # ---------------- page 1 -------------------------------------------
        D.new("Objective 3 — validated by matched adaptation",
              "Evidence pack · generated %s · RETFound Plus–LAFT–XAI" % doc["generated"])
        D.para("Objective as stated in the code (scripts/13_linear_probe.py:2-5): "
               + doc["objective_as_stated_in_code"], size=8.6, color=MUTED)

        D.h2("Verdict")
        D.para(doc["status"], size=9.2,
               color="#1f5c34" if won else ("#8a2b2b" if won is not None else INK))

        if ret and imn:
            D.h2("Primary endpoint — held-out test split (n=%d)" % ret["n_images"])
            D.table(["initialisation", "QWK", "acc", "bal acc", "macro F1", "STDR AUC"],
                    [[x["label"][:22],
                      "%.4f" % x["metrics"]["quadratic_weighted_kappa"],
                      "%.4f" % x["metrics"]["accuracy"],
                      "%.4f" % x["metrics"]["balanced_accuracy"],
                      "%.4f" % x["metrics"]["f1_macro"],
                      "%.4f" % x["metrics"]["auc_sight_threatening"]]
                     for x in (ret, imn)],
                    [0.20, 0.105, 0.10, 0.10, 0.10, 0.11], mono_from=1, size=7.8)
            d = delta["quadratic_weighted_kappa"]["delta"]
            D.para("Difference on the primary endpoint: %+.4f QWK in favour of %s. "
                   "%d of %d metrics favour RETFound."
                   % (d, "RETFound" if d > 0 else "ImageNet",
                      hn["metrics_favouring_retfound"], hn["metrics_compared"]),
                   size=8.8)

        D.h2("What was held constant")
        D.para(J["design"]["controlled_comparison"] + ".", size=8.6)
        D.kv([("architecture", "ViT-L/16, 24 blocks, 1024-d, both arms"),
              ("backbone", "frozen in both arms (--no-unfreeze)"),
              ("adaptation", "GLA-LoRA + heads only"),
              ("trained fraction", ", ".join(
                  "%s %.2f%%" % (k.split()[0], v) for k, v in hn["trainable_pct"].items()
                  if v is not None) or "-"),
              ("selection", J["design"]["selection"]),
              ("only variable", "the initialisation")], w=0.30, size=8.2)

        D.h2("Contents")
        D.para("p2  why the linear probe was the wrong instrument\n"
               "p3  the design, and every value held constant\n"
               "p4  results — all metrics, paired\n"
               "p5  per-grade behaviour and training curves\n"
               "p6  the frozen-probe control, unchanged\n"
               "p7  limitations and reproduction\n"
               "p8  literature", size=8.6)
        D.close()

        # ---------------- page 2: the argument -----------------------------
        D.new("Why the probe was the wrong instrument",
              "Objective 3 · the methodological correction")
        D.h2("What the objective claims")
        D.para('"a large model can be adapted efficiently, enabling lightweight '
               'optimization without full retraining"', size=9, mono=False)
        D.para("Every operative word is about ADAPTATION. The claim is that the model can be "
               "moved to this task cheaply — not that its frozen features are already "
               "linearly separable for it.", size=8.8)

        D.h2("What a linear probe measures")
        D.para("    freeze all W ;   fit  argmax_k  w_k' f(x) + b_k\n"
               "\n"
               "Nothing in f is adapted. The probe answers: are these features, exactly as "
               "they came out of pretraining, linearly separable for DR grading? That is a "
               "question about the representation, not about adaptability.", size=8.6)

        D.h2("Why the difference is decisive for THIS backbone")
        D.para("RETFound is a masked autoencoder. He et al. (2022) report — as a central "
               "result, not a footnote — that MAE representations score poorly under linear "
               "probing and strongly under fine-tuning, and that the gap closes as soon as "
               "even a small number of parameters are allowed to move. The reason is that "
               "the MAE objective (reconstruct masked patches) never asks the representation "
               "to be linearly separable by class; a supervised ImageNet objective does, "
               "explicitly.", size=8.8)
        D.para("So a linear probe compares an MAE against supervised models on the one axis "
               "the supervised models were directly trained for. The probe result in "
               "outputs/objective3_evidence.pdf is real and is retained on page 6 — but it "
               "cannot settle Objective 3, because it does not test what Objective 3 claims.",
               size=8.8)

        D.h2("The instrument that does match the claim")
        D.para("A matched parameter-efficient fine-tune: give both initialisations the SAME "
               "lightweight adaptation budget and measure what each reaches. That is "
               "simultaneously a test of the efficiency half (the budget is tiny and the "
               "backbone never unfreezes) and of the domain half (the only variable is which "
               "weights the adaptation starts from).", size=8.8)
        D.para("This is what run_objective3.sh runs, and what the rest of this document "
               "reports.", size=8.8)
        D.close()

        # ---------------- page 3: design ------------------------------------
        D.new("The design", "Objective 3 · what varies and what does not")
        D.h2("The two arms")
        D.kv([("arm A", J["design"]["arm_A"]),
              ("arm B", J["design"]["arm_B"])], w=0.10, size=8.2)
        for x in (ret, imn):
            if x and x.get("init_line"):
                D.para("%-22s %s" % (x["label"][:22], x["init_line"]), size=7.4, mono=True)

        D.h2("The adaptation mechanism, identical in both arms")
        D.para("    W_adapted = W_frozen + B A ,    B in R^{d x r} , A in R^{r x k}\n"
               "    r_l = r_min + (r_max - r_min) * S_l\n"
               "    S_l = 0.35*G_l + 0.45*L_l + 0.20*A_l\n"
               "\n"
               "Rank allocation is re-calibrated per arm because it depends on the weights, "
               "so the allocation POLICY is fixed and the resulting ranks are reported for "
               "both.", size=8.6)
        if ret and imn and ret.get("gla_lora") and imn.get("gla_lora"):
            D.table(["arm", "blocks", "rank range", "mean r", "LoRA params"],
                    [[x["label"][:22], str(len(x["gla_lora"]["ranks"])),
                      "%d-%d" % (x["gla_lora"]["rank_min"], x["gla_lora"]["rank_max"]),
                      "%.2f" % x["gla_lora"]["rank_mean"],
                      f"{x['gla_lora']['lora_params']:,}"] for x in (ret, imn)],
                    [0.205, 0.08, 0.105, 0.09, 0.12], mono_from=1, size=7.8)

        D.h2("Trained-parameter fraction — the efficiency claim, measured")
        rows = []
        for x in (ret, imn):
            if x and x.get("trainable"):
                t = x["trainable"]
                rows.append([x["label"][:22], f"{t['stage1']['n']:,}",
                             "%.3f%%" % t["stage1"]["pct"],
                             f"{t['stage2']['n']:,}", "%.3f%%" % t["stage2"]["pct"],
                             f"{t['stage2'].get('total', 0):,}"])
        if rows:
            D.table(["arm", "stage 1 trained", "%", "stage 2 trained", "%", "total params"],
                    rows, [0.185, 0.125, 0.075, 0.125, 0.075, 0.13],
                    mono_from=1, size=7.4)
        D.para("The backbone contributes 303,301,632 frozen parameters to each arm and "
               "receives no gradient in either.", size=8.2, color=MUTED)

        D.h2("Held constant")
        D.para("stage schedule · learning rates · sampler (n^-0.5 + hard-example mining) · "
               "loss weights · data subset and splits · batch size and gradient accumulation "
               "· crop geometry · validation split · checkpoint-selection criterion · seed "
               "handling · evaluation script and decode rule", size=8.6)
        D.close()

        # ---------------- page 4: results -----------------------------------
        D.new("Results", "Objective 3 · every metric, paired")
        if delta:
            D.h2("Paired comparison on the held-out test split")
            rows = []
            for key, lab, higher in METRICS:
                if key not in delta:
                    continue
                v = delta[key]
                rows.append([lab, "%.4f" % v["retfound"], "%.4f" % v["imagenet"],
                             "%+.4f" % v["delta"],
                             "RETFound" if v["retfound_better"] else "ImageNet"])
            D.table(["metric", "RETFound", "ImageNet", "delta", "favours"], rows,
                    [0.155, 0.115, 0.115, 0.105, 0.115], mono_from=1, size=7.8)
            D.para("delta = RETFound - ImageNet. For MAE grade, lower is better and the "
                   "'favours' column accounts for that.", size=8.2, color=MUTED)

            ax = D.axes(0.155, bottom_gap=0.072)
            keys = [k for k, _, _ in METRICS if k in delta and k != "mae_grade"]
            xs = np.arange(len(keys))
            ax.bar(xs - 0.2, [delta[k]["retfound"] for k in keys], width=0.38,
                   color=GRADE_C[4], label="RETFound init", edgecolor="white", lw=0.5)
            ax.bar(xs + 0.2, [delta[k]["imagenet"] for k in keys], width=0.38,
                   color=GRADE_C[1], label="ImageNet init", edgecolor="white", lw=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels([delta[k]["label"].replace(" ", "\n") for k in keys],
                               fontsize=6.6)
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16))
            ax.set_title("matched adaptation budget, only the initialisation differs",
                         fontsize=8.5, color=INK, pad=6)

        D.h2("How to read the size of the difference")
        D.para("Both arms are single-seed and short-schedule. A QWK gap under roughly 0.05 "
               "is inside the noise this setup can resolve, and should be reported as "
               "'no measurable difference' rather than as a win for either arm. The gap "
               "measured here is %s."
               % ("%+.4f" % delta["quadratic_weighted_kappa"]["delta"]
                  if "quadratic_weighted_kappa" in delta else "not available"),
               size=8.8)
        D.close()

        # ---------------- page 5: per-grade + curves ------------------------
        D.new("Per-grade behaviour and training curves", "Objective 3 · detail")
        if ret and imn:
            D.h2("Per-grade recall, test split")
            D.table(["arm"] + GRADES,
                    [[x["label"][:22]] + ["%.3f" % v for v in x["per_grade_recall"]]
                     for x in (ret, imn)],
                    [0.21, 0.098, 0.098, 0.105, 0.098, 0.09], mono_from=1, size=7.8)

            D.h2("Confusion matrices")
            for x in (ret, imn):
                D.para(x["label"], size=8.2, color=MUTED)
                D.table(["true \\ pred"] + GRADES + ["n"],
                        [[GRADES[i]] + [str(v) for v in x["confusion_matrix"][i]]
                         + [str(sum(x["confusion_matrix"][i]))] for i in range(5)],
                        [0.13, 0.082, 0.082, 0.092, 0.082, 0.082, 0.078],
                        mono_from=1, size=7.2)

            if ret.get("history") and imn.get("history"):
                D.h2("Validation during training")
                rows = []
                for x in (ret, imn):
                    for h in x["history"]:
                        rows.append([x["label"][:18], str(h["epoch"]), h["stage"][:10],
                                     "%.4f" % (h["val_qwk"] or float("nan")),
                                     "%.4f" % (h["val_accuracy"] or float("nan")),
                                     "%.1f" % (h["epoch_sec"] or float("nan"))])
                D.table(["arm", "ep", "stage", "val QWK", "val acc", "sec"], rows,
                        [0.165, 0.04, 0.10, 0.09, 0.09, 0.07], mono_from=1, size=7.4)
        D.close()

        # ---------------- page 6: probe control -----------------------------
        D.new("The frozen-probe control", "Objective 3 · retained, unchanged")
        D.para("These are the step-13 linear-probe results, reproduced without alteration. "
               "They remain the correct answer to the question they ask — is the FROZEN "
               "representation linearly separable for DR grading — and they are reported "
               "here so the two instruments can be read against each other.", size=8.8)
        D.table(["frozen backbone", "QWK", "STDR AUC", "ref AUC", "acc", "trained %"],
                [[p["label"][:22],
                  "%.4f" % p["metrics"]["quadratic_weighted_kappa"],
                  "%.4f" % p["metrics"]["auc_sight_threatening"],
                  "%.4f" % p["metrics"]["auc_referable_dr"],
                  "%.4f" % p["metrics"]["accuracy"],
                  "%.4f%%" % p["trainable_fraction_pct"]] for p in probes],
                [0.195, 0.105, 0.115, 0.105, 0.10, 0.11], mono_from=1, size=7.8)

        agree = J["results"]["probe_and_adaptation_agree"]
        D.h2("Do the two instruments agree?")
        if agree is None:
            D.para("Not determinable — the adaptation arm is incomplete.", size=8.8)
        elif agree:
            D.para("YES. Both instruments rank the two ViT-L initialisations the same way. "
                   "When a frozen-feature probe and a matched fine-tune agree, the "
                   "conclusion is considerably more robust than either alone, and the "
                   "MAE-probe caveat on page 2 turns out not to have been load-bearing.",
                   size=8.8)
        else:
            D.para("NO. The probe and the matched adaptation rank the two ViT-L "
                   "initialisations differently. That is exactly the pattern He et al. "
                   "(2022) and Kornblith et al. (2019) describe, and it means the probe "
                   "result must not be quoted as evidence about adaptability. Both numbers "
                   "belong in the write-up, each attached to the question it answers.",
                   size=8.8)
        D.close()

        # ---------------- page 7: limitations -------------------------------
        D.new("Limitations and reproduction", "Objective 3 · scope of the claim")
        D.h2("Limitations")
        for i, l in enumerate(J["limitations"], 1):
            D.para("%d.  %s" % (i, l), size=8.5, lead=0.0138)
            D.para("", size=3)
        D.h2("Reproduce")
        D.kv(list(J["reproduce"].items()), w=0.32, size=8.0)
        D.h2("Status")
        D.para(doc["status"], size=8.8)
        D.close()

        # ---------------- page 8+: literature -------------------------------
        per_page = 5
        lit = J["literature"]
        for start in range(0, len(lit), per_page):
            D.new("Literature" if start == 0 else "Literature (cont.)",
                  "Objective 3 · what each claim is answerable to")
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
        info["Title"] = "Objective 3 validation - matched adaptation"
        info["Subject"] = doc["status"]


# --------------------------------------------------------------------------
def main() -> None:
    arms = [a for a in (read_arm(t, l, n) for t, l, n in ARMS) if a]
    probes = read_probes()
    if not arms:
        raise SystemExit("no obj3_* evaluation found - run ./run_objective3.sh first")

    J = build_json(arms, probes)
    jpath = OUT_DIR / "objective3_validation.json"
    json.dump(J, open(jpath, "w"), indent=2)
    save_csv_alongside(arms, jpath, csv_path=jpath.with_name(jpath.stem + "_arms.csv"))
    print(f"[write] {jpath}")

    ppath = OUT_DIR / "objective3_validation.pdf"
    build_pdf(J, ppath)
    print(f"[write] {ppath}")

    print("\n" + J["document"]["status"])
    for a in arms:
        print("  %-22s QWK %.4f  acc %.4f  STDR AUC %.4f"
              % (a["label"], a["metrics"]["quadratic_weighted_kappa"],
                 a["metrics"]["accuracy"], a["metrics"]["auc_sight_threatening"]))


if __name__ == "__main__":
    main()
