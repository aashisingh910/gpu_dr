#!/usr/bin/env python3
"""Step 20 - Objective 3, complete: one document, both instruments.

Consolidates the two halves of the Objective 3 evidence into a single PDF:

    the efficiency claim   - parameter counts, LoRA allocation, memory envelope
    the domain claim       - measured with TWO instruments:
                               1. linear probing of the frozen representation
                                  (step 13; answers "are these features already
                                  linearly separable")
                               2. matched GLA-LoRA adaptation, RETFound vs
                                  ImageNet initialisation, everything else
                                  pinned identical (run_objective3.sh; answers
                                  the question Objective 3 actually asks)

Instrument 2 is the validation.  If its runs are not yet complete the document
still builds, with that section marked IN PROGRESS rather than guessed at, so
re-running this script after the arms finish produces the final version at the
same path.

    python scripts/20_objective3_complete.py

Writes  outputs/objective3_complete.json
        outputs/objective3_complete.pdf
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
from dr.config import OUT_DIR  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402


def _load(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / stem)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_o1 = _load("15_objective1_report.py", "o1report")
_o3 = _load("18_objective3_report.py", "o3report")
_o3v = _load("19_objective3_validation.py", "o3val")

Doc, GRADES, INK, MUTED, RULE, GRADE_C = (
    _o1.Doc, _o1.GRADES, _o1.INK, _o1.MUTED, _o1.RULE, _o1.GRADE_C)
METRICS = _o3v.METRICS


def merge_literature() -> list:
    """Union of both packs' bibliographies, deduplicated on the citation."""
    seen, out = set(), []
    for ref in list(_o3.LITERATURE) + list(_o3v.LITERATURE):
        key = ref["cite"].split("(")[0].strip() + ref["cite"][:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(ref))
    return out


# ==========================================================================
def build_json() -> dict:
    probes = _o3.read_probes()
    lora = _o3.read_lora()
    frac = _o3.read_trainable_fractions()
    mem = _o3.read_memory_envelope()
    arms = [a for a in (_o3v.read_arm(t, l, n) for t, l, n in _o3v.ARMS) if a]

    a = {x["run"]: x for x in arms}
    ret_a, imn_a = a.get("obj3_retfound"), a.get("obj3_imagenet")
    delta = {}
    if ret_a and imn_a:
        for key, lab, higher in METRICS:
            rv, iv = ret_a["metrics"].get(key), imn_a["metrics"].get(key)
            if rv is None or iv is None:
                continue
            d = rv - iv
            delta[key] = {"label": lab, "retfound": rv, "imagenet": iv, "delta": d,
                          "higher_is_better": higher,
                          "retfound_better": bool(d > 0) if higher else bool(d < 0)}
    adapt_won = delta.get("quadratic_weighted_kappa", {}).get("retfound_better")

    pr = {p["label"]: p["test"]["quadratic_weighted_kappa"] for p in probes}
    probe_won = pr.get("RETFound ViT-L/16", -9) > pr.get("ImageNet ViT-L/16", 9)
    min_frac = min((p["trainable_fraction_pct"] for p in probes), default=None)
    lora_frac = frac[0]["stages"][-1]["pct"] if frac else None

    if adapt_won is None:
        status = ("EFFICIENCY HALF: MET. DOMAIN HALF: VALIDATION IN PROGRESS - the matched "
                  "adaptation arms have not both produced an evaluation yet. The linear "
                  "probe alone does not settle it; see the instrument argument.")
    elif adapt_won:
        status = ("MET. Under the adaptation budget the objective actually describes, the "
                  "RETFound initialisation beats the ImageNet initialisation on the primary "
                  "endpoint with the backbone frozen in both arms, and the efficiency half "
                  "was already supported. Both halves hold.")
    else:
        status = ("EFFICIENCY HALF: MET, strongly. DOMAIN HALF: NOT MET under a matched "
                  "lightweight adaptation budget - the RETFound initialisation does not "
                  "beat the ImageNet initialisation on the primary endpoint.")

    return {
        "document": {
            "title": "Objective 3 - complete evidence and validation",
            "objective_as_stated_in_code":
                'a large model can be adapted "efficiently, enabling lightweight '
                'optimization without full retraining"; research gap 3 says generic '
                'pretrained backbones (ImageNet/ResNet/EfficientNet) lack retinal '
                'domain awareness',
            "source_of_wording": "scripts/13_linear_probe.py:2-5",
            "generated": date.today().isoformat(),
            "generator": "scripts/20_objective3_complete.py",
            "status": status,
        },
        "claims_under_test": {
            "claim_A_efficiency": {
                "statement": "A ~300M-parameter backbone can be adapted to DR grading "
                             "without full retraining.",
                "instrument": "trained-parameter fraction, in two regimes",
                "verdict": "SUPPORTED. %s of the backbone trained in the probe regime; "
                           "%s of the network trained in the GLA-LoRA regime, with the "
                           "backbone never unfrozen."
                           % (f"{min_frac:.4f}%" if min_frac else "-",
                              f"{lora_frac:.2f}%" if lora_frac else "-"),
            },
            "claim_B_retinal_domain_awareness": {
                "statement": "Retinal-pretrained (RETFound) features carry domain awareness "
                             "that generic ImageNet features lack.",
                "instrument_1": "linear probe of the frozen representation",
                "instrument_1_verdict":
                    "NOT SUPPORTED (RETFound QWK %.4f vs ImageNet ViT-L %.4f). This "
                    "instrument is biased against an MAE backbone by construction."
                    % (pr.get("RETFound ViT-L/16", float("nan")),
                       pr.get("ImageNet ViT-L/16", float("nan"))),
                "instrument_2": "matched GLA-LoRA adaptation, initialisation the only "
                                "variable",
                "instrument_2_verdict": (
                    "IN PROGRESS" if adapt_won is None else
                    ("SUPPORTED (delta %+.4f QWK in favour of RETFound)"
                     % delta["quadratic_weighted_kappa"]["delta"] if adapt_won else
                     "NOT SUPPORTED (delta %+.4f QWK)"
                     % delta["quadratic_weighted_kappa"]["delta"])),
            },
        },
        "instrument_argument": {
            "point": "Objective 3 claims ADAPTABILITY. A linear probe freezes every weight "
                     "and adapts nothing, so it measures a different property.",
            "why_it_matters_here": "RETFound is a masked autoencoder. He et al. (2022) "
                                   "report that MAE representations score poorly under "
                                   "linear probing and strongly under fine-tuning, because "
                                   "the reconstruction objective never asks for linear "
                                   "class separability while a supervised ImageNet "
                                   "objective does explicitly.",
            "checkpoint_evidence": "the published checkpoint contains 296 tensors, no "
                                   "decoder keys and a classification head - it is the "
                                   "released MAE encoder, and RETFound's own downstream "
                                   "protocol is fine-tuning, not probing",
            "consequence": "the probe result is retained as a control, not as the test",
        },
        "method": {
            "probe_protocol": "every backbone weight frozen; features extracted once, "
                              "forward-only; one multinomial logistic regression fitted on "
                              "top with class_weight='balanced'; C selected on validation "
                              "QWK, never on test",
            "probe_feature": "global CLS token concatenated with the mean of the local "
                             "lesion-crop CLS tokens",
            "adaptation_protocol": "identical architecture, rank-allocation policy, stage "
                                   "schedule, learning rates, sampler, loss weights, data "
                                   "subset, batch size, crop geometry, validation split and "
                                   "selection criterion; backbone frozen in both arms; the "
                                   "initialisation is the only variable",
            "selection": "0.5*QWK + 0.3*MacroF1 + 0.2*minority recall on validation",
            "splits": {"train": 4196, "val": 920, "test": 926,
                       "note": "patient-disjoint"},
            "loader_guarantee": "the RETFound loader refuses to proceed below 95% tensor "
                                "coverage; the recorded run matched 294 tensors, 0 missing, "
                                "0 unexpected, 100.0% coverage",
        },
        "results": {
            "linear_probes": probes,
            "adaptation_arms": arms,
            "paired_deltas": delta,
            "gla_lora_allocation": lora,
            "trainable_fraction_by_stage": frac,
            "memory_envelope": mem,
            "instruments_agree": (None if adapt_won is None
                                  else bool(probe_won == adapt_won)),
        },
        "headline_numbers": {
            "probe_qwk": pr,
            "smallest_trained_fraction_pct": min_frac,
            "lora_trained_fraction_pct": lora_frac,
            "adaptation_delta_qwk": delta.get("quadratic_weighted_kappa", {}).get("delta"),
            "metrics_favouring_retfound": sum(1 for v in delta.values()
                                              if v["retfound_better"]) or None,
            "metrics_compared": len(delta) or None,
        },
        "threats_to_validity": _o3.LITERATURE and [
            "MAE VERSUS SUPERVISED PRETRAINING. RETFound is an MAE; the ImageNet arms are "
            "supervised. Under linear probing this favours the supervised arms by "
            "construction. This is the reason the adaptation instrument was added, and it "
            "does not apply to that instrument.",
            "PREPROCESSING. The cache holds A1/A2 output rather than raw fundus pixels. "
            "Measured per-channel statistics are close (cache 0.405/0.295/0.214 against raw "
            "0.391/0.268/0.181), so this is a weaker threat than it first appeared, but "
            "CLAHE changes local structure that global statistics do not capture.",
            "SHORT SCHEDULE. The adaptation arms run a reduced schedule on a sampled subset "
            "under a wall-clock budget. Both arms receive the same budget, so the "
            "comparison is fair, but neither is trained to convergence.",
            "REDUCED GEOMETRY. Two local crops and a 224 px global view in both instruments, "
            "against the flagship's six crops at 448 px.",
            "SINGLE SEED. No confidence interval on the difference; QWK gaps under roughly "
            "0.05 should be read as no measurable difference.",
            "BACKBONE FROZEN THROUGHOUT. This tests the initialisation under a LoRA-only "
            "budget. He et al. 2022 report the MAE advantage growing once blocks unfreeze, "
            "which this design deliberately does not test.",
        ],
        "reproduce": {
            "linear probes": "python scripts/13_linear_probe.py --backbone "
                             "vit_large_patch16_224 --retfound <ckpt|none> --tag <tag>",
            "adaptation arms": "./run_objective3.sh",
            "this document": "python scripts/20_objective3_complete.py",
            "machine-readable twin": "outputs/objective3_complete.json",
        },
        "literature": merge_literature(),
    }


# ==========================================================================
def build_pdf(J, path: Path) -> None:
    doc = J["document"]
    R = J["results"]
    probes, arms, delta = R["linear_probes"], R["adaptation_arms"], R["paired_deltas"]
    lora, frac, mem = R["gla_lora_allocation"], R["trainable_fraction_by_stage"], R["memory_envelope"]
    hn = J["headline_numbers"]
    a = {x["run"]: x for x in arms}
    ret_a, imn_a = a.get("obj3_retfound"), a.get("obj3_imagenet")
    adapt_won = delta.get("quadratic_weighted_kappa", {}).get("retfound_better")
    ret_p = next((p for p in probes if "retfound" in p["tag"]), None)

    with PdfPages(path) as pdf:
        D = Doc(pdf)

        # ---------------- 1 verdict ----------------------------------------
        D.new("Objective 3 — efficient adaptation of a foundation model",
              "Complete evidence and validation · generated %s" % doc["generated"])
        D.para("Objective as stated in the code (scripts/13_linear_probe.py:2-5): "
               + doc["objective_as_stated_in_code"], size=8.6, color=MUTED)

        D.h2("Verdict")
        col = ("#1f5c34" if adapt_won else ("#8a2b2b" if adapt_won is False else "#8a5a1b"))
        D.para(doc["status"], size=9.2, color=col)

        D.h2("The objective contains two claims; they are tested separately")
        c = J["claims_under_test"]
        D.para("A — EFFICIENCY", size=8.6, color=MUTED)
        D.para(c["claim_A_efficiency"]["statement"], size=8.6, indent=0.012)
        D.para(c["claim_A_efficiency"]["verdict"], size=8.6, indent=0.012, color="#1f5c34")
        D.para("", size=4)
        D.para("B — RETINAL DOMAIN AWARENESS", size=8.6, color=MUTED)
        D.para(c["claim_B_retinal_domain_awareness"]["statement"], size=8.6, indent=0.012)
        D.para("instrument 1, frozen probe:  "
               + c["claim_B_retinal_domain_awareness"]["instrument_1_verdict"],
               size=8.4, indent=0.012, color="#8a2b2b")
        D.para("instrument 2, matched adaptation:  "
               + c["claim_B_retinal_domain_awareness"]["instrument_2_verdict"],
               size=8.4, indent=0.012,
               color=("#1f5c34" if adapt_won else
                      ("#8a2b2b" if adapt_won is False else "#8a5a1b")))

        D.h2("Headline numbers")
        rows = [("smallest trained fraction (probe)",
                 "%.4f%% of 303,301,632 frozen parameters"
                 % hn["smallest_trained_fraction_pct"]
                 if hn["smallest_trained_fraction_pct"] else "-"),
                ("trained fraction (GLA-LoRA path)",
                 "%.2f%% of the network, backbone never unfrozen"
                 % hn["lora_trained_fraction_pct"]
                 if hn["lora_trained_fraction_pct"] else "-"),
                ("probe QWK, RETFound vs ImageNet ViT-L",
                 "%.4f  vs  %.4f" % (hn["probe_qwk"].get("RETFound ViT-L/16", float("nan")),
                                     hn["probe_qwk"].get("ImageNet ViT-L/16", float("nan"))))]
        if hn["adaptation_delta_qwk"] is not None:
            rows.append(("adaptation delta QWK",
                         "%+.4f (%d of %d metrics favour RETFound)"
                         % (hn["adaptation_delta_qwk"], hn["metrics_favouring_retfound"],
                            hn["metrics_compared"])))
        else:
            rows.append(("adaptation delta QWK", "pending — arms still running"))
        D.kv(rows, w=0.40, size=8.2)

        D.h2("Contents")
        D.para("p2  the objective, and the two claims separated\n"
               "p3  claim A — efficiency: every parameter count\n"
               "p4  claim A — the memory envelope that made it necessary\n"
               "p5  claim B, instrument 1 — the frozen linear probe\n"
               "p6  why a linear probe cannot settle this objective\n"
               "p7  claim B, instrument 2 — the matched adaptation design\n"
               "p8  instrument 2 — results\n"
               "p9  GLA-LoRA rank allocation and formulas\n"
               "p10 threats to validity\n"
               "p11 reproduction\n"
               "p12 literature", size=8.6)
        D.close()

        # ---------------- 2 objective + method ------------------------------
        D.new("The objective and the protocols", "Objective 3 · method")
        m = J["method"]
        D.h2("Wording, quoted not reconstructed")
        D.para(doc["objective_as_stated_in_code"], size=8.8)
        D.para("source: %s" % doc["source_of_wording"], size=8.0, color=MUTED)

        D.h2("Instrument 1 — linear probing")
        D.para(m["probe_protocol"], size=8.6)
        D.para("    f = [ CLS(global 224px view) ; mean_c CLS(local crop c) ]\n"
               "    yhat = argmax_k  w_k' z(f) + b_k ,   z = train-split z-score",
               size=8.4, mono=True)
        D.kv([("feature", m["probe_feature"]),
              ("splits", "train %d / val %d / test %d, patient-disjoint"
               % (m["splits"]["train"], m["splits"]["val"], m["splits"]["test"])),
              ("selection", "validation QWK over C in {0.001, 0.01, 0.1, 1.0}")],
             w=0.24, size=8.2)

        D.h2("Instrument 2 — matched adaptation")
        D.para(m["adaptation_protocol"], size=8.6)
        D.para("    W_adapted = W_frozen + B A ,   B in R^{d x r} ,  A in R^{r x k}",
               size=8.4, mono=True)
        D.kv([("selection", m["selection"]),
              ("driver", "run_objective3.sh"),
              ("arm A", "RETFound MAE ViT-L/16"),
              ("arm B", "ImageNet ViT-L/16 (--allow-no-retfound --pretrained)")],
             w=0.24, size=8.2)

        D.h2("The weights really are RETFound")
        D.para(m["loader_guarantee"] + ".", size=8.6)
        D.para("The loader hard-fails rather than falling back to ImageNet silently, so "
               "'the checkpoint did not load' is excluded as an explanation of any result "
               "in this document.", size=8.2, color=MUTED)
        D.close()

        # ---------------- 3 efficiency --------------------------------------
        D.new("Claim A — efficiency", "Objective 3 · SUPPORTED")
        D.h2("Regime 1 — linear probing, the extreme case")
        D.table(["backbone", "frozen params", "trained", "trained %",
                 "screen NPV", "cleared"],
                [[p["label"], f"{p['frozen_backbone_params']:,}",
                  f"{p['trained_probe_params']:,}",
                  "%.4f%%" % p["trainable_fraction_pct"],
                  "%.4f" % p["screening_test"]["npv"],
                  "%.1f%%" % (100 * p["screening_test"]["workload_reduction"])]
                 for p in probes],
                [0.175, 0.145, 0.09, 0.10, 0.10, 0.09], mono_from=1, size=7.6)
        D.para("A 5-way linear layer on frozen features reaches NPV %.4f while clearing "
               "%.1f%% of the workload, threshold fitted on validation and frozen."
               % (max(p["screening_test"]["npv"] for p in probes),
                  100 * max(p["screening_test"]["workload_reduction"] for p in probes)),
               size=8.4, color=MUTED)

        D.h2("Regime 2 — GLA-LoRA, the path that trained the system")
        rows = []
        for f_ in frac:
            for i, st in enumerate(f_["stages"]):
                rows.append([f_["label"][:24] if i == 0 else "",
                             "frozen head only" if i == 0 else "+ GLA-LoRA",
                             f"{st['trainable']:,}", f"{st['total']:,}",
                             "%.3f%%" % st["pct"]])
        D.table(["run", "regime", "trainable", "total", "fraction"], rows,
                [0.20, 0.155, 0.115, 0.115, 0.09], mono_from=2, size=7.6)
        D.para("The backbone is never unfrozen. What moves is the head, the A2 ALPP "
               "preprocessing weights, the lesion experts, the fusion stack and the LoRA "
               "adapters.", size=8.4, color=MUTED)

        D.h2("LoRA parameters allocated")
        D.table(["run", "blocks", "ranks", "mean r", "LoRA params"],
                [[l["label"][:24], str(l["n_blocks"]),
                  "%d-%d" % (l["rank_min"], l["rank_max"]),
                  "%.1f" % l["rank_mean"], f"{l['lora_params']:,}"] for l in lora],
                [0.215, 0.075, 0.09, 0.085, 0.12], mono_from=1, size=7.8)
        D.close()

        # ---------------- 4 memory envelope ---------------------------------
        D.new("Claim A — why efficiency was not optional",
              "Objective 3 · the measured hardware envelope")
        D.para("Measured on the 8 GB Apple M3 under a 2.67 GiB accelerator ceiling "
               "(PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5). Recorded in run_50pct.sh at the "
               "time the configuration was chosen.", size=8.8)
        if mem:
            D.table(["configuration", "peak memory", "img/s", "fits"],
                    [[r["configuration"][:38], r["peak_memory"],
                      "%.2f" % r["throughput_img_s"] if r["throughput_img_s"] else "-",
                      "yes" if r["fits_in_envelope"] else "no"] for r in mem],
                    [0.32, 0.135, 0.085, 0.07], mono_from=1, size=7.6)
        D.para("The result that matters: the largest and best-initialised backbone is the "
               "ONLY configuration that fits, and it fits only because the backbone stays "
               "frozen. A ViT-Base with 33% unfreeze and a ViT-Small with full unfreeze both "
               "exceed the ceiling. Parameter-efficient adaptation is not a convenience "
               "here — it is what made the flagship model trainable at all, and that is the "
               "efficiency claim demonstrated rather than asserted.", size=8.8)

        D.h2("What 'without full retraining' costs and buys")
        if lora and frac:
            l0, f0 = lora[0], frac[0]["stages"][-1]
            D.kv([("full fine-tune would train", f"{f0['total']:,} parameters"),
                  ("GLA-LoRA trains", f"{f0['trainable']:,} ({f0['pct']:.3f}%)"),
                  ("of which LoRA adapters", f"{l0['lora_params']:,}"),
                  ("backbone gradient", "none, in every stage"),
                  ("peak accelerator memory", mem[0]["peak_memory"] if mem else "-"),
                  ("throughput", "%.2f img/s" % mem[0]["throughput_img_s"]
                   if mem and mem[0]["throughput_img_s"] else "-")], w=0.32, size=8.2)
        D.close()

        # ---------------- 5 probe results -----------------------------------
        D.new("Claim B, instrument 1 — the frozen linear probe",
              "Objective 3 · what a frozen representation gives you")
        keys = [("quadratic_weighted_kappa", "QWK"), ("accuracy", "acc"),
                ("balanced_accuracy", "bal acc"), ("f1_macro", "F1"),
                ("auc_sight_threatening", "STDR AUC"), ("auc_referable_dr", "ref AUC"),
                ("mae_grade", "MAE")]
        D.h2("Full comparison, held-out test split (n=926)")
        D.table(["backbone"] + [k[1] for k in keys],
                [[p["label"][:20]] + ["%.4f" % p["test"][k[0]] for k in keys]
                 for p in probes],
                [0.155, 0.088, 0.078, 0.088, 0.075, 0.10, 0.09, 0.075],
                mono_from=1, size=7.4)

        D.h2("Per-grade recall")
        D.table(["backbone"] + GRADES,
                [[p["label"][:20]] + ["%.3f" % x for x in p["per_grade_recall"]]
                 for p in probes],
                [0.20, 0.10, 0.10, 0.11, 0.10, 0.10], mono_from=1, size=7.8)

        D.h2("Screening, threshold fitted on val and frozen")
        D.table(["backbone", "NPV", "sens", "spec", "cleared", "missed"],
                [[p["label"][:20], "%.4f" % p["screening_test"]["npv"],
                  "%.3f" % p["screening_test"]["sensitivity"],
                  "%.3f" % p["screening_test"]["specificity"],
                  "%.1f%%" % (100 * p["screening_test"]["workload_reduction"]),
                  str(p["screening_test"]["missed"])] for p in probes],
                [0.19, 0.10, 0.085, 0.085, 0.09, 0.08], mono_from=1, size=7.8)

        ax = D.axes(0.15, bottom_gap=0.075)
        xs = np.arange(len(probes))
        for i, (key, lab, c_) in enumerate([
                ("quadratic_weighted_kappa", "QWK", GRADE_C[1]),
                ("auc_sight_threatening", "STDR AUC", GRADE_C[3]),
                ("accuracy", "accuracy", GRADE_C[0])]):
            ax.bar(xs + i * 0.26 - 0.26, [p["test"][key] for p in probes],
                   width=0.24, color=c_, label=lab, edgecolor="white", lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([p["label"].replace(" ", "\n") for p in probes], fontsize=7.2)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16))
        ax.set_title("frozen-feature quality", fontsize=8.5, color=INK, pad=6)
        D.para("Taken at face value this says the retinal foundation model is behind. "
               "The next page explains why it cannot be taken at face value.",
               size=8.6, color=MUTED)
        D.close()

        # ---------------- 6 the instrument argument -------------------------
        D.new("Why a linear probe cannot settle this objective",
              "Objective 3 · the instrument argument")
        ia = J["instrument_argument"]
        D.h2("What the objective claims")
        D.para('"a large model can be adapted efficiently, enabling lightweight '
               'optimization without full retraining"', size=9)
        D.para("Every operative word is about ADAPTATION.", size=8.8)

        D.h2("What a linear probe measures")
        D.para("    freeze all W ;   fit only  w_k, b_k  in  argmax_k w_k'f(x) + b_k",
               size=8.6, mono=True)
        D.para("Nothing in f moves. The probe asks whether the pretrained features are "
               "ALREADY linearly separable for DR grading — a property of the "
               "representation, not of its adaptability.", size=8.6)

        D.h2("Why this is decisive for this backbone specifically")
        D.para(ia["why_it_matters_here"], size=8.8)
        D.para("Verified directly: " + ia["checkpoint_evidence"] + ".", size=8.6)
        D.para("So the probe compared an MAE against supervised models on the one axis the "
               "supervised models were explicitly trained for. The result is real and is "
               "retained on page 5, but it answers a question Objective 3 does not ask.",
               size=8.8)

        D.h2("The instrument that does match the claim")
        D.para("Give both initialisations the SAME lightweight adaptation budget and measure "
               "what each reaches. This tests the efficiency half (the budget is tiny, the "
               "backbone never unfreezes) and the domain half (the only variable is which "
               "weights the adaptation starts from) in one experiment.", size=8.8)
        D.para("A threat to validity honestly recorded in the previous pack becomes, on "
               "inspection, a reason to change instrument rather than a caveat to append. "
               "That is what pages 7 and 8 report.", size=8.6, color=MUTED)
        D.close()

        # ---------------- 7 adaptation design -------------------------------
        D.new("Claim B, instrument 2 — the matched adaptation",
              "Objective 3 · design")
        D.para(J["method"]["adaptation_protocol"] + ".", size=8.8)
        D.h2("Held constant across the two arms")
        D.para("architecture (ViT-L/16, 24 blocks, 1024-d)  ·  rank-allocation policy  ·  "
               "stage schedule  ·  learning rates  ·  sampler (n^-0.5 + hard-example "
               "mining)  ·  loss weights  ·  data subset and patient-disjoint splits  ·  "
               "batch size and gradient accumulation  ·  crop geometry  ·  validation "
               "split  ·  checkpoint-selection criterion  ·  evaluation script and decode "
               "rule", size=8.6)
        D.h2("The single variable")
        D.kv([("arm A", "RETFound MAE ViT-L/16 initialisation"),
              ("arm B", "ImageNet ViT-L/16 initialisation"),
              ("backbone gradient", "none, in either arm"),
              ("what adapts", "GLA-LoRA adapters + heads only")], w=0.24, size=8.4)
        for x in (ret_a, imn_a):
            if x and x.get("init_line"):
                D.para("%-20s %s" % (x["label"][:20], x["init_line"][:78]),
                       size=7.2, mono=True)
        if ret_a and imn_a and ret_a.get("gla_lora") and imn_a.get("gla_lora"):
            D.h2("Rank allocation, recalibrated per arm (policy fixed, weights differ)")
            D.table(["arm", "blocks", "rank range", "mean r", "LoRA params"],
                    [[x["label"][:22], str(len(x["gla_lora"]["ranks"])),
                      "%d-%d" % (x["gla_lora"]["rank_min"], x["gla_lora"]["rank_max"]),
                      "%.2f" % x["gla_lora"]["rank_mean"],
                      f"{x['gla_lora']['lora_params']:,}"] for x in (ret_a, imn_a)],
                    [0.205, 0.08, 0.105, 0.09, 0.12], mono_from=1, size=7.8)
        D.close()

        # ---------------- 8 adaptation results ------------------------------
        D.new("Claim B, instrument 2 — results", "Objective 3 · the validation")
        if not delta:
            D.h2("IN PROGRESS")
            D.para("The matched adaptation arms have not both produced an evaluation yet. "
                   "This section is deliberately left empty rather than filled with an "
                   "estimate; re-running scripts/20_objective3_complete.py after "
                   "./run_objective3.sh finishes replaces this page with the measured "
                   "result.", size=8.8, color="#8a5a1b")
            if arms:
                D.h2("Arms present so far")
                D.table(["arm", "n test", "QWK", "acc", "STDR AUC"],
                        [[x["label"][:24], str(x["n_images"]),
                          "%.4f" % x["metrics"]["quadratic_weighted_kappa"],
                          "%.4f" % x["metrics"]["accuracy"],
                          "%.4f" % x["metrics"]["auc_sight_threatening"]] for x in arms],
                        [0.22, 0.09, 0.10, 0.10, 0.11], mono_from=1, size=7.8)
        else:
            D.h2("Paired comparison, held-out test split (n=%d)" % ret_a["n_images"])
            D.table(["metric", "RETFound", "ImageNet", "delta", "favours"],
                    [[delta[k]["label"], "%.4f" % delta[k]["retfound"],
                      "%.4f" % delta[k]["imagenet"], "%+.4f" % delta[k]["delta"],
                      "RETFound" if delta[k]["retfound_better"] else "ImageNet"]
                     for k, _, _ in METRICS if k in delta],
                    [0.155, 0.115, 0.115, 0.105, 0.115], mono_from=1, size=7.8)
            D.para("delta = RETFound - ImageNet. For MAE grade lower is better and the "
                   "'favours' column accounts for it.", size=8.2, color=MUTED)

            ax = D.axes(0.15, bottom_gap=0.072)
            ks = [k for k, _, _ in METRICS if k in delta and k != "mae_grade"]
            xs = np.arange(len(ks))
            ax.bar(xs - 0.2, [delta[k]["retfound"] for k in ks], width=0.38,
                   color=GRADE_C[4], label="RETFound init", edgecolor="white", lw=0.5)
            ax.bar(xs + 0.2, [delta[k]["imagenet"] for k in ks], width=0.38,
                   color=GRADE_C[1], label="ImageNet init", edgecolor="white", lw=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels([delta[k]["label"].replace(" ", "\n") for k in ks],
                               fontsize=6.6)
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16))
            ax.set_title("matched adaptation budget — only the initialisation differs",
                         fontsize=8.5, color=INK, pad=6)

            D.h2("Per-grade recall")
            D.table(["arm"] + GRADES,
                    [[x["label"][:22]] + ["%.3f" % v for v in x["per_grade_recall"]]
                     for x in (ret_a, imn_a)],
                    [0.21, 0.098, 0.098, 0.105, 0.098, 0.09], mono_from=1, size=7.8)

            agree = R["instruments_agree"]
            D.h2("Do the two instruments agree?")
            D.para(("YES — both rank the two ViT-L initialisations the same way. When a "
                    "frozen probe and a matched adaptation agree, the conclusion is more "
                    "robust than either alone." if agree else
                    "NO — the probe and the matched adaptation rank the two "
                    "initialisations differently. That is the pattern He et al. (2022) and "
                    "Kornblith et al. (2019) describe, and it means the probe result must "
                    "not be quoted as evidence about adaptability. Both numbers belong in "
                    "the write-up, each attached to the question it answers."), size=8.8)
            D.para("A QWK gap under roughly 0.05 is inside what this single-seed, "
                   "short-schedule setup can resolve. The measured gap is %+.4f."
                   % delta["quadratic_weighted_kappa"]["delta"], size=8.6)
        D.close()

        # ---------------- 9 GLA-LoRA ----------------------------------------
        D.new("GLA-LoRA rank allocation", "Objective 3 · how the adaptation budget is spent")
        D.para("Ranks are not uniform. Each block gets r_l from a pooled importance score:\n"
               "\n"
               "    S_l = 0.35*G_l + 0.45*L_l + 0.20*A_l\n"
               "    r_l = r_min + (r_max - r_min) * S_l ,   S_l >= 0.85 pinned to r_max\n"
               "\n"
               "    G_l   ||dL/dW|| inside block l on a calibration batch\n"
               "    L_l   |corr(patch-token energy, lesion density)|\n"
               "    A_l   share of block l's attention landing on lesion patches",
               size=8.6)
        flag = lora[0] if lora else None
        if flag:
            D.h2("Allocated ranks, %s" % flag["label"])
            D.para(str(flag["ranks"]), size=7.6, mono=True)
            D.kv([("blocks", flag["n_blocks"]),
                  ("rank range", "%d - %d" % (flag["rank_min"], flag["rank_max"])),
                  ("mean rank", "%.2f" % flag["rank_mean"]),
                  ("LoRA parameters", f"{flag['lora_params']:,}"),
                  ("blocks pinned to r_max", flag["n_at_max_rank"])], w=0.30, size=8.2)
            if flag.get("importance_S"):
                ax = D.axes(0.155, bottom_gap=0.078)
                b = np.arange(flag["n_blocks"])
                for key, lab, c_, ls in (("grad_G", "G  gradient", GRADE_C[0], "-"),
                                         ("lesion_L", "L  lesion tracking", GRADE_C[2], "-"),
                                         ("attn_A", "A  lesion attention", "#8a5a1b", "--"),
                                         ("importance_S", "S  pooled", GRADE_C[4], "-")):
                    v = flag.get(key)
                    if v:
                        ax.plot(b, v, ls, color=c_,
                                lw=1.8 if key == "importance_S" else 1.1, label=lab,
                                alpha=1.0 if key == "importance_S" else 0.75)
                ax.set_xlabel("transformer block (0 = closest to the pixels)",
                              fontsize=8, color=MUTED)
                ax.set_ylabel("normalised signal", fontsize=8, color=MUTED)
                ax.legend(fontsize=6.6, frameon=False, ncol=4, loc="upper center",
                          bbox_to_anchor=(0.5, -0.17))
                ax.set_title("the three signals and the rank they pool into",
                             fontsize=8.4, color=INK, pad=6)
                D.para("The gradient signal collapses with depth while the lesion signal "
                       "stays high, so pooling moves adaptation capacity toward the early "
                       "and middle blocks rather than the last ones.", size=8.2, color=MUTED)
        D.close()

        # ---------------- 10 threats ----------------------------------------
        D.new("Threats to validity", "Objective 3 · what would change the conclusion")
        for i, t in enumerate(J["threats_to_validity"], 1):
            head, _, body = t.partition(". ")
            D.para("%d.  %s" % (i, head), size=8.8, color=INK)
            D.para(body, size=8.4, indent=0.016, color=MUTED)
            D.para("", size=4)
        D.h2("What each instrument licenses")
        D.para("The frozen probe licenses statements about the representation as it comes "
               "out of pretraining. The matched adaptation licenses statements about which "
               "initialisation adapts better under a fixed lightweight budget. Neither "
               "licenses a statement about full fine-tuning with unfrozen blocks, which "
               "nothing in this repository has run.", size=8.8)
        D.close()

        # ---------------- 11 reproduce --------------------------------------
        D.new("Reproduction", "Objective 3 · every number above, from scratch")
        for k, v in J["reproduce"].items():
            D.para(k, size=8.2, color=MUTED)
            D.para(v, size=7.6, mono=True, indent=0.016, lead=0.0132)
            D.para("", size=3)
        D.h2("Artefacts read")
        D.para("outputs/linear_probe_{retfound_vitl,imagenet_vitl,imagenet_vits}.json\n"
               "outputs/obj3_{retfound,imagenet}/evaluation.json, history.json, gla_lora.json\n"
               "outputs/<run>/gla_lora.json          rank allocation, LoRA parameters\n"
               "outputs/logs/*_train.log             trainable fraction per stage\n"
               "run_50pct.sh                         recorded memory envelope\n"
               "run_objective3.sh                    the matched adaptation driver",
               size=7.6, mono=True)
        D.h2("Status")
        D.para(doc["status"], size=8.8)
        D.close()

        # ---------------- 12 literature -------------------------------------
        lit = J["literature"]
        per_page = 5
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
        info["Title"] = "Objective 3 - complete evidence and validation"
        info["Subject"] = doc["status"]


# ==========================================================================
def main() -> None:
    J = build_json()
    jpath = OUT_DIR / "objective3_complete.json"
    json.dump(J, open(jpath, "w"), indent=2)
    save_csv_alongside(J, jpath)
    print(f"[write] {jpath}")
    ppath = OUT_DIR / "objective3_complete.pdf"
    build_pdf(J, ppath)
    print(f"[write] {ppath}")
    print("\n" + J["document"]["status"])
    d = J["headline_numbers"]["adaptation_delta_qwk"]
    print("adaptation delta QWK: %s"
          % ("%+.4f" % d if d is not None else "pending - arms still running"))


if __name__ == "__main__":
    main()
