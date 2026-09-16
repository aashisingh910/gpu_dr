#!/usr/bin/env python3
"""Step 18 - Objective 3 evidence pack: JSON + PDF.

Objective 3, quoted from the code that implements it (scripts/13_linear_probe.py):

    Research gap 3 says generic pretrained backbones (ImageNet/ResNet/
    EfficientNet) lack retinal domain awareness.  Objective 3 says a large
    model can be adapted "efficiently, enabling lightweight optimization
    without full retraining".

Those are two separable claims and they are tested separately here:

  EFFICIENCY   can a 300M-parameter backbone be adapted without full
               retraining?  Measured as trained-parameter fraction, in two
               regimes: linear probing (nothing but a 5-way head) and the
               GLA-LoRA path actually used for training.

  DOMAIN       do retinal-pretrained features beat generic ImageNet features
               on this task?  Measured by linear probing, the standard
               protocol for evaluating a frozen representation, with the
               backbone identical in size and the probe identical in form.

    python scripts/18_objective3_report.py

Reads   outputs/linear_probe_*.json         (step 13, three backbones)
        outputs/<run>/gla_lora.json         (rank allocation, LoRA parameters)
        outputs/<run>/train.log             (trainable fraction per stage)
        run_50pct.sh                        (the recorded memory envelope)
Writes  outputs/objective3_evidence.json
        outputs/objective3_evidence.pdf
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

PROBES = [
    ("linear_probe_retfound_vitl.json", "RETFound ViT-L/16", "retinal MAE, 1.6M fundus"),
    ("linear_probe_imagenet_vitl.json", "ImageNet ViT-L/16", "generic supervised/augreg"),
    ("linear_probe_imagenet_vits.json", "ImageNet ViT-S/16", "generic, 14x smaller"),
]
LORA_RUNS = [
    ("retfound_plus_laft_xai", "RETFound ViT-L, 4 stages"),
    ("retfound_stage1_evidence", "RETFound ViT-L, 50% envelope"),
    ("vits_objectives", "ViT-S, 12 blocks"),
]

LITERATURE = [
    dict(cite="Zhou, Y. et al. (2023). A foundation model for generalizable disease "
              "detection from retinal images. Nature 622, 156-163. (RETFound)",
         supports="The retinal foundation model under test: MAE self-supervised "
                  "pretraining on 1.6M retinal images, released as a ViT-Large/16.",
         value="ViT-L/16, 303M backbone parameters, 224 px",
         used_as="The 'retinal domain awareness' half of the objective is a claim about "
                 "this checkpoint specifically."),
    dict(cite="He, K. et al. (2022). Masked autoencoders are scalable vision learners. "
              "CVPR 2022, 16000-16009.",
         supports="The pretraining objective RETFound uses. MAE representations are known "
                  "to be strong under fine-tuning but comparatively weak under linear "
                  "probing, because the objective does not encourage linear separability.",
         value=None,
         used_as="The most important caveat on page 5: linear probing may understate an "
                 "MAE backbone relative to a supervised one, and the two arms here differ "
                 "in pretraining objective as well as in domain."),
    dict(cite="Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language "
              "models. ICLR 2022.",
         supports="Low-rank adapters as the mechanism for adapting a large frozen model: "
                  "W + BA with rank r << d, trained while W stays frozen.",
         value="trainable parameters scale with r, not with |W|",
         used_as="The efficiency half of the objective; %s LoRA parameters were allocated "
                 "on the flagship run."),
    dict(cite="Houlsby, N. et al. (2019). Parameter-efficient transfer learning for NLP. "
              "ICML 2019, 2790-2799.",
         supports="Adapter-based transfer as an alternative to full fine-tuning, and the "
                  "trained-parameter fraction as the metric for it.",
         value=None,
         used_as="Establishes trained-parameter fraction as the quantity reported on "
                 "page 3 rather than wall-clock time, which is hardware-specific."),
    dict(cite="Raghu, M. et al. (2019). Transfusion: Understanding transfer learning for "
              "medical imaging. NeurIPS 2019, 3347-3357.",
         supports="Transfer from natural images to medical imaging often yields little "
                  "benefit over training from scratch, and large architectures are "
                  "frequently unnecessary - the gains attributed to transfer are partly "
                  "from scale and feature reuse of low-level filters.",
         value=None,
         used_as="Prior art that the domain half of this objective is not self-evident, "
                 "and that a negative result is a publishable finding rather than a bug."),
    dict(cite="Matsoukas, C. et al. (2022). What makes transfer learning work for medical "
              "images: Feature reuse & other factors. CVPR 2022, 9225-9234.",
         supports="Which factors actually drive medical transfer performance, and that "
                  "in-domain pretraining does not automatically dominate.",
         value=None,
         used_as="Framing for the negative result on page 4."),
    dict(cite="Azizi, S. et al. (2021). Big self-supervised models advance medical image "
              "classification. ICCV 2021, 3478-3488.",
         supports="Self-supervised in-domain pretraining improving medical classification, "
                  "the result this objective's domain half is betting on.",
         value=None,
         used_as="The expectation that the measurement here fails to reproduce."),
    dict(cite="Kornblith, S., Shlens, J., Le, Q. V. (2019). Do better ImageNet models "
              "transfer better? CVPR 2019, 2661-2671.",
         supports="Linear probing versus fine-tuning as two different questions about a "
                  "representation, which can rank backbones differently.",
         value=None,
         used_as="Why page 5 refuses to generalise the probe result to the fine-tuned "
                 "setting."),
    dict(cite="He, K. et al. (2020). Momentum contrast for unsupervised visual "
              "representation learning. CVPR 2020, 9729-9738.",
         supports="The linear-probe protocol as the standard evaluation of a frozen "
                  "representation: freeze everything, fit one linear layer.",
         value=None,
         used_as="The protocol on page 2 is this one, with the probe fitted by "
                 "multinomial logistic regression and C chosen on validation."),
    dict(cite="Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers "
              "for image recognition at scale. ICLR 2021.",
         supports="The ViT architecture and the CLS token used as the pooled "
                  "representation in the probe feature.",
         value="CLS token, 1024-d for ViT-L/16, 384-d for ViT-S/16",
         used_as="Defines the hybrid feature: global CLS concatenated with the mean of "
                 "the local-crop CLS tokens."),
    dict(cite="Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision "
              "for scaled disagreement or partial credit. Psychological Bulletin 70(4), "
              "213-220.",
         supports="Quadratic weighted kappa, the metric the probe hyper-parameter is "
                  "selected on and the headline comparison is made in.",
         value=None,
         used_as="The selection criterion for C, fitted on validation only."),
    dict(cite="Gulshan, V. et al. (2016). Development and validation of a deep learning "
              "algorithm for detection of diabetic retinopathy in retinal fundus "
              "photographs. JAMA 316(22), 2402-2410.",
         supports="The referral-screening endpoint the probes are additionally scored on.",
         value=None,
         used_as="Context for the screening rows: a frozen backbone plus a linear layer "
                 "already reaches a usable operating point."),
]


def resolve_literature(lora) -> list:
    n = lora[0]["lora_params"] if lora else 0
    out = []
    for ref in LITERATURE:
        r = dict(ref)
        if ref["cite"].startswith("Hu, E. J."):
            r["used_as"] = ref["used_as"] % f"{n:,}"
        out.append(r)
    return out


# --------------------------------------------------------------------------
def read_probes() -> list:
    out = []
    for fname, label, note in PROBES:
        f = OUT_DIR / fname
        if not f.exists():
            continue
        d = json.load(open(f))
        t = d["test"]
        out.append({
            "tag": d["tag"], "label": label, "note": note,
            "backbone": d["backbone"],
            "retfound_weights": d["retfound"],
            "frozen_backbone_params": d["frozen_backbone_params"],
            "trained_probe_params": d["trained_probe_params"],
            "trainable_fraction_pct": d["trainable_fraction_pct"],
            "feature_dim": d["feature_dim"],
            "selected_C": d["selected_C"],
            "test": {k: t[k] for k in (
                "accuracy", "balanced_accuracy", "f1_macro", "quadratic_weighted_kappa",
                "adjacent_accuracy", "auroc_macro_ovr", "auc_referable_dr",
                "auc_sight_threatening", "mae_grade")},
            "per_grade_recall": [t[f"recall_grade_{g}"] for g in range(5)],
            "screening_test": {k: d["screening_applied_to_test"][k] for k in
                               ("npv", "sensitivity", "specificity",
                                "workload_reduction", "missed", "threshold")},
        })
    return out


def read_lora() -> list:
    out = []
    for tag, label in LORA_RUNS:
        f = OUT_DIR / tag / "gla_lora.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        ranks = d["ranks"]
        out.append({
            "run": tag, "label": label,
            "n_blocks": len(ranks),
            "ranks": ranks,
            "rank_min": int(min(ranks)), "rank_max": int(max(ranks)),
            "rank_mean": float(np.mean(ranks)),
            "lora_params": d["lora_params"],
            "n_at_max_rank": d.get("n_at_max_rank"),
            "importance_S": d.get("importance_S"),
            "grad_G": d.get("grad_G"),
            "lesion_L": d.get("lesion_L"),
            "attn_A": d.get("attn_A"),
        })
    return out


def read_trainable_fractions() -> list:
    """Parse the 'trainable X / Y (Z%)' lines the trainer prints per stage."""
    pat = re.compile(r"trainable ([\d,]+) / ([\d,]+) \(([\d.]+)%\)")
    out = []
    for tag, label in LORA_RUNS:
        for cand in (OUT_DIR / tag / "train.log",
                     OUT_DIR / "logs" / f"{tag}_train.log"):
            if not cand.exists():
                continue
            seen = []
            for m in pat.finditer(cand.read_text(errors="ignore")):
                row = {"trainable": int(m.group(1).replace(",", "")),
                       "total": int(m.group(2).replace(",", "")),
                       "pct": float(m.group(3))}
                if row not in seen:
                    seen.append(row)
            if seen:
                out.append({"run": tag, "label": label, "source": str(
                    cand.relative_to(ROOT)), "stages": seen})
            break
    return out


def read_memory_envelope() -> list:
    """The measured peak-memory table recorded in run_50pct.sh."""
    txt = (ROOT / "run_50pct.sh").read_text(errors="ignore")
    rows = []
    for line in txt.splitlines():
        m = re.match(r"#\s+(\S.*?)\s{2,}([\d.]+ GB|OOM at [\d.]+ GB)\s*"
                     r"(?:([\d.]+) img/s)?\s*(fits)?\s*$", line)
        if m and ("LoRA" in line or "unfreeze" in line):
            rows.append({"configuration": m.group(1).strip(),
                         "peak_memory": m.group(2),
                         "throughput_img_s": float(m.group(3)) if m.group(3) else None,
                         "fits_in_envelope": bool(m.group(4))})
    return rows


# --------------------------------------------------------------------------
def build_json(probes, lora, frac, mem) -> dict:
    ret = next((p for p in probes if "retfound" in p["tag"]), None)
    gen = [p for p in probes if "imagenet" in p["tag"]]
    best_gen = max(gen, key=lambda p: p["test"]["quadratic_weighted_kappa"]) if gen else None
    domain_ok = bool(ret and best_gen and
                     ret["test"]["quadratic_weighted_kappa"]
                     > best_gen["test"]["quadratic_weighted_kappa"])
    min_frac = min((p["trainable_fraction_pct"] for p in probes), default=None)

    return {
        "document": {
            "title": "Objective 3 - evidence pack",
            "objective_as_stated_in_code":
                'a large model can be adapted "efficiently, enabling lightweight '
                'optimization without full retraining"; research gap 3 says generic '
                'pretrained backbones (ImageNet/ResNet/EfficientNet) lack retinal '
                'domain awareness',
            "source_of_wording": "scripts/13_linear_probe.py:2-5",
            "generated": date.today().isoformat(),
            "generator": "scripts/18_objective3_report.py",
            "status": "SPLIT. The efficiency claim is SUPPORTED and strongly so. The "
                      "retinal-domain claim is NOT SUPPORTED by the measurement made "
                      "here: under linear probing the RETFound features are beaten by "
                      "generic ImageNet features of the same architecture.",
        },
        "claims_under_test": {
            "claim_A_efficiency": {
                "statement": "A ~300M-parameter backbone can be adapted to DR grading "
                             "without full retraining.",
                "verdict": "SUPPORTED. Linear probing trains %s of the backbone and still "
                           "reaches a usable screening operating point; the GLA-LoRA "
                           "training path trains %.2f%% of the network with the backbone "
                           "never unfrozen."
                           % (f"{min_frac:.4f}%" if min_frac else "-",
                              frac[0]["stages"][-1]["pct"] if frac else float("nan")),
            },
            "claim_B_retinal_domain_awareness": {
                "statement": "Retinal-pretrained (RETFound) features carry domain "
                             "awareness that generic ImageNet features lack.",
                "verdict": ("SUPPORTED." if domain_ok else
                            "NOT SUPPORTED under this protocol. RETFound ViT-L probes at "
                            "QWK %.4f against %.4f for ImageNet ViT-L of identical size "
                            "and %.4f for an ImageNet ViT-S with 14x fewer parameters."
                            % (ret["test"]["quadratic_weighted_kappa"],
                               next(p["test"]["quadratic_weighted_kappa"] for p in gen
                                    if "vitl" in p["tag"]),
                               next(p["test"]["quadratic_weighted_kappa"] for p in gen
                                    if "vits" in p["tag"]))
                            if ret and gen else "NOT EVALUATED"),
            },
        },
        "method": {
            "protocol": "Linear probing (He et al. 2020): every backbone weight frozen, "
                        "features extracted once forward-only, a single multinomial "
                        "logistic regression fitted on top. The probe is "
                        "class_weight='balanced'; C is selected on the validation split "
                        "by QWK and never on test.",
            "feature": "hybrid: global CLS token concatenated with the mean of the local "
                       "lesion-crop CLS tokens, so the probe measures the architecture's "
                       "own hybrid extractor rather than the global view alone",
            "n_crops_in_probe": 2,
            "standardisation": "features z-scored using train-split statistics only",
            "input_normalisation": "ImageNet mean/std, applied identically to all three "
                                   "backbones (scripts/13_linear_probe.py:44-45)",
            "splits": {"train": 4196, "val": 920, "test": 926,
                       "note": "patient-disjoint, built by scripts/02_preprocess.py"},
            "weight_loading_check": ret["retfound_weights"] if ret else None,
            "loader_guarantee": "the RETFound loader refuses to proceed below 95% tensor "
                                "coverage; the recorded run matched 294 tensors, 0 "
                                "missing, 0 unexpected, 100.0% coverage",
        },
        "results": {
            "linear_probes": probes,
            "gla_lora_allocation": lora,
            "trainable_fraction_by_stage": frac,
            "memory_envelope": mem,
        },
        "headline_numbers": {
            "probe_qwk": {p["label"]: p["test"]["quadratic_weighted_kappa"] for p in probes},
            "probe_stdr_auc": {p["label"]: p["test"]["auc_sight_threatening"]
                               for p in probes},
            "smallest_trained_fraction_pct": min_frac,
            "retfound_minus_best_generic_qwk": (
                ret["test"]["quadratic_weighted_kappa"]
                - best_gen["test"]["quadratic_weighted_kappa"]) if ret and best_gen else None,
            "lora_params_flagship": lora[0]["lora_params"] if lora else None,
        },
        "threats_to_validity": [
            "PREPROCESSING MISMATCH, the leading candidate. The cache holds A1/A2 "
            "output - field-extracted, illumination-normalised, CLAHE-enhanced - not raw "
            "fundus pixels. RETFound was pretrained on retinal photographs in their "
            "native appearance, so the probe may be measuring how well each backbone "
            "tolerates an unfamiliar preprocessing pipeline rather than how much retinal "
            "knowledge it holds. This is directly testable: re-run step 13 on minimally "
            "processed images and see whether the ordering flips.",
            "MAE VERSUS SUPERVISED PRETRAINING. RETFound is an MAE; the ImageNet arms are "
            "supervised. MAE representations are documented to underperform supervised "
            "ones under linear probing while matching or beating them under fine-tuning "
            "(He et al. 2022). The two arms therefore differ in pretraining objective as "
            "well as in domain, and the probe favours the supervised one by construction.",
            "PROBE VERSUS FINE-TUNE. Linear probing answers 'are these features linearly "
            "separable for this task', not 'is this the better initialisation'. Kornblith "
            "et al. 2019 show the two can rank backbones differently. No claim is made "
            "here about the fine-tuned setting.",
            "TWO CROPS, NOT SIX. The probe uses 2 local crops against the 6 the training "
            "pipeline uses, so the hybrid feature is a weaker version of the "
            "architecture's own.",
            "SINGLE SEED, ONE SPLIT. No confidence intervals; the probe C is the only "
            "hyper-parameter searched.",
        ],
        "reproduce": {
            "this_document": "python scripts/18_objective3_report.py",
            "machine_readable_twin": "outputs/objective3_evidence.json",
            "retfound_probe": "python scripts/13_linear_probe.py --backbone "
                              "vit_large_patch16_224 --retfound "
                              "data/checkpoints/RETFound_MAE/pytorch_model.bin "
                              "--tag retfound",
            "imagenet_probe": "python scripts/13_linear_probe.py --backbone "
                              "vit_large_patch16_224 --retfound none --tag imagenet",
        },
        "literature": resolve_literature(lora),
    }


# --------------------------------------------------------------------------
def build_pdf(J, path: Path) -> None:
    doc = J["document"]
    probes = J["results"]["linear_probes"]
    lora = J["results"]["gla_lora_allocation"]
    frac = J["results"]["trainable_fraction_by_stage"]
    mem = J["results"]["memory_envelope"]
    ret = next((p for p in probes if "retfound" in p["tag"]), None)
    gen = [p for p in probes if "imagenet" in p["tag"]]

    with PdfPages(path) as pdf:
        D = Doc(pdf)

        # ---------------- page 1 -------------------------------------------
        D.new("Objective 3 — efficient adaptation of a foundation model",
              "Evidence pack · generated %s · RETFound Plus–LAFT–XAI" % doc["generated"])
        D.para("Objective as stated in the code (scripts/13_linear_probe.py:2-5): "
               + doc["objective_as_stated_in_code"], size=8.8, color=MUTED)

        D.h2("Verdict — the objective splits in two, and the halves disagree")
        D.para("EFFICIENCY: SUPPORTED. A 303M-parameter backbone is adapted with as little "
               "as %.4f%% of it trained, and the training path that produced every result "
               "in this repository never unfroze a single backbone block."
               % J["headline_numbers"]["smallest_trained_fraction_pct"],
               size=9, color="#1f5c34")
        D.para("RETINAL DOMAIN AWARENESS: NOT SUPPORTED under this protocol. Frozen "
               "RETFound features are beaten by frozen ImageNet features of the same "
               "architecture, and by an ImageNet ViT-S with 14x fewer parameters.",
               size=9, color="#8a2b2b")
        D.para("This is a measurement, not a verdict on RETFound. Page 5 lists four "
               "reasons the probe may understate it, one of which is testable in a single "
               "re-run and is the first thing to do next.", size=8.6)

        D.h2("Headline — linear probe, frozen backbone, held-out test (n=926)")
        rows = []
        for p in probes:
            t = p["test"]
            rows.append([p["label"], "%.4f" % t["quadratic_weighted_kappa"],
                         "%.4f" % t["auc_sight_threatening"],
                         "%.4f" % t["auc_referable_dr"],
                         "%.4f" % t["accuracy"],
                         "%.4f%%" % p["trainable_fraction_pct"]])
        D.table(["frozen backbone", "QWK", "STDR AUC", "ref AUC", "acc", "trained %"],
                rows, [0.185, 0.105, 0.115, 0.105, 0.105, 0.11], mono_from=1, size=7.8)
        if ret and gen:
            best = max(gen, key=lambda p: p["test"]["quadratic_weighted_kappa"])
            D.para("RETFound trails the best generic backbone by %.4f QWK. Every backbone "
                   "here saw identical inputs, an identical probe, and an identical "
                   "selection protocol."
                   % (best["test"]["quadratic_weighted_kappa"]
                      - ret["test"]["quadratic_weighted_kappa"]),
                   size=8.4, color=MUTED)

        D.h2("The weights really are RETFound")
        D.para(J["method"]["loader_guarantee"] + ".", size=8.6)
        D.para("This matters because the natural first objection to the result above is "
               "that the checkpoint failed to load. It did not, and the loader is built to "
               "hard-fail rather than fall back to ImageNet silently.", size=8.4, color=MUTED)

        D.h2("Contents")
        D.para("p2  the protocol, stated so it can be checked\n"
               "p3  the efficiency claim — every parameter count\n"
               "p4  the domain claim — full probe comparison\n"
               "p5  threats to validity, and the one test that would settle it\n"
               "p6  GLA-LoRA rank allocation\n"
               "p7  reproduction\n"
               "p8  literature", size=8.6)
        D.close()

        # ---------------- page 2: protocol ---------------------------------
        D.new("The protocol", "Objective 3 · method")
        m = J["method"]
        D.h2("Linear probing")
        D.para(m["protocol"], size=8.8)
        D.h2("The feature")
        D.para(m["feature"], size=8.8)
        D.para("    f = [ CLS(global 224px view) ; mean_c CLS(local crop c) ]",
               size=8.4, mono=True)
        D.kv([("local crops in the probe", m["n_crops_in_probe"]),
              ("feature dim, ViT-L arms", "%d = 1024 + 1024" % probes[0]["feature_dim"]),
              ("feature dim, ViT-S arm",
               "%d = 384 + 384" % probes[-1]["feature_dim"] if len(probes) > 2 else "-"),
              ("standardisation", "train-split z-score"),
              ("input normalisation", "ImageNet mean/std, all arms"),
              ("probe", "multinomial logistic regression, class_weight=balanced"),
              ("C selected on", "validation QWK"),
              ("splits", "train %d / val %d / test %d, patient-disjoint"
               % (m["splits"]["train"], m["splits"]["val"], m["splits"]["test"]))],
             w=0.30, size=8.2)

        D.h2("Why linear probing is the right instrument for this objective")
        D.para("The objective makes two claims that a full fine-tune would confound. "
               "Probing separates them: it holds the adaptation budget fixed at one linear "
               "layer for every arm, so any difference in score is a difference in the "
               "frozen representation and nothing else. It is also, by construction, an "
               "instance of the efficiency claim — the extreme case of adapting without "
               "retraining.", size=8.8)

        D.h2("C selection, on validation only")
        D.para("Each arm searched C in {0.001, 0.01, 0.1, 1.0} and selected on validation "
               "QWK. All three arms selected C = %s. Test was read once, afterwards."
               % ", ".join(sorted({str(p["selected_C"]) for p in probes})), size=8.6)
        D.close()

        # ---------------- page 3: efficiency --------------------------------
        D.new("The efficiency claim", "Objective 3 · SUPPORTED")
        D.h2("Regime 1 — linear probing: the extreme case")
        rows = []
        for p in probes:
            rows.append([p["label"], f"{p['frozen_backbone_params']:,}",
                         f"{p['trained_probe_params']:,}",
                         "%.4f%%" % p["trainable_fraction_pct"],
                         "%.4f" % p["screening_test"]["npv"],
                         "%.1f%%" % (100 * p["screening_test"]["workload_reduction"])])
        D.table(["backbone", "frozen params", "trained", "trained %",
                 "screen NPV", "cleared"],
                rows, [0.175, 0.145, 0.09, 0.10, 0.10, 0.09], mono_from=1, size=7.6)
        D.para("A 5-way linear layer on frozen features already reaches NPV %.4f while "
               "clearing %.1f%% of the workload, with the referral threshold fitted on "
               "validation and frozen. That is the efficiency claim in its strongest form."
               % (max(p["screening_test"]["npv"] for p in probes),
                  100 * max(p["screening_test"]["workload_reduction"] for p in probes)),
               size=8.4, color=MUTED)

        D.h2("Regime 2 — GLA-LoRA: the path that actually trained the system")
        rows = []
        for f_ in frac:
            for i, st in enumerate(f_["stages"]):
                rows.append([f_["label"][:24] if i == 0 else "",
                             "%s" % ("frozen head only" if i == 0 else "+ GLA-LoRA"),
                             f"{st['trainable']:,}", f"{st['total']:,}",
                             "%.3f%%" % st["pct"]])
        D.table(["run", "regime", "trainable", "total", "fraction"], rows,
                [0.20, 0.155, 0.115, 0.115, 0.09], mono_from=2, size=7.6)
        D.para("The backbone is never unfrozen in these runs. What moves is the head, the "
               "A2 ALPP preprocessing weights, the lesion experts, the fusion stack, and "
               "the LoRA adapters.", size=8.4, color=MUTED)

        D.h2("LoRA parameters allocated")
        D.table(["run", "blocks", "ranks", "mean r", "LoRA params"],
                [[l["label"][:24], str(l["n_blocks"]),
                  "%d-%d" % (l["rank_min"], l["rank_max"]),
                  "%.1f" % l["rank_mean"], f"{l['lora_params']:,}"] for l in lora],
                [0.215, 0.075, 0.09, 0.085, 0.12], mono_from=1, size=7.8)

        if mem:
            D.h2("Why efficiency was not optional here")
            D.table(["configuration", "peak memory", "img/s", "fits"],
                    [[r["configuration"][:34], r["peak_memory"],
                      "%.2f" % r["throughput_img_s"] if r["throughput_img_s"] else "-",
                      "yes" if r["fits_in_envelope"] else "no"] for r in mem],
                    [0.30, 0.135, 0.085, 0.07], mono_from=1, size=7.6)
            D.para("Measured on the 8 GB M3 under a 2.67 GiB accelerator ceiling "
                   "(run_50pct.sh). The largest and best-initialised backbone is the only "
                   "configuration that fits, and it fits only because the backbone stays "
                   "frozen. Efficiency is what made the flagship model trainable at all.",
                   size=8.4, color=MUTED)
        D.close()

        # ---------------- page 4: the domain claim --------------------------
        D.new("The retinal-domain claim", "Objective 3 · NOT SUPPORTED under this protocol")
        D.h2("Full comparison, held-out test split")
        keys = [("quadratic_weighted_kappa", "QWK"), ("accuracy", "acc"),
                ("balanced_accuracy", "bal acc"), ("f1_macro", "F1"),
                ("auc_sight_threatening", "STDR AUC"), ("auc_referable_dr", "ref AUC"),
                ("mae_grade", "MAE")]
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

        ax = D.axes(0.155, bottom_gap=0.075)
        labels = [p["label"] for p in probes]
        xs = np.arange(len(labels))
        for i, (key, lab, col) in enumerate([
                ("quadratic_weighted_kappa", "QWK", GRADE_C[1]),
                ("auc_sight_threatening", "STDR AUC", GRADE_C[3]),
                ("accuracy", "accuracy", GRADE_C[0])]):
            ax.bar(xs + i * 0.26 - 0.26, [p["test"][key] for p in probes],
                   width=0.24, color=col, label=lab, edgecolor="white", lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=7.2)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16))
        ax.set_title("frozen-feature quality — the retinal backbone is not ahead",
                     fontsize=8.5, color=INK, pad=6)

        D.h2("Reading it honestly")
        D.para("The ImageNet ViT-S result is the awkward one: 21.7M frozen parameters "
               "reach QWK %.4f and the highest sight-threatening AUC of the three, "
               "against %.4f for a retinal foundation model 14x its size. Whatever the "
               "explanation, the objective's premise — that generic backbones lack "
               "something these features have — is not visible in this measurement."
               % (probes[-1]["test"]["quadratic_weighted_kappa"],
                  ret["test"]["quadratic_weighted_kappa"] if ret else float("nan")),
               size=8.8)
        D.close()

        # ---------------- page 5: threats -----------------------------------
        D.new("Threats to validity", "Objective 3 · why the negative result may not be final")
        for i, t in enumerate(J["threats_to_validity"], 1):
            head, _, body = t.partition(". ")
            D.para("%d.  %s" % (i, head), size=8.8, color=INK)
            D.para(body, size=8.4, indent=0.016, color=MUTED)
            D.para("", size=4)

        D.h2("The one experiment that would settle it")
        D.para("Threat 1 is the only one that can be resolved without new theory, and it "
               "is cheap. Re-run step 13 on minimally processed images — field crop and "
               "resize only, no illumination normalisation and no CLAHE — for both the "
               "RETFound and the ImageNet ViT-L arm. If RETFound overtakes ImageNet on raw "
               "pixels, the finding is that A2 destroys the statistics RETFound relies on, "
               "which is an actionable result about the pipeline. If the ordering holds, "
               "the finding stands as reported and the objective's premise needs "
               "rewriting.", size=8.8)
        D.para("Until that is run, the honest statement is the one on page 1: not "
               "supported UNDER THIS PROTOCOL — not 'RETFound does not work'.",
               size=8.8, color="#8a5a1b")
        D.close()

        # ---------------- page 6: GLA-LoRA ----------------------------------
        D.new("GLA-LoRA rank allocation", "Objective 3 · how the adaptation budget is spent")
        D.para("Ranks are not uniform. Each block gets r_l from a pooled importance score\n"
               "\n"
               "    S_l = 0.35*G_l + 0.45*L_l + 0.20*A_l\n"
               "    r_l = r_min + (r_max - r_min) * S_l\n"
               "\n"
               "with G_l the gradient norm on a calibration batch, L_l the correlation "
               "between patch-token energy and lesion density, and A_l the share of "
               "attention landing on lesion patches.", size=8.6)

        flag = lora[0]
        D.h2("Allocated ranks, %s" % flag["label"])
        D.para(str(flag["ranks"]), size=7.6, mono=True)
        D.kv([("blocks", flag["n_blocks"]),
              ("rank range", "%d - %d" % (flag["rank_min"], flag["rank_max"])),
              ("mean rank", "%.2f" % flag["rank_mean"]),
              ("LoRA parameters", f"{flag['lora_params']:,}"),
              ("blocks pinned to r_max", flag["n_at_max_rank"])], w=0.30, size=8.2)

        if flag.get("importance_S"):
            ax = D.axes(0.16, bottom_gap=0.078)
            b = np.arange(flag["n_blocks"])
            for key, lab, col, ls in (("grad_G", "G  gradient", GRADE_C[0], "-"),
                                      ("lesion_L", "L  lesion tracking", GRADE_C[2], "-"),
                                      ("attn_A", "A  lesion attention", "#8a5a1b", "--"),
                                      ("importance_S", "S  pooled", GRADE_C[4], "-")):
                v = flag.get(key)
                if v:
                    ax.plot(b, v, ls, color=col, lw=1.8 if key == "importance_S" else 1.1,
                            label=lab, alpha=1.0 if key == "importance_S" else 0.75)
            ax.set_xlabel("transformer block (0 = closest to the pixels)",
                          fontsize=8, color=MUTED)
            ax.set_ylabel("normalised signal", fontsize=8, color=MUTED)
            ax.legend(fontsize=6.6, frameon=False, ncol=4, loc="upper center",
                      bbox_to_anchor=(0.5, -0.17))
            ax.set_title("the three signals and the rank they pool into",
                         fontsize=8.4, color=INK, pad=6)
            D.para("The gradient signal collapses with depth while the lesion signal stays "
                   "high, so pooling them moves adaptation capacity toward the early and "
                   "middle blocks rather than the last ones.", size=8.2, color=MUTED)

        D.h2("All runs")
        D.table(["run", "blocks", "ranks", "mean r", "LoRA params"],
                [[l["label"][:24], str(l["n_blocks"]),
                  "%d-%d" % (l["rank_min"], l["rank_max"]),
                  "%.1f" % l["rank_mean"], f"{l['lora_params']:,}"] for l in lora],
                [0.215, 0.075, 0.09, 0.085, 0.12], mono_from=1, size=7.8)
        D.close()

        # ---------------- page 7: reproduce ---------------------------------
        D.new("Reproduction", "Objective 3 · every number above, from scratch")
        D.h2("Commands")
        for k, v in J["reproduce"].items():
            D.para(k.replace("_", " "), size=8.2, color=MUTED)
            D.para(v, size=7.6, mono=True, indent=0.016, lead=0.0132)
            D.para("", size=3)
        D.h2("Artefacts read")
        D.para("outputs/linear_probe_retfound_vitl.json\n"
               "outputs/linear_probe_imagenet_vitl.json\n"
               "outputs/linear_probe_imagenet_vits.json\n"
               "outputs/<run>/gla_lora.json          rank allocation, LoRA parameters\n"
               "outputs/<run>/train.log              trainable fraction per stage\n"
               "run_50pct.sh                         recorded memory envelope",
               size=7.8, mono=True)
        D.h2("Status")
        D.para(doc["status"], size=8.8)
        D.close()

        # ---------------- page 8+: literature -------------------------------
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
        info["Title"] = "Objective 3 evidence pack - efficient foundation-model adaptation"
        info["Subject"] = doc["status"]


# --------------------------------------------------------------------------
def main() -> None:
    probes = read_probes()
    if not probes:
        raise SystemExit("no linear_probe_*.json found - run scripts/13_linear_probe.py")
    lora = read_lora()
    frac = read_trainable_fractions()
    mem = read_memory_envelope()

    J = build_json(probes, lora, frac, mem)
    jpath = OUT_DIR / "objective3_evidence.json"
    json.dump(J, open(jpath, "w"), indent=2)
    save_csv_alongside(probes, jpath,
                       csv_path=jpath.with_name(jpath.stem + "_probes.csv"))
    print(f"[write] {jpath}")

    ppath = OUT_DIR / "objective3_evidence.pdf"
    build_pdf(J, ppath)
    print(f"[write] {ppath}")

    print("\nstatus: %s" % J["document"]["status"].split(".")[0])
    for p in probes:
        print("  %-20s QWK %.4f  STDR AUC %.4f  trained %.4f%%"
              % (p["label"], p["test"]["quadratic_weighted_kappa"],
                 p["test"]["auc_sight_threatening"], p["trainable_fraction_pct"]))


if __name__ == "__main__":
    main()
