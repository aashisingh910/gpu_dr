#!/usr/bin/env python3
"""Step 24 - Methods and algorithms reference: every formula, in one document.

A single technical compendium of the whole system: each block A1-A10 with the
formulas it implements, the training objective, the staged schedule, the
sampler, the evaluation metric groups, the screening rule, and the
instruments added during objective validation.

Every numeric constant is read LIVE from src/dr/config.py at build time, so
this document cannot drift from the code.  Formulas are transcribed from the
module that implements them and each section names its source file.

    python scripts/24_methods_reference.py

Writes  outputs/methods_reference.pdf
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
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

_spec = importlib.util.spec_from_file_location(
    "o1report", ROOT / "scripts" / "15_objective1_report.py")
_o1 = importlib.util.module_from_spec(_spec)
sys.modules["o1report"] = _o1
_spec.loader.exec_module(_o1)
Doc, INK, MUTED, RULE, GRADE_C = (_o1.Doc, _o1.INK, _o1.MUTED, _o1.RULE, _o1.GRADE_C)

REFERENCES = [
    "Cao, W., Mirjalili, V., Raschka, S. (2020). Rank consistent ordinal regression for "
    "neural networks with application to age estimation. Pattern Recognition Letters 140, "
    "325-331.   [CORAL, A6]",
    "Frank, E., Hall, M. (2001). A simple approach to ordinal classification. ECML 2001, "
    "LNCS 2167, 145-156.   [ordinal decomposition, A6]",
    "Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. "
    "ICLR 2022.   [A4 adapters]",
    "Houlsby, N. et al. (2019). Parameter-efficient transfer learning for NLP. ICML 2019, "
    "2790-2799.   [trained-parameter fraction]",
    "Zhou, Y. et al. (2023). A foundation model for generalizable disease detection from "
    "retinal images. Nature 622, 156-163.   [RETFound backbone, A4]",
    "He, K. et al. (2022). Masked autoencoders are scalable vision learners. CVPR 2022, "
    "16000-16009.   [MAE pretraining; probe-vs-finetune gap]",
    "Dosovitskiy, A. et al. (2021). An image is worth 16x16 words. ICLR 2021.   [ViT]",
    "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS 2017.   [cross-attention, A5]",
    "Velickovic, P. et al. (2018). Graph attention networks. ICLR 2018.   [GAT, A5]",
    "Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV 2017, 2980-2988. "
    "  [focal loss and prior bias init]",
    "Cui, Y. et al. (2019). Class-balanced loss based on effective number of samples. "
    "CVPR 2019, 9268-9277.   [class-balanced weighting, L_CBF]",
    "Khosla, P. et al. (2020). Supervised contrastive learning. NeurIPS 2020.   "
    "[L_contrastive, A6]",
    "Shrivastava, A., Gupta, A., Girshick, R. (2016). Training region-based object "
    "detectors with online hard example mining. CVPR 2016, 761-769.   [hard-example miner]",
    "Menon, A. K. et al. (2021). Long-tail learning via logit adjustment. ICLR 2021.   "
    "[decode-cut calibration]",
    "Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian approximation. ICML 2016, "
    "1050-1059.   [MC-dropout, A10]",
    "Kendall, A., Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning "
    "for computer vision? NeurIPS 2017.   [aleatoric/epistemic split, A10]",
    "Guo, C. et al. (2017). On calibration of modern neural networks. ICML 2017, 1321-1330. "
    "  [temperature scaling, ECE]",
    "Selvaraju, R. R. et al. (2017). Grad-CAM. ICCV 2017, 618-626.   [A9 attribution]",
    "Zhang, J. et al. (2018). Top-down neural attention by excitation backprop. IJCV 126, "
    "1084-1102.   [pointing game]",
    "Wang, H. et al. (2020). Score-CAM. CVPR Workshops 2020, 24-25.   [energy pointing game]",
    "Petsiuk, V., Das, A., Saenko, K. (2018). RISE. BMVC 2018.   [counterfactual faithfulness]",
    "Saito, T., Rehmsmeier, M. (2015). The precision-recall plot is more informative than "
    "the ROC plot. PLoS ONE 10(3), e0118432.   [AUPRC under sparsity]",
    "Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics "
    "Gems IV, 474-485.   [CLAHE, A2]",
    "Graham, B. (2015). Kaggle Diabetic Retinopathy Detection, 1st place solution.   "
    "[illumination normalisation, A2]",
    "Frangi, A. F. et al. (1998). Multiscale vessel enhancement filtering. MICCAI 1998, "
    "130-137.   [vessel/lesion separation, A2]",
    "Walter, T. et al. (2002). A contribution of image processing to the diagnosis of "
    "diabetic retinopathy. IEEE TMI 21(10), 1236-1243.   [top-hat exudate detection]",
    "Rose, A. (1948). The sensitivity performance of the human eye on an absolute scale. "
    "JOSA 38(2), 196-208.   [CNR detectability criterion]",
    "Cohen, J. (1968). Weighted kappa. Psychological Bulletin 70(4), 213-220.   [QWK]",
    "Wilkinson, C. P. et al. (2003). Proposed international clinical DR and DME severity "
    "scales. Ophthalmology 110(9), 1677-1682.   [grade definitions]",
    "Gulshan, V. et al. (2016). Development and validation of a deep learning algorithm "
    "for detection of DR. JAMA 316(22), 2402-2410.   [screening framing]",
    "Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the knowledge in a neural "
    "network. arXiv:1503.02531.   [deployment distillation]",
]


def fmt(v) -> str:
    if isinstance(v, float):
        return ("%.4g" % v)
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    if isinstance(v, dict):
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items())
    return str(v)


def build(path: Path) -> None:
    c = default_config()
    q, p, moe, bb = c.quality, c.preproc, c.moe, c.backbone
    fu, hd, sm, lo = c.fusion, c.head, c.sampling, c.loss
    xa = c.xai

    with PdfPages(path) as pdf:
        D = Doc(pdf)

        def formula(t, size=8.4):
            D.para(t, size=size, mono=True)

        def src(f):
            D.para("source: " + f, size=7.4, color=MUTED)

        # ============ 1 cover =========================================
        D.new("Methods and algorithms",
              "Technical reference · RETFound Plus–LAFT–XAI · %s" % date.today().isoformat())
        D.para("Every formula in this document is transcribed from the module that "
               "implements it, and every numeric constant is read live from "
               "src/dr/config.py at build time. Sections name their source file so any "
               "line can be checked against the code.", size=8.8)

        D.h2("The pipeline, end to end")
        formula(
            "raw fundus (any size)\n"
            "  |\n"
            "  +-- A1  quality gate      5 axes -> Q* -> retake / enhance / accept\n"
            "  +-- A2  field extraction  FOV crop -> %d x %d retinal field\n"
            "  v\n"
            "cached field  ---+--- global resize %d px\n"
            "                 +--- %d local crops, %d px cut at native res, fed at %d\n"
            "  |\n"
            "  +-- A2  ALPP      I* = sum_j w_j I_j ,  w = Softmax(G(I))\n"
            "  +-- A3  lesion MoE    alpha = Softmax(G(Z_G,Z_L,Q,H))\n"
            "  +-- A4  ViT backbone + GLA-LoRA adapters\n"
            "  +-- A5  bidirectional P<->A cross-attention -> kNN graph -> GAT\n"
            "  +-- A8  clinical fusion (gated; inert without metadata)\n"
            "  v\n"
            "  A6 ordinal CORAL   A7 temporal   A9 XAI chain   A10 uncertainty\n"
            "  v\n"
            "screening triage -> REFER / REVIEW / CLEAR"
            % (p.cache_size, p.cache_size, p.global_size, p.n_crops,
               p.crop_size, p.crop_input), size=7.6)

        D.h2("Notation")
        D.table(["symbol", "meaning"],
                [["I, I*", "input image; ALPP-fused image"],
                 ["Q*, Q_k", "pooled quality score; the five quality axes"],
                 ["Z_G, Z_L, Z_A", "global / lesion / anatomy token sets"],
                 ["Z_P, Z_F", "pathology stream; fused representation"],
                 ["z", "the shared CORAL rank (a scalar per image)"],
                 ["b_k", "CORAL thresholds, k = 0..K-2, strictly decreasing"],
                 ["y, yhat", "true and predicted grade in {0..4}"],
                 ["alpha_k", "MoE gate weight for expert k"],
                 ["r_l", "LoRA rank allocated to transformer block l"],
                 ["H_i", "hard-example score of sample i"],
                 ["lambda_XAI", "weight on the attribution-consistency loss"]],
                [0.155, 0.60], size=7.6)

        D.h2("Contents")
        D.para("p2   A1 quality gate            p8   A9 explainability\n"
               "p3   A2 preprocessing + ALPP    p9   A10 uncertainty + decision gate\n"
               "p4   A3 lesion mixture-of-experts  p10  the training objective\n"
               "p5   A4 backbone + GLA-LoRA     p11  sampling and hard-example mining\n"
               "p6   staged fine-tuning         p12  evaluation metric groups\n"
               "p7   A5 fusion + graph, A6 ordinal, A7/A8   p13  screening triage\n"
               "                                p14  instruments added in validation\n"
               "                                p15  live constants\n"
               "                                p16  script map\n"
               "                                p17  references", size=8.4)
        D.close()

        # ============ A1 ==============================================
        D.new("A1 — adaptive quality and domain gate", "Methods · src/dr/modules/a1_quality.py")
        D.para("Five independent axes are scored on the raw photograph and pooled. The "
               "lesion axis is the one a classical quality score misses: a sharp, "
               "well-exposed but hazy image scores high everywhere else and still hides "
               "every microaneurysm.", size=8.6)
        D.h2("Pooled score")
        formula("Q* = w1 Q_quality + w2 Q_domain + w3 Q_lesion + w4 Q_blur + w5 Q_illum")
        D.table(["axis", "weight", "what it measures"],
                [["Q_quality", fmt(q.weights["quality"]),
                  "gradability, artefact, FOV completeness"],
                 ["Q_domain", fmt(q.weights["domain"]),
                  "how typical the camera/colour fingerprint is"],
                 ["Q_lesion", fmt(q.weights["lesion"]),
                  "contrast-to-noise of lesion-scale round blobs"],
                 ["Q_blur", fmt(q.weights["blur"]), "variance of the Laplacian in-field"],
                 ["Q_illum", fmt(q.weights["illumination"]),
                  "exposure and illumination uniformity"]],
                [0.135, 0.09, 0.50], size=7.6)
        D.h2("Sub-weights inside Q_quality")
        formula("Q_quality = %s" % " + ".join(
            "%s*%s" % (fmt(v), k) for k, v in q.quality_sub.items()))
        D.h2("Lesion axis normalisation")
        formula("Q_lesion = clip( (CNR - %s) / (%s - %s), 0, 1 )\n"
                "Q_blur   = clip( var(Laplacian) / %s, 0, 1 )"
                % (fmt(q.lesion_floor_cnr), fmt(q.lesion_ref_cnr),
                   fmt(q.lesion_floor_cnr), fmt(q.blur_ref)))
        D.h2("Routing")
        formula("Q* <  %s              -> RETAKE\n"
                "%s <= Q* < %s   -> ENHANCE  (high-resolution enhancement path)\n"
                "Q* >= %s              -> ACCEPT"
                % (fmt(q.tau_r), fmt(q.tau_r), fmt(q.tau_d), fmt(q.tau_d)))
        D.para("The domain fingerprint (rg_ratio, fill, saturation against reference "
               "means and SDs) is also what the cross-domain evaluation uses to group "
               "test images.", size=8.4, color=MUTED)
        src("a1_quality.py, constants from QualityCfg")
        D.close()

        # ============ A2 ==============================================
        D.new("A2 — lesion-preserving preprocessing and ALPP",
              "Methods · src/dr/modules/a2_preprocess.py")
        D.h2("Classical chain (CPU, cached once by step 02)")
        formula("field extraction -> illumination normalisation -> adaptive CLAHE\n"
                "  -> vessel enhancement -> lesion enhancement -> quality-aware fusion")
        D.h2("Illumination normalisation (Graham 2015)")
        formula("I_norm = I - G_sigma(I) + 128 ,      sigma = %s px" % fmt(p.illum_sigma))
        D.h2("Adaptive CLAHE (Zuiderveld 1994)")
        formula("clip(Q) = clahe_clip * ( 1 + 0.8 (1 - Q) )       clahe_clip = %s\n"
                "tile grid = %d x %d,  applied to the L channel of LAB"
                % (fmt(p.clahe_clip), p.clahe_grid, p.clahe_grid))
        D.para("The clip limit RISES as quality falls: a clean image is left close to "
               "native because enhancement destroys microaneurysms, a poor image is "
               "pushed hard.", size=8.4, color=MUTED)
        D.h2("Lesion/vessel separation (Frangi 1998, Walter 2002)")
        formula("bright = white top-hat(green)                     hard exudates\n"
                "dark   = black top-hat(green)\n"
                "round_dark = dark - 0.85 * long_dark              elongated penalty")
        D.h2("Quality-aware fusion")
        formula("alpha  = gamma (1 - Q) ,   gamma = %s\n"
                "I_out  = (1 - alpha) I_native + alpha I_enhanced" % fmt(p.fusion_gamma))
        D.h2("ALPP — Adaptive Learned Preprocessing Policy (differentiable)")
        formula("I* = sum_j w_j I_j ,   j in %s\n"
                "[w_1..w_%d] = Softmax( G(I) )        per image, not per dataset\n"
                "G: 128x128 downsample -> conv gate -> %d hidden units\n"
                "prior initialisation  %s"
                % (fmt(p.alpp_branches), len(p.alpp_branches), p.alpp_gate_dim,
                   fmt(p.alpp_prior)))
        D.h2("Geometry")
        formula("cache %d px  ->  global view %d px\n"
                "            ->  %d local crops of %d px at NATIVE resolution, fed at %d px\n"
                "sampling density  global %.3f  vs  crop %.3f  output px per cache px"
                % (p.cache_size, p.global_size, p.n_crops, p.crop_size, p.crop_input,
                   p.crop_input / p.cache_size, p.crop_input / p.crop_size))
        D.para("Crop placement is not a grid: the macula and optic disc are always "
               "included, the remainder go to non-max-suppressed peaks of the lesion "
               "prior, with jitter %s." % fmt(p.crop_jitter), size=8.4, color=MUTED)
        src("a2_preprocess.py, lesion_priors.py; constants from PreprocCfg")
        D.close()

        # ============ A3 ==============================================
        D.new("A3 — lesion specialist mixture-of-experts",
              "Methods · src/dr/modules/a3_lesion_moe.py")
        D.para("Six specialist branches share a shallow stem; each emits a lesion "
               "evidence map and a pooled descriptor. Receptive fields differ per lesion "
               "class (dilated convolutions).", size=8.6)
        D.h2("Experts")
        formula("experts   %s\nexpert_dim %d   lesion_dim %d   mask_size %d"
                % (fmt(moe.experts), moe.expert_dim, moe.lesion_dim, moe.mask_size))
        D.h2("Adaptive gate")
        formula("alpha = Softmax( G(Z_G, Z_L, Q, H) / T ) ,   T = %s\n"
                "Z_MoE = sum_k alpha_k E_k(I)\n"
                "gate context %s ,  hidden %d"
                % (fmt(moe.gate_temperature), fmt(moe.gate_context), moe.gate_hidden))
        D.table(["context", "why it is in the gate"],
                [["Z_G", "what kind of retina this is, before choosing a specialist"],
                 ["Z_L", "pooled lesion-stem features"],
                 ["Q", "on low lesion-visibility images lean on large-scale experts"],
                 ["H", "images the model keeps getting wrong get an informed mixture"]],
                [0.115, 0.63], size=7.6)
        D.h2("Supervision")
        formula("L_lesion = Dice(E, M) + weighted BCE(E, M)\n"
                "channels a dataset does not annotate are MASKED OUT of the loss,\n"
                "not scored as all-negative")
        D.para("Pretrained on 838 images with pixel-level ophthalmologist masks "
               "(IDRiD 81 + DDR 757) covering MA / HE / EX_H / EX_S; NV and ME are "
               "unannotated and masked. The stem and experts then transfer into EyePACS "
               "grading.", size=8.4, color=MUTED)
        src("a3_lesion_moe.py, scripts/10_lesion_pretrain.py; constants from MoECfg")
        D.close()

        # ============ A4 ==============================================
        D.new("A4 — backbone and GLA-LoRA", "Methods · src/dr/modules/a4_backbone_lora.py")
        D.h2("Backbone")
        formula("%s      dynamic_img_size = True\n"
                "one tower serves the %d px global view and the %d px crops\n"
                "loader refuses to train below 95%% tensor coverage"
                % (bb.name, p.global_size, p.crop_input))
        D.h2("Low-rank adaptation (Hu et al. 2022)")
        formula("W_adapted = W_frozen + (alpha / r) B A\n"
                "  B in R^{d x r} , A in R^{r x k} , alpha = %s , dropout %s\n"
                "  targets %s"
                % (fmt(bb.lora_alpha), fmt(bb.lora_dropout), fmt(bb.lora_targets)))
        D.h2("GLA-LoRA — gradient / lesion / attention rank allocation")
        formula("S_l = %s G_l + %s L_l + %s A_l\n"
                "r_l = r_min + (r_max - r_min) S_l ,   S_l >= %s pinned to r_max\n"
                "r_min = %d ,  r_max = %d"
                % (fmt(bb.lam_grad), fmt(bb.lam_lesion), fmt(bb.lam_attn),
                   fmt(bb.lesion_layer_boost), bb.lora_rank_min, bb.lora_rank_max))
        D.table(["signal", "definition"],
                [["G_l", "||dL/dW|| inside block l on a calibration batch"],
                 ["L_l", "|corr( patch-token energy , lesion density )|"],
                 ["A_l", "share of block l's attention landing on lesion patches"]],
                [0.09, 0.66], size=7.6)
        D.para("All three signals are min-max normalised across blocks before pooling, "
               "so S_l is in [0,1] and the rank map is scale-free.", size=8.4, color=MUTED)
        D.h2("Measured allocation, flagship run")
        g = OUT_DIR / "retfound_plus_laft_xai" / "gla_lora.json"
        if g.exists():
            gj = json.load(open(g))
            formula("ranks  %s\nLoRA parameters  %s"
                    % (gj["ranks"], f"{gj['lora_params']:,}"), size=7.4)
        src("a4_backbone_lora.py; constants from BackboneCfg")
        D.close()

        # ============ staged schedule =================================
        D.new("Staged fine-tuning", "Methods · src/dr/config.py default_stages()")
        D.para("Four stages with three learning rates live at once. The backbone is "
               "released last and slowest.", size=8.6)
        rows = []
        for i, s in enumerate(c.train.stages, 1):
            rows.append([str(i), s.name, str(s.epochs),
                         "%.0f%%" % (100 * s.unfreeze_frac),
                         "on" if s.train_lora else "off",
                         "%.0e" % s.head_lr,
                         "%.0e" % s.lora_lr if s.lora_lr else "-",
                         "%.0e" % s.backbone_lr if s.backbone_lr else "-"])
        D.table(["#", "stage", "ep", "unfrozen", "LoRA", "head LR", "LoRA LR", "bb LR"],
                rows, [0.03, 0.135, 0.04, 0.085, 0.055, 0.085, 0.085, 0.08],
                mono_from=2, size=7.6)
        D.h2("ALPP scheduling")
        D.para("A2 ALPP sits in front of the backbone, so any gradient it needs traverses "
               "every transformer block. While the backbone is fully frozen and no LoRA is "
               "active that is a full-depth backward pass bought to update ~57k weights - "
               "measured on a ViT-Large it turns a 362 ms step into 3322 ms, a 9.2x "
               "penalty. ALPP is therefore off in stage 1 and on from stage 2, where the "
               "adapters already force the same backward.", size=8.5)
        D.h2("lambda_XAI over the schedule")
        formula("lambda_XAI(stage) = %s          ramp within stage: %s"
                % (fmt(xa.weight_schedule), xa.ramp_within_stage))
        D.para("Held at zero through stages 1-2: tying attribution to lesion masks while "
               "the attention map is still noise only teaches the model to match noise. "
               "CONSEQUENCE, measured during objective validation: a run that stops in "
               "stage 2 never applies L_XAI at all. --xai-weight overrides this.",
               size=8.5, color="#8a5a1b")
        src("config.py default_stages(), a9_xai.xai_weight_for_stage()")
        D.close()

        # ============ A5 / A6 / A7 / A8 ===============================
        D.new("A5 fusion, A6 ordinal head, A7 temporal, A8 clinical",
              "Methods · a5_fusion_gnn.py, a6_ordinal.py, a7_temporal.py, a8_multimodal.py")
        D.h2("A5 — bidirectional pathology/anatomy cross-attention")
        formula("Z_{P->A} = CA(Z_P, Z_A)        pathology attends to anatomy\n"
                "Z_{A->P} = CA(Z_A, Z_P)        anatomy attends to pathology\n"
                "Z_F      = Fusion(Z_P, Z_A, Z_{P->A}, Z_{A->P}, G)\n"
                "L_PA     = || P(Z_P) - P(Z_A) ||_2^2       shared probe, dim %d"
                % fu.probe_dim)
        formula("heads %d   regions %s\ngraph: kNN k=%d + spatial + lesion-in-region edges,"
                " %d GAT layers, fused_dim %d"
                % (fu.n_heads, fu.anatomy_regions, fu.knn, fu.gnn_layers, fu.fused_dim),
                size=7.6)

        D.h2("A6 — CORAL ordinal head")
        formula("b_0 = beta_0 ,  b_k = b_{k-1} - softplus(delta_k)      strictly decreasing\n"
                "P(Y > k)  = sigmoid( z + b_k )\n"
                "P(Y = 0)  = 1 - P(Y>0)\n"
                "P(Y = k)  = P(Y>k-1) - P(Y>k)        <= tanh( (b_{k-1}-b_k) / 4 )\n"
                "P(Y = K-1)= P(Y>K-2)\n"
                "yhat      = sum_k 1[ P(Y>k) > tau_k ] ,   tau_k = 0.5 canonical")
        D.para("The bound on the interior classes is why argmax over the differenced "
               "categorical cannot reach them when the threshold gaps are narrow. See "
               "page 14 for the calibrated tau_k.", size=8.4, color=MUTED)
        formula("hierarchy heads %s at thresholds %s\n"
                "L_boundary = |y - E[y]|^%s      E[y] = sum_k k P(Y=k)\n"
                "L_contrastive: supervised contrastive, temperature %s,\n"
                "               pair weight decreasing in ordinal distance |y_i - y_j|"
                % (fmt(hd.hierarchy), fmt(hd.hierarchy_thresholds),
                   fmt(lo.boundary_gamma), fmt(lo.contrastive_temp)), size=7.8)

        D.h2("A7 — temporal prognosis (inert on EyePACS)")
        formula("time-embedded causal transformer over visits I_1..I_T\n"
                "risk heads at %s months + discrete-time survival head\n"
                "L = L_BCE(per-horizon) + L_survival(hazard NLL)" % fmt(hd.horizons))
        D.para("EyePACS is cross-sectional, so this block is never trained and evaluation "
               "reports the prognosis group as NOT EVALUATED rather than inventing "
               "numbers.", size=8.4, color=MUTED)

        D.h2("A8 — gated clinical fusion")
        formula("Z_MC = Z_F + g * CrossAttn(Z_F, MLP(C)) ,   g in [0,1] learned\n"
                "valid = 0  =>  g := 0     hard zero, image-only degradation\n"
                "clinical_dim %d" % hd.clinical_dim)
        D.close()

        # ============ A9 ==============================================
        D.new("A9 — lesion-grounded explainability", "Methods · src/dr/modules/a9_xai.py")
        D.h2("Attribution operators")
        formula("Grad-CAM      alpha_k = mean_ij dy/dA^k_ij ,  L = ReLU( sum_k alpha_k A^k )\n"
                "rollout       product of layer attention matrices with residual mixing\n"
                "differentiable attribution: from the global<->local cross-attention map,\n"
                "               used inside the training loss (not Grad-CAM)")
        D.h2("Attribution-consistency loss")
        formula("L_XAI = 1 - Dice( attribution , lesion mask )\n"
                "L    += lambda_XAI(stage) * L_XAI")
        D.h2("Counterfactual")
        formula("x_cf    = inpaint(x, lesion mask)      blur radius %d\n"
                "delta_P = P(referable | x) - P(referable | x_cf)\n"
                "positive delta_P means the model is using the cited lesions"
                % xa.counterfactual_blur)
        D.h2("Grounded explanation chain")
        formula("prediction -> attention -> lesion -> anatomical region -> severity\n"
                "top-%d lesion classes cited, ranked by attention-lesion overlap\n"
                "chain confidence = f( overlap, causal delta, model confidence )"
                % xa.grounding_topk)
        D.h2("Localisation metrics")
        formula("Dice      = 2|A n B| / (|A| + |B|) ,  A = 1[sal >= 0.5]\n"
                "Dice_max  = 2 min(|A|,|B|) / (|A| + |B|)      ceiling at those areas\n"
                "AUROC     = ROC-AUC( B.ravel() , sal.ravel() )        chance 0.5\n"
                "AUPRC lift= average precision / |B|                   chance 1.0\n"
                "energy    = sum(sal * B) / sum(sal)\n"
                "conc      = energy / |B|                              chance 1.0\n"
                "pointing  = 1[ argmax(sal) in B ]")
        D.para("Dice_max and the four threshold-free metrics were added during objective "
               "validation; only Dice, IoU and the pointing game were being computed "
               "before.", size=8.4, color=MUTED)
        src("a9_xai.py, scripts/21_xai_evaluation.py")
        D.close()

        # ============ A10 =============================================
        D.new("A10 — uncertainty, calibration and the decision gate",
              "Methods · src/dr/modules/a10_uncertainty.py")
        D.h2("MC-dropout ensemble (Gal & Ghahramani 2016)")
        formula("p(y|x) = (1/T) sum_t p_t(y|x)      dropout active at inference")
        D.h2("Uncertainty decomposition (Kendall & Gal 2017)")
        formula("H_total     = - sum_k p_k log p_k                  predictive entropy\n"
                "H_aleatoric = (1/T) sum_t ( - sum_k p_tk log p_tk )\n"
                "H_epistemic = H_total - H_aleatoric                mutual information\n"
                "entropy reported normalised to [0,1] by log K")
        D.h2("Calibration (Guo et al. 2017)")
        formula("temperature   p = Softmax( logits / T ) ,   T fitted by NLL on VAL\n"
                "vector        p = Softmax( a * logits + b )\n"
                "isotonic      monotone piecewise-constant fit per class\n"
                "ECE = sum_m (|B_m|/n) | acc(B_m) - conf(B_m) |")
        D.para("Calibrators are fitted on validation logits only and applied unchanged to "
               "test, which is what keeps the reported ECE honest.", size=8.4, color=MUTED)
        D.h2("Clinical decision gate")
        formula("low uncertainty AND good quality   -> AI_DECISION\n"
                "high uncertainty                   -> HUMAN_REVIEW\n"
                "low quality (Q* < tau_r)           -> RETAKE_IMAGE")
        D.close()

        # ============ objective =======================================
        D.new("The training objective", "Methods · scripts/03_train.py compute_losses()")
        formula("L = %s L_ordinal        CORAL rank loss on the monotone cumulative head\n"
                "  + %s L_hierarchical   any-DR / referable / STDR / PDR, one head each\n"
                "  + %s L_boundary       |y - E[y]|^%s on the continuous expected grade\n"
                "  + %s L_contrastive    supervised contrastive, ordinal-distance weighted\n"
                "  + %s L_lesion         A3 expert supervision (Dice + weighted BCE)\n"
                "  + %s L_PA             ||P(Z_P) - P(Z_A)||^2 pathology/anatomy agreement\n"
                "  + lambda(t) L_XAI     1 - Dice(attribution, lesion mask)\n"
                "  + %s L_CBF            class-balanced focal, a moderate auxiliary"
                % (fmt(lo.w_ordinal), fmt(lo.w_hierarchical), fmt(lo.w_boundary),
                   fmt(lo.boundary_gamma), fmt(lo.w_contrastive), fmt(lo.w_lesion),
                   fmt(lo.w_pa), fmt(lo.w_cbf)), size=8.0)
        D.h2("Component definitions")
        formula("L_ordinal = - sum_k [ t_k log s_k + (1-t_k) log(1-s_k) ]\n"
                "            t_k = 1[y > k] ,  s_k = sigmoid(z + b_k)\n"
                "\n"
                "L_CBF     = - w_y (1 - p_y)^gamma log p_y        gamma = %s\n"
                "            w_c = (1 - beta) / (1 - beta^{n_c})  beta  = %s\n"
                "\n"
                "L_contrastive = - sum_i (1/|P(i)|) sum_{p in P(i)} v_ip\n"
                "                log exp(z_i.z_p / tau) / sum_a exp(z_i.z_a / tau)\n"
                "                tau = %s ,  v_ip decreasing in |y_i - y_p|"
                % (fmt(lo.focal_gamma), fmt(lo.cb_beta), fmt(lo.contrastive_temp)),
                size=7.8)
        D.h2("Model selection")
        formula("S = 0.5 QWK + 0.3 MacroF1 + 0.2 mean recall over grades %s\n"
                "never accuracy: predicting No DR for everything scores 0.73 on EyePACS"
                % fmt(c.train.minority_grades))
        D.close()

        # ============ sampling ========================================
        D.new("Sampling and hard-example mining", "Methods · src/dr/sampling.py")
        D.h2("Draw probability")
        formula("P_i  proportional to  n_{class(i)}^{power} * min( (1 + H_i)^eta , clip )\n"
                "                      * (1 + b1 * boundary_i) * (1 + b2 * lowconf_i)\n"
                "\n"
                "power = %s     n^-0.5, gentler than full inverse frequency\n"
                "eta   = %s     how hard the mining pulls\n"
                "clip  = %s     cap on the mining multiplier\n"
                "b1    = %s     extra weight for adjacent-class confusions\n"
                "b2    = %s     extra weight for low-margin predictions"
                % (fmt(sm.power), fmt(sm.hard_eta), fmt(sm.hard_clip),
                   fmt(sm.boundary_bonus), fmt(sm.lowconf_bonus)))
        D.h2("Hardness state, refreshed after every epoch")
        formula("H_i     <- a H_i + (1-a) L_i / (mean L + eps)     a = %s , eps = %s\n"
                "boundary_i <- a boundary_i + (1-a) 1[ |pred - y| = 1 ]\n"
                "lowconf_i  <- a lowconf_i  + (1-a) 1[ margin < 0.15 ]"
                % (fmt(sm.hard_ema), fmt(sm.hard_eps)))
        D.para("The sampler draws WITH replacement, so a hard sample legitimately appears "
               "several times per epoch. Duplicates are averaged with np.add.at rather "
               "than overwritten, which would keep only the last and noisiest observation.",
               size=8.4, color=MUTED)
        D.h2("Loss weighting on top")
        formula("w_c = n_c^{%s} , normalised to mean 1" % fmt(sm.class_weight_power))
        D.para("Deliberately moderate. Full inverse-frequency sampling stacked on full "
               "class-balanced weighting over-corrects until grade 4 dominates the "
               "gradient and the middle grades collapse.", size=8.4, color=MUTED)
        D.close()

        # ============ metrics =========================================
        D.new("Evaluation metric groups", "Methods · src/dr/metrics.py")
        D.para("38 parameters in 7 groups, as pure-numpy functions so any model's outputs "
               "can be scored.", size=8.6)
        D.table(["group", "n", "members"],
                [["1 diagnostic", "11", "accuracy, balanced accuracy, macro sens/spec/"
                  "prec/NPV/F1, AUROC-ovr, AUPRC, referable AUC, STDR AUC"],
                 ["2 ordinal", "3", "QWK, MAE grade, adjacent accuracy"],
                 ["3 calibration", "4", "ECE, MCE, Brier, NLL"],
                 ["4 lesion / XAI", "6", "attribution Dice, IoU, pointing game, "
                  "consistency, counterfactual delta-P, expert AUROC"],
                 ["5 prognosis", "4", "C-index, AUC@12m, AUC@24m, time-dependent Brier "
                  "(NOT EVALUATED on EyePACS)"],
                 ["6 robustness", "4", "mean corruption accuracy, relative robustness, "
                  "external-domain QWK, quality-stratified spread"],
                 ["7 deployment", "4", "model size, latency, throughput, distilled "
                  "accuracy retention"]],
                [0.135, 0.035, 0.60], size=7.2)
        D.h2("Key definitions")
        formula("QWK    = 1 - sum w_ij O_ij / sum w_ij E_ij ,   w_ij = (i-j)^2/(K-1)^2\n"
                "adjacent accuracy = mean 1[ |yhat - y| <= 1 ]\n"
                "ECE    = sum_m (|B_m|/n) | acc(B_m) - conf(B_m) |\n"
                "relative robustness = mean corruption accuracy / clean accuracy\n"
                "corruptions: gaussian noise, defocus blur, brightness, low contrast, jpeg")
        D.close()

        # ============ screening =======================================
        D.new("Screening triage", "Methods · src/dr/screening.py, scripts/09_screening.py")
        D.para("The grader is scored as a screening service, not a grade-namer: pick the "
               "threshold that clears the most patients while keeping the cleared bucket "
               "safe.", size=8.6)
        D.h2("Operating-point selection")
        formula("choose thr* = argmax_thr  cleared(thr)\n"
                "  subject to   NPV(thr) >= target        (or sensitivity >= target)\n"
                "\n"
                "NPV      = TN / (TN + FN)\n"
                "cleared  = (TN + FN) / n          workload reduction\n"
                "referral rate = 1 - cleared")
        D.h2("Protocol, and why it matters")
        D.para("Thresholds are selected on the VALIDATION split and applied unchanged to "
               "test and to external data. This is why a target of NPV >= 0.98 lands at "
               "0.9799 on test rather than exactly 0.98. Tuning the threshold on the data "
               "you report it on inflates the number and is the most common way a "
               "screening result becomes meaningless.", size=8.6)
        D.h2("Three-way triage")
        formula("score >= thr_refer                 -> REFER\n"
                "thr_review <= score < thr_refer    -> REVIEW\n"
                "score < thr_review                 -> CLEAR      thr_review = 0.6 thr_refer")
        D.close()

        # ============ validation instruments ==========================
        D.new("Instruments added during objective validation",
              "Methods · steps 15-23")
        D.h2("1. Decode reachability sweep  (Objective 1)")
        formula("sweep z over [-10, 10] on a 40,001-point grid\n"
                "apply each decode rule to sigmoid(z + b)\n"
                "record which grades are emitted anywhere on the grid")
        D.para("Uses no images. A grade absent from the output cannot be predicted for any "
               "input under any sampling scheme.", size=8.4, color=MUTED)
        D.h2("2. Prior-matched CORAL ladder  (Objective 1)")
        formula("b_k = logit P(Y > k)  on the training class frequencies\n"
                "     = log( P(Y>k) / (1 - P(Y>k)) )")
        D.h2("3. Post-hoc decode-cut calibration  (Objective 1)")
        formula("yhat = sum_k 1[ P(Y>k) > tau_k ]\n"
                "tau fitted on VAL by coordinate ascent on S, frozen, applied to test\n"
                "constraint: reject any tau for which some grade is never predicted\n"
                "equivalence:  b_k^eff = b_k - logit(tau_k)      (logit adjustment)")
        D.h2("4. Paired CNR with bootstrap  (Objective 2)")
        formula("CNR = |mean(lesion px) - mean(local background)| / std(local background)\n"
                "background = annular ring around each lesion\n"
                "effect = per-lesion paired ratio, bootstrapped 95% CI\n"
                "detectability endpoint = fraction with CNR < 1")
        D.h2("5. Matched adaptation  (Objective 3)")
        formula("two arms, identical everything, initialisation the only variable\n"
                "backbone frozen in both; GLA-LoRA + heads adapt\n"
                "reported as a paired delta per metric")
        D.h2("6. Threshold-free attribution scoring  (Objective 4)")
        formula("see page 8: Dice_max, AUROC, AUPRC lift, energy, concentration ratio")
        D.close()

        # ============ constants =======================================
        D.new("Live constants", "Methods · read from src/dr/config.py at build time")
        for title, obj in (("QualityCfg", q), ("PreprocCfg", p), ("MoECfg", moe),
                           ("BackboneCfg", bb), ("FusionCfg", fu), ("HeadCfg", hd),
                           ("SamplingCfg", sm), ("LossCfg", lo), ("XaiCfg", xa)):
            try:
                d = asdict(obj)
            except Exception:
                continue
            rows = [[k, fmt(v)[:70]] for k, v in d.items()
                    if not isinstance(v, (dict,)) or len(str(v)) < 90]
            # start a fresh page when this block would not fit below the cursor
            need = 0.036 + 0.0125 * (len(rows) + 1)
            if D.y - need < 0.075:
                D.close()
                D.new("Live constants (cont.)",
                      "Methods · read from src/dr/config.py at build time")
            D.h2(title, gap=0.012)
            D.table(["", ""], rows, [0.235, 0.60], size=6.6, row_h=0.0125)
        D.close()

        # ============ script map ======================================
        D.new("Script map", "Methods · what produces what")
        D.table(["script", "produces"],
                [["01_download_data.py", "raw EyePACS mirror"],
                 ["02_preprocess.py", "A1+A2 cache: images, lesions, anatomy, crops, meta"],
                 ["10_lesion_pretrain.py", "A3 expert weights from real masks"],
                 ["03_train.py", "staged training -> best.pt, history, gla_lora"],
                 ["04_evaluate.py", "38 parameters -> evaluation.json"],
                 ["05_predict.py", "single-image clinical report + XAI figure"],
                 ["06_deploy.py", "distilled student, ONNX, latency benchmark"],
                 ["07_compare_models.py", "ablation study"],
                 ["08_selftest.py", "13-block synthetic self-test"],
                 ["09_screening.py", "val-fitted operating points -> screening.json"],
                 ["11-12", "lesion download; lesion-visibility CNR"],
                 ["13_linear_probe.py", "frozen-feature probes"],
                 ["14_objective2_report.py", "Objective 2 evidence pack"],
                 ["15_objective1_report.py", "Objective 1 evidence pack"],
                 ["16/17", "decode-cut calibration + its report"],
                 ["18/19/20", "Objective 3 evidence, validation, consolidated"],
                 ["21/22", "threshold-free XAI scoring + Objective 4 pack"],
                 ["23", "four-objective consolidated report"],
                 ["24", "this document"]],
                [0.245, 0.57], size=7.2)
        D.h2("Drivers")
        formula("run_pipeline.sh     sequential end-to-end\n"
                "run_50pct.sh        training inside a 50% hardware envelope\n"
                "run_objective3.sh   matched adaptation arms\n"
                "run_objective4.sh   lambda_XAI A/B")
        D.close()

        # ============ references ======================================
        per = 11
        for start in range(0, len(REFERENCES), per):
            D.new("References" if start == 0 else "References (cont.)",
                  "Methods · sources for the formulas above")
            for r in REFERENCES[start:start + per]:
                D.para(r, size=7.8, lead=0.0135)
                D.para("", size=3)
            D.close()

        info = pdf.infodict()
        info["Title"] = "RETFound Plus-LAFT-XAI - methods and algorithms reference"


def main() -> None:
    out = OUT_DIR / "methods_reference.pdf"
    build(out)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
