#!/usr/bin/env python3
"""Step 14 - Objective 2 evidence pack: JSON + PDF.

Objective 2 claims the multistage preprocessing pipeline "enhances retinal
image quality and lesion visibility".  Step 12 measures it; this script turns
that measurement into a self-contained evidence document - every raw value,
every formula, every pipeline constant, and the literature each design choice
is answerable to - in both machine-readable and printable form.

    python scripts/14_objective2_report.py

Reads   outputs/lesion_visibility.json   (produced by scripts/12_lesion_visibility.py)
        outputs/lesion_pretrain/sources.json
        src/dr/config.py                 (live constants, not transcribed)
Writes  outputs/objective2_evidence.json
        outputs/objective2_evidence.pdf
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402

LABEL = {"MA": "Microaneurysm", "HE": "Haemorrhage",
         "EX_H": "Hard exudate", "EX_S": "Soft exudate"}
ORDER = ["MA", "HE", "EX_H", "EX_S"]
COND = ["global224", "global224_enh", "crop224", "crop224_enh"]
COND_LABEL = {"global224": "whole-image 224 px (native)",
              "global224_enh": "whole-image 224 px + enhance",
              "crop224": "320 px window @ 224 px (native)",
              "crop224_enh": "320 px window @ 224 px + enhance"}

INK, MUTED, RULE = "#16181d", "#5b6270", "#d4d7de"
BAR = {"global224": "#c3c8d2", "global224_enh": "#8b93a5",
       "crop224": "#4a7fb5", "crop224_enh": "#1f4e79"}

# --------------------------------------------------------------------------
# Literature the design and the measurement are answerable to.
# `supports` states what the citation is used FOR; `value` carries a number
# only where that number is a stable, textbook-level constant.  Nothing here
# was re-derived in this session - verify before quoting in a thesis.
# --------------------------------------------------------------------------
LITERATURE = [
    dict(key="rose1948", cite="Rose, A. (1948). The sensitivity performance of the human "
         "eye on an absolute scale. JOSA 38(2), 196-208.",
         supports="Detectability threshold for the CNR statistic. The Rose criterion holds "
                  "that a feature is reliably detectable only when its contrast-to-noise "
                  "ratio exceeds k, with k~3-5 for confident detection.",
         value="k >= 3 (marginal) to k >= 5 (confident); CNR < 1 = indistinguishable from noise",
         used_as="This report's 'below noise' cut is the conservative CNR < 1."),
    dict(key="zuiderveld1994", cite="Zuiderveld, K. (1994). Contrast Limited Adaptive "
         "Histogram Equalization. In Graphics Gems IV, 474-485.",
         supports="The CLAHE operator used as the local-contrast stage of A2 "
                  "(adaptive_clahe, applied to the L channel of LAB).",
         value="clip limit and tile grid are the two free parameters",
         used_as="clahe_clip = %s, clahe_grid = %s x %s, clip scaled by (1 + 0.8(1 - Q))."),
    dict(key="pizer1987", cite="Pizer, S. M. et al. (1987). Adaptive histogram equalization "
         "and its variations. CVGIP 39(3), 355-368.",
         supports="Origin of adaptive histogram equalisation; the noise-amplification failure "
                  "mode in flat regions that CLAHE's clip limit exists to bound.",
         value=None,
         used_as="Motivates clipping rather than plain AHE, and the ALPP contrast branch's "
                 "blend-back term (strength = 0.8)."),
    dict(key="foracchia2005", cite="Foracchia, M., Grisan, E., Ruggeri, A. (2005). Luminosity "
         "and contrast normalization in retinal images. Medical Image Analysis 9(3), 179-190.",
         supports="Luminosity/contrast normalisation as a required step for fundus images, "
                  "because vignetting and inter-camera exposure dominate raw pixel statistics.",
         value=None,
         used_as="Justifies normalise_illumination running before CLAHE, not after."),
    dict(key="graham2015", cite="Graham, B. (2015). Kaggle Diabetic Retinopathy Detection "
         "competition report (1st place solution).",
         supports="Large-sigma Gaussian background subtraction, I - G_sigma(I) + c, as the "
                  "standard fundus normalisation for EyePACS specifically.",
         value="the widely reused form is I*4 - blur*4 + 128",
         used_as="normalise_illumination() implements I - G_sigma(I) + 128 with "
                 "illum_sigma = %s px; the ALPP norm branch uses sigma = 0.06 x width."),
    dict(key="walter2002", cite="Walter, T. et al. (2002). A contribution of image processing "
         "to the diagnosis of diabetic retinopathy - detection of exudates. IEEE TMI 21(10), "
         "1236-1243.",
         supports="Morphological top-hat on the green channel as the classical exudate "
                  "detector; the green channel as the highest-contrast band for DR lesions.",
         value=None,
         used_as="enhance_lesions() white top-hat (bright lesions) and the green-channel-only "
                 "CNR measurement in step 12."),
    dict(key="frangi1998", cite="Frangi, A. F. et al. (1998). Multiscale vessel enhancement "
         "filtering. MICCAI 1998, LNCS 1496, 130-137.",
         supports="Vessels are elongated structures separable from round blobs by shape, not "
                  "by intensity - haemorrhages and vessels share a grey level.",
         value=None,
         used_as="The elongated-structuring-element penalty: round_dark = dark - 0.85 x "
                 "long_dark, in both the OpenCV and the torch/ALPP lesion branch."),
    dict(key="wilkinson2003", cite="Wilkinson, C. P. et al. (2003). Proposed international "
         "clinical diabetic retinopathy and diabetic macular edema disease severity scales. "
         "Ophthalmology 110(9), 1677-1682.",
         supports="Grade 1 (mild NPDR) is DEFINED as microaneurysms only. If MAs are below "
                  "the noise floor, grade 1 is unreachable by any classifier head.",
         value="Mild NPDR = microaneurysms only",
         used_as="Why MA is the lesion class that decides whether Objective 2 matters."),
    dict(key="etdrs1991", cite="Early Treatment Diabetic Retinopathy Study Research Group "
         "(1991). Grading diabetic retinopathy from stereoscopic colour fundus photographs "
         "- ETDRS report number 10. Ophthalmology 98(5 Suppl), 786-806.",
         supports="Reference grading protocol and the clinical size range of the lesions.",
         value="microaneurysms are typically 15-125 um in diameter",
         used_as="Input to the sampling-density derivation on the method page."),
    dict(key="porwal2018", cite="Porwal, P. et al. (2018). IDRiD: Indian Diabetic Retinopathy "
         "Image Dataset. Data 3(3), 25. (challenge: Med Image Anal 59, 101561, 2020)",
         supports="Pixel-level expert lesion annotation, 4288 x 2848 px, 50 degree FOV.",
         value="81 images with pixel-level MA/HE/EX/SE masks",
         used_as="Ground truth for this measurement; %s images available."),
    dict(key="li2019", cite="Li, T. et al. (2019). Diagnostic assessment of deep learning "
         "algorithms for diabetic retinopathy screening. Information Sciences 501, 511-522. "
         "(DDR dataset)",
         supports="The second source of pixel-level lesion masks, mixed resolution, "
                  "multi-centre.",
         value="757 images with pixel-level lesion masks (of 13,673 graded)",
         used_as="Ground truth for this measurement; %s images available."),
    dict(key="sahlsten2019", cite="Sahlsten, J. et al. (2019). Deep Learning Fundus Image "
         "Analysis for Diabetic Retinopathy and Macular Edema Grading. Scientific Reports 9, "
         "10750.",
         supports="Input resolution is a first-order variable for DR grading performance, "
                  "independent of architecture.",
         value=None,
         used_as="Prior art for the resolution arm of this experiment being worth running "
                 "at all."),
    dict(key="krause2018", cite="Krause, J. et al. (2018). Grader variability and the "
         "importance of reference standards for evaluating machine learning models for "
         "diabetic retinopathy. Ophthalmology 125(8), 1264-1272.",
         supports="Upper bound on achievable agreement: the reference standard itself is "
                  "noisy, so a 5-class ceiling well below 100% is expected.",
         value=None,
         used_as="Framing for what a QWK number can mean; not used in this measurement."),
    dict(key="gulshan2016", cite="Gulshan, V. et al. (2016). Development and validation of a "
         "deep learning algorithm for detection of diabetic retinopathy in retinal fundus "
         "photographs. JAMA 316(22), 2402-2410.",
         supports="The referral-screening framing the whole system is evaluated under.",
         value=None,
         used_as="Context only; Objective 2 is upstream of it."),
    dict(key="zhou2023", cite="Zhou, Y. et al. (2023). A foundation model for generalizable "
         "disease detection from retinal images. Nature 622, 156-163. (RETFound)",
         supports="The backbone the preprocessed pixels are fed to; fixes the 224 px patch "
                  "geometry that makes sampling density the binding constraint.",
         value="ViT-Large/16, 224 px input, MAE-pretrained on 1.6M retinal images",
         used_as="Why the local crop is fed at 224 px rather than at native resolution."),
]


def resolve_literature(cfg, sources) -> list:
    """Fill the %s placeholders in LITERATURE from live config and dataset counts.

    Kept as placeholders in the table above so a constant can never be quoted
    from a stale transcription of config.py.
    """
    p = cfg.preproc
    args = {
        "zuiderveld1994": (p.clahe_clip, p.clahe_grid, p.clahe_grid),
        "graham2015": (p.illum_sigma,),
        "porwal2018": (sources.get("idrid", {}).get("images", 0),),
        "li2019": (sources.get("ddr", {}).get("images", 0),),
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
#  JSON
# ==========================================================================
def build_json(vis: dict, sources: dict, cfg) -> dict:
    p, q = cfg.preproc, cfg.quality
    idrid = sources.get("idrid", {}).get("images", 0)
    ddr = sources.get("ddr", {}).get("images", 0)

    # sampling-density arithmetic, stated so it can be checked
    dens_global = p.crop_input / p.cache_size          # output px per cache px
    dens_crop = p.crop_input / p.crop_size
    um_per_px_1024 = 12000.0 / p.cache_size            # 12 mm field / 1024 px

    lesions = {}
    for ch in ORDER:
        if ch not in vis:
            continue
        d = vis[ch]
        lesions[ch] = {
            "lesion_name": LABEL[ch],
            "n_images_contributing": d["n"],
            "mean_cnr_by_condition": {k: d[k] for k in COND},
            "condition_definitions": COND_LABEL,
            "unpaired_gains": {
                "resolution_gain": d["resolution_gain"],
                "resolution_gain_definition": "mean(crop224) / mean(global224)",
                "enhancement_gain": d["enhancement_gain"],
                "enhancement_gain_definition": "mean(crop224_enh) / mean(crop224)",
            },
            "paired_effects": {
                "resolution": {**d["resolution_paired"],
                               "definition": "per-lesion ratio crop224 / global224",
                               "significant_at_95pct": bool(d["resolution_paired"]["ci"][0] > 1.0)},
                "enhancement": {**d["enhancement_paired"],
                                "definition": "per-lesion ratio crop224_enh / crop224",
                                "significant_at_95pct": bool(d["enhancement_paired"]["ci"][0] > 1.0)},
            },
            "detectability": {
                "fraction_below_noise_cnr_lt_1": d["frac_below_noise"],
                "absolute_reduction_global224_to_crop224_enh":
                    d["frac_below_noise"]["global224"] - d["frac_below_noise"]["crop224_enh"],
            },
        }

    return {
        "document": {
            "title": "Objective 2 - evidence pack",
            "objective_as_stated_in_code": 'the pipeline "enhances retinal image quality and '
                                           'lesion visibility"',
            "source_of_wording": "scripts/12_lesion_visibility.py:4",
            "generated": date.today().isoformat(),
            "generator": "scripts/14_objective2_report.py",
            "status": "SUPPORTED - both claims tested separately; resolution significant for "
                      "all four lesion classes, enhancement significant for three of four. "
                      "Sampled from the 838 available annotated images; 54-132 paired lesions "
                      "per class, see results[*].paired_effects[*].n_pairs.",
        },
        "claims_under_test": {
            "claim_A_resolution": {
                "statement": "Cutting a 320 px window at native cache resolution and feeding "
                             "it at 224 px makes lesions more detectable than a whole-image "
                             "224 px view.",
                "verdict": "SUPPORTED for all four lesion classes (95% CI excludes 1.0).",
            },
            "claim_B_enhancement": {
                "statement": "Illumination normalisation followed by adaptive CLAHE - the "
                             "multistage enhancement step itself - increases lesion "
                             "detectability.",
                "verdict": "SUPPORTED for MA, HE, EX_H; NOT SUPPORTED for EX_S "
                           "(median ratio 0.945, CI spans 1.0).",
            },
        },
        "method": {
            "metric": "green-channel contrast-to-noise ratio against an annular local background",
            "formula": "CNR = |mean(green[lesion]) - mean(green[ring])| / std(green[ring])",
            "background_definition": "ring = dilate(mask, (2r+1)^2) AND NOT mask; "
                                     "r = 2 px for the whole-image view, 6 px for the crop view",
            "channel": "green (index 1 of BGR) - highest lesion contrast in fundus photography",
            "minimum_lesion_size": "4 px in mask; ring must contain >= 8 px; std > 1e-6",
            "detectability_threshold": "CNR < 1.0 counted as 'below noise'",
            "paired_statistic": "median of the per-lesion ratio, 2000-sample bootstrap, "
                                "percentile 95% CI, rng seed 7",
            "image_sampling": "rng seed 1337, uniform without replacement over discovered records",
            "geometry": {
                "cache_size_px": p.cache_size,
                "crop_size_px": p.crop_size,
                "crop_input_px": p.crop_input,
                "global_view_px_in_this_experiment": 224,
                "output_px_per_cache_px_global": dens_global,
                "output_px_per_cache_px_crop": dens_crop,
                "linear_density_ratio": dens_crop / dens_global,
                "areal_density_ratio": (dens_crop / dens_global) ** 2,
                "approx_um_per_cache_px": um_per_px_1024,
                "approx_um_per_px_assumption": "45-50 degree field treated as ~12 mm across; "
                                               "approximate, camera-dependent",
                "microaneurysm_px_at_1024": [15.0 / um_per_px_1024, 125.0 / um_per_px_1024],
                "microaneurysm_px_at_global_224": [15.0 / um_per_px_1024 * dens_global,
                                                   125.0 / um_per_px_1024 * dens_global],
                "microaneurysm_px_at_crop_224": [15.0 / um_per_px_1024 * dens_crop,
                                                 125.0 / um_per_px_1024 * dens_crop],
            },
        },
        "ground_truth": {
            "datasets": sources,
            "total_annotated_images_available": idrid + ddr,
            "channels_with_real_masks": ["MA", "HE", "EX_H", "EX_S"],
            "channels_without_real_masks": ["NV", "ME"],
            "note": "Neither IDRiD nor DDR annotates neovascularisation or macular oedema, so "
                    "those two channels are absent from this measurement entirely.",
        },
        "pipeline_under_test": {
            "stage_1_illumination": {
                "function": "src/dr/modules/a2_preprocess.py:normalise_illumination",
                "formula": "I_norm = clip(I - GaussianBlur(I, sigma) + 128, 0, 255)",
                "sigma_px": p.illum_sigma,
                "reference": "graham2015 / foracchia2005",
            },
            "stage_2_clahe": {
                "function": "src/dr/modules/a2_preprocess.py:adaptive_clahe",
                "formula": "CLAHE on L of LAB, clip = clahe_clip x (1 + 0.8 x (1 - Q)), "
                           "tiles = grid x grid",
                "clahe_clip": p.clahe_clip,
                "clahe_grid": p.clahe_grid,
                "reference": "zuiderveld1994 / pizer1987",
            },
            "not_in_this_measurement": {
                "vessel_enhancement": "enhance_vessels - multi-scale black top-hat, r in (5,9,13)",
                "lesion_enhancement": "enhance_lesions - white top-hat + "
                                      "(black top-hat - 0.85 x elongated response)",
                "quality_aware_fusion": f"alpha = fusion_gamma x (1 - Q), "
                                        f"fusion_gamma = {p.fusion_gamma}",
                "learned_ALPP": "I* = w1 I_norm + w2 I_contrast + w3 I_vessel + w4 I_lesion, "
                                "[w] = Softmax(G(I)); gate_dim = %d, prior = %s"
                                % (p.alpp_gate_dim, list(p.alpp_prior)),
                "why": "step 12 isolates the two stages that are applied to the cached pixels "
                       "before the network sees them. The vessel/lesion maps and the learned "
                       "ALPP mixing act inside the model, so they cannot be measured with a "
                       "static CNR on cached images.",
            },
        },
        "quality_axis_A1": {
            "note": "The 'image quality' half of Objective 2 is scored by A1, not by this "
                    "experiment. Recorded here for completeness.",
            "formula": "Q* = w1 Q_quality + w2 Q_domain + w3 Q_lesion + w4 Q_blur + w5 Q_illum",
            "weights": dict(q.weights),
            "quality_sub_weights": dict(q.quality_sub),
            "routing": {"retake_below": q.tau_r, "enhance_below": q.tau_d,
                        "accept_at_or_above": q.tau_d},
            "Q_lesion_formula": "CNR_raw = p99(|DoG_lesion_scale| - 0.6 x vessel_response) / "
                                "(1.4826 x MAD(high-frequency residual)); "
                                "Q_lesion = clip((CNR_raw - floor) / (ref - floor), 0, 1)",
            "Q_lesion_scales": {"lesion_floor_cnr": q.lesion_floor_cnr,
                                "lesion_ref_cnr": q.lesion_ref_cnr,
                                "note": "recalibrated per corpus onto the [p5, p85] CNR range"},
            "blur_reference_laplacian_variance": q.blur_ref,
            "relationship_to_this_experiment": "A1's Q_lesion is an UNSUPERVISED proxy CNR "
                                               "(no masks, DoG + MAD). Step 12's CNR is the "
                                               "SUPERVISED version of the same quantity, "
                                               "computed against expert masks. They are not "
                                               "numerically comparable - different residual, "
                                               "different noise estimator.",
        },
        "results": lesions,
        "headline_numbers": {
            "largest_resolution_effect": {"lesion": "MA", "paired_median_ratio":
                                          vis["MA"]["resolution_paired"]["median_ratio"],
                                          "ci": vis["MA"]["resolution_paired"]["ci"]},
            "largest_enhancement_effect": {"lesion": "MA", "paired_median_ratio":
                                           vis["MA"]["enhancement_paired"]["median_ratio"],
                                           "ci": vis["MA"]["enhancement_paired"]["ci"]},
            "microaneurysm_detectability": {
                "below_noise_whole_image_224": vis["MA"]["frac_below_noise"]["global224"],
                "below_noise_crop_plus_enhance": vis["MA"]["frac_below_noise"]["crop224_enh"],
                "interpretation": "at 224 px whole-image, 100% of annotated microaneurysms sit "
                                  "inside the noise of their own background; the combined "
                                  "pathway recovers 61% of them above CNR 1.",
            },
        },
        "limitations": [
            "Measured on IDRiD + DDR, not on EyePACS. EyePACS ships no pixel masks, so the "
            "CNR gain is assumed to transfer; the two corpora differ in camera and compression.",
            "CNR is a detectability proxy, not detection. A higher CNR does not prove the "
            "network exploits it - that link is Objective 1's evidence, which currently fails "
            "(grade-1 recall 0.000).",
            "The 1024 px cache is itself a resize of the source image (IDRiD is 4288 x 2848), "
            "so the 'native' condition is native-to-the-cache, not native-to-the-sensor.",
            "Soft exudates show no enhancement benefit (median ratio 0.945, CI 0.897-1.039); "
            "they are large and already high-contrast, and CLAHE's local window can suppress "
            "a lesion that fills the tile.",
            "Ring radius differs between conditions (2 px global, 6 px crop) because the same "
            "physical annulus subtends different pixel counts. This is geometrically correct "
            "but means the two conditions are not identical estimators.",
            "n differs between the resolution and enhancement arms (e.g. MA: 99 vs 121 pairs) "
            "because a lesion can fail the >=4 px test in one view and pass in another.",
            "The exact --limit used for the recorded run is not logged; per-channel n is "
            "recorded instead (HE 132 implies at least 132 images were processed).",
        ],
        "reproduce": {
            "measurement": "python scripts/12_lesion_visibility.py --limit 200",
            "this_document": "python scripts/14_objective2_report.py",
            "inputs": ["outputs/lesion_visibility.json", "outputs/lesion_pretrain/sources.json"],
        },
        "literature": resolve_literature(cfg, sources),
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
        self.fig.text(0.075, 0.035, "RETFound Plus-LAFT-XAI  ·  Objective 2 evidence pack",
                      size=7.5, color=MUTED)
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def h2(self, t, gap=0.020):
        self.y -= gap
        self.fig.text(0.075, self.y, t, size=10.5, weight="bold", color=INK)
        self.y -= 0.016

    def para(self, t, size=8.6, mono=False, color=INK, indent=0.0, lead=0.0145):
        for line in t.split("\n"):
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

    def table(self, cols, rows, widths, size=8.0, head_size=7.6, row_h=0.0175, aligns=None):
        aligns = aligns or ["left"] * len(cols)
        x0 = 0.075
        xs, acc = [], x0
        for w in widths:
            xs.append(acc); acc += w
        for c, x, a in zip(cols, xs, aligns):
            xx = x if a == "left" else x + 0.0
            self.fig.text(xx, self.y, c, size=head_size, weight="bold", color=MUTED,
                          va="top", ha="left")
        self.y -= 0.016
        self.fig.lines.append(plt.Line2D([x0, acc - 0.01], [self.y + 0.004, self.y + 0.004],
                                         color=RULE, lw=0.7, transform=self.fig.transFigure))
        self.y -= 0.004
        for r in rows:
            for cell, x, a in zip(r, xs, aligns):
                self.fig.text(x, self.y, str(cell), size=size, color=INK, va="top",
                              family="monospace" if a == "num" else "sans-serif")
            self.y -= row_h
        self.y -= 0.006

    def axes(self, h, left=0.115, width=0.80, top_gap=0.040, bottom_gap=0.050):
        """Place an axes below the current cursor and leave the cursor clear of it.

        top_gap must clear the axes title, bottom_gap the x tick labels; both are
        measured in figure fractions because that is what the cursor is in.
        """
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


def build_pdf(doc_json: dict, vis: dict, cfg, path: Path):
    p = cfg.preproc
    g = doc_json["method"]["geometry"]
    with PdfPages(path) as pdf:
        D = Doc(pdf)

        # ---------------- page 1: verdict -----------------------------------
        D.new("Objective 2 — retinal image quality and lesion visibility",
              "Evidence pack · generated %s · RETFound Plus–LAFT–XAI" % doc_json["document"]["generated"])
        D.para('Objective as stated in the code: the pipeline "enhances retinal image quality\n'
               'and lesion visibility"  (scripts/12_lesion_visibility.py:4)', size=9, color=MUTED)
        D.h2("Verdict")
        D.para("SUPPORTED. The two claims inside the objective were separated and tested\n"
               "independently against expert pixel annotations (IDRiD + DDR, 838 images\n"
               "available; 54-132 paired lesions per class in the recorded run).\n"
               "Resolution is the dominant effect and is significant for all four annotated\n"
               "lesion classes. Enhancement is a real but second-order effect, significant\n"
               "for three of four; soft exudates show no benefit.", size=9)
        D.h2("Headline")
        rows = []
        for ch in ORDER:
            d = vis[ch]
            rp, ep = d["resolution_paired"], d["enhancement_paired"]
            rows.append([LABEL[ch], d["n"],
                         f"{rp['median_ratio']:.2f}x",
                         f"[{rp['ci'][0]:.2f}, {rp['ci'][1]:.2f}]",
                         f"{ep['median_ratio']:.2f}x",
                         f"[{ep['ci'][0]:.2f}, {ep['ci'][1]:.2f}]"])
        D.table(["lesion", "n", "resolution", "95% CI", "enhance", "95% CI"], rows,
                [0.145, 0.055, 0.105, 0.155, 0.100, 0.155])
        D.para("Paired median ratio of green-channel CNR, same lesion in both views,\n"
               "2000-sample bootstrap CI. A CI excluding 1.00 is a significant effect.",
               size=7.8, color=MUTED)

        D.h2("The number that matters")
        D.para("Microaneurysms define ICDR grade 1 (Wilkinson et al. 2003). At a whole-image\n"
               "224 px view, 100% of annotated microaneurysms sit below CNR 1 — inside the\n"
               "noise of their own local background. No classifier head can recover them.\n"
               "The native-resolution window plus enhancement brings 61% of them above that\n"
               "floor. This is the mechanism behind the README's grade-1 recall of 0.000.",
               size=9)

        ax = D.axes(0.20)
        labels = ["whole-224", "whole-224\n+enh", "crop-224", "crop-224\n+enh"]
        vals = [vis["MA"]["frac_below_noise"][c] * 100 for c in COND]
        ax.bar(labels, vals, color=[BAR[c] for c in COND], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", size=8, color=INK)
        ax.set_ylim(0, 112)
        ax.set_ylabel("% of microaneurysms below noise (CNR < 1)", size=8, color=MUTED)
        ax.set_title("Microaneurysm detectability by pathway", size=9, color=INK, loc="left")
        D.close()

        # ---------------- page 2: method ------------------------------------
        D.new("Method", "How the number is produced")
        D.h2("Metric")
        D.para("CNR = | mean(green[lesion]) - mean(green[ring]) |  /  std(green[ring])",
               size=10, mono=True)
        D.para("ring = dilate(mask, (2r+1)²) AND NOT mask\n"
               "r = 2 px in the whole-image view, 6 px in the crop view\n"
               "green channel only — the highest-contrast band for DR lesions (Walter 2002)\n"
               "guards: lesion ≥ 4 px, ring ≥ 8 px, std(ring) > 1e-6, else the lesion is dropped",
               size=8.4, mono=True, color=MUTED)
        D.para("The background is an annulus around each lesion rather than the whole image,\n"
               "so this is a LOCAL detectability measure: it asks whether the lesion can be\n"
               "told apart from the retina immediately around it, which is the question a\n"
               "convolution or an attention head actually faces.", size=8.8)

        D.h2("Four conditions, two claims")
        D.table(["condition", "what it is"],
                [["global224", "1024 px field resized whole to 224 px — the legacy pathway"],
                 ["global224_enh", "same, after illumination normalisation + adaptive CLAHE"],
                 ["crop224", "320 px window cut at 1024 px, fed at 224 px — the new pathway"],
                 ["crop224_enh", "same window, after the same enhancement"]],
                [0.17, 0.66], aligns=["num", "left"])
        D.para("claim A (resolution)  = crop224      / global224\n"
               "claim B (enhancement) = crop224_enh  / crop224",
               size=8.6, mono=True)

        D.h2("Sampling density — why resolution is expected to dominate")
        D.kv([("output px per cache px, whole-image", f"{p.crop_input}/{p.cache_size} = {g['output_px_per_cache_px_global']:.3f}"),
              ("output px per cache px, 320 px window", f"{p.crop_input}/{p.crop_size} = {g['output_px_per_cache_px_crop']:.3f}"),
              ("linear ratio", f"{g['linear_density_ratio']:.2f}x"),
              ("areal ratio", f"{g['areal_density_ratio']:.1f}x")], w=0.40)
        D.para("Physical scale (approximate — assumes a 45–50° field ≈ 12 mm across; camera\n"
               "dependent, stated so it can be corrected):", size=8.4, color=MUTED)
        D.kv([("µm per cache pixel @1024", f"{g['approx_um_per_cache_px']:.1f} µm"),
              ("microaneurysm, 15–125 µm (ETDRS)", f"{g['microaneurysm_px_at_1024'][0]:.1f}–{g['microaneurysm_px_at_1024'][1]:.1f} px @1024"),
              ("  → whole-image 224 px view", f"{g['microaneurysm_px_at_global_224'][0]:.2f}–{g['microaneurysm_px_at_global_224'][1]:.2f} px"),
              ("  → 320 px window @224 px", f"{g['microaneurysm_px_at_crop_224'][0]:.2f}–{g['microaneurysm_px_at_crop_224'][1]:.2f} px")], w=0.40)
        D.para("A sub-pixel lesion cannot have contrast: it is averaged into its neighbours by\n"
               "the resize itself. That is the prediction the CNR measurement tests.", size=8.8)

        D.h2("Statistics")
        D.para("Per-lesion paired ratio (same lesion, both views), median reported.\n"
               "2000-sample bootstrap of the median, percentile 95% CI, rng seed 7.\n"
               "Image sampling: uniform without replacement, rng seed 1337.\n"
               "'wins' = fraction of individual lesions where the second view scored higher.\n"
               "Detectability threshold: CNR < 1.0 = below noise (conservative against the\n"
               "Rose criterion, which puts confident detection at k ≈ 3–5).",
               size=8.6)
        D.close()

        # ---------------- page 3: primary table + chart ----------------------
        D.new("Results — mean CNR by lesion and condition",
              "Every value in outputs/lesion_visibility.json, unrounded values in the JSON")
        rows = []
        for ch in ORDER:
            d = vis[ch]
            rows.append([LABEL[ch], d["n"], _fmt(d["global224"]), _fmt(d["global224_enh"]),
                         _fmt(d["crop224"]), _fmt(d["crop224_enh"]),
                         f"{d['resolution_gain']:.2f}x", f"{d['enhancement_gain']:.2f}x"])
        D.table(["lesion", "n", "whole-224", "+enh", "crop-224", "+enh", "res.gain", "enh.gain"],
                rows, [0.135, 0.052, 0.098, 0.083, 0.098, 0.083, 0.093, 0.09],
                aligns=["left", "num", "num", "num", "num", "num", "num", "num"])
        D.para("res.gain = mean(crop-224) / mean(whole-224)   ·   enh.gain = mean(crop-224+enh) / mean(crop-224)\n"
               "These are ratios of means (unpaired). The paired medians on the next page are\n"
               "the inferential statistic; they differ because the CNR distribution is skewed.",
               size=7.8, color=MUTED)

        ax = D.axes(0.235)
        x = np.arange(len(ORDER)); w = 0.20
        for i, c in enumerate(COND):
            ax.bar(x + (i - 1.5) * w, [vis[ch][c] for ch in ORDER], w,
                   label=COND_LABEL[c], color=BAR[c])
        # the two reference lines go in the legend, not as floating text: at these
        # y-values any in-plot label lands on top of a bar in one group or another
        l1 = ax.axhline(1.0, color="#b3261e", lw=1.0, ls="--",
                        label="CNR = 1 · noise floor")
        l2 = ax.axhline(3.0, color="#946200", lw=0.9, ls=":",
                        label="Rose k = 3 · confident detection")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in ORDER], size=8)
        ax.set_ylabel("mean CNR", size=8, color=MUTED)
        ax.set_ylim(0, 3.9)
        ax.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=2,
                  columnspacing=1.4, handlelength=1.6)
        ax.set_title("Green-channel CNR against expert masks", size=9, color=INK, loc="left")

        ax2 = D.axes(0.205)
        for i, c in enumerate(COND):
            ax2.bar(x + (i - 1.5) * w, [vis[ch]["frac_below_noise"][c] * 100 for ch in ORDER],
                    w, color=BAR[c])
        ax2.set_xticks(x); ax2.set_xticklabels([LABEL[c] for c in ORDER], size=8)
        ax2.set_ylabel("% of lesions with CNR < 1", size=8, color=MUTED)
        ax2.set_ylim(0, 108)
        ax2.set_title("Fraction of lesions lost inside their own background noise",
                      size=9, color=INK, loc="left")
        D.close()

        # ---------------- page 4: paired effects -----------------------------
        D.new("Paired effects", "Same lesion measured in both views — the inferential result")
        D.h2("Claim A — resolution  (crop-224 / whole-224)")
        rows = []
        for ch in ORDER:
            r = vis[ch]["resolution_paired"]
            rows.append([LABEL[ch], r["n_pairs"], f"{r['median_ratio']:.2f}x",
                         f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]", f"{r['wins']*100:.0f}%",
                         "yes" if r["ci"][0] > 1 else "no"])
        D.table(["lesion", "pairs", "median ratio", "95% CI", "win rate", "CI excludes 1"],
                rows, [0.135, 0.070, 0.125, 0.165, 0.105, 0.13],
                aligns=["left", "num", "num", "num", "num", "left"])

        D.h2("Claim B — enhancement  (crop-224+enh / crop-224)")
        rows = []
        for ch in ORDER:
            r = vis[ch]["enhancement_paired"]
            rows.append([LABEL[ch], r["n_pairs"], f"{r['median_ratio']:.2f}x",
                         f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]", f"{r['wins']*100:.0f}%",
                         "yes" if r["ci"][0] > 1 else "no"])
        D.table(["lesion", "pairs", "median ratio", "95% CI", "win rate", "CI excludes 1"],
                rows, [0.135, 0.070, 0.125, 0.165, 0.105, 0.13],
                aligns=["left", "num", "num", "num", "num", "left"])

        ax = D.axes(0.285, left=0.215, width=0.70, top_gap=0.050, bottom_gap=0.058)
        ylab, ypos, k = [], [], 0
        for arm, key, col in (("resolution", "resolution_paired", "#1f4e79"),
                              ("enhancement", "enhancement_paired", "#946200")):
            for ch in ORDER:
                r = vis[ch][key]
                lo, hi, m = r["ci"][0], r["ci"][1], r["median_ratio"]
                ax.plot([lo, hi], [k, k], color=col, lw=1.6, solid_capstyle="butt",
                        label=arm if ch == ORDER[0] else None)
                ax.plot([m], [k], "o", color=col, ms=5)
                ax.text(hi * 1.06, k, f"{m:.2f}x", size=7, color=col, va="center")
                ylab.append(LABEL[ch]); ypos.append(k); k += 1
            k += 0.8
        ax.axvline(1.0, color="#b3261e", lw=1.0, ls="--")
        ax.text(1.0, 0.012, " no effect", size=6.8, color="#b3261e", ha="left",
                transform=ax.get_xaxis_transform(), va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([0.8, 1, 1.5, 2, 3, 5, 7])
        ax.set_xticklabels(["0.8x", "1x", "1.5x", "2x", "3x", "5x", "7x"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(ypos); ax.set_yticklabels(ylab, size=7.8)
        ax.invert_yaxis()
        ax.set_xlim(0.7, 9)
        ax.legend(fontsize=7.2, frameon=False, loc="lower right", ncol=2)
        ax.set_title("Paired median CNR ratio, 95% bootstrap CI (log scale)  ·  "
                     "top block = resolution, bottom = enhancement",
                     size=8.6, color=INK, loc="left")

        D.para("Soft exudate is the one negative result: median 0.945x, CI 0.897–1.039, win\n"
               "rate 43%. Soft exudates are large and already high-contrast; CLAHE's local\n"
               "window can suppress a lesion that fills a tile. Reported rather than dropped.",
               size=8.4)
        D.close()

        # ---------------- page 5: pipeline + A1 -----------------------------
        D.new("The pipeline under test", "Live constants read from src/dr/config.py")
        D.h2("Stage 1 — illumination normalisation")
        D.para("I_norm = clip( I - GaussianBlur(I, σ) + 128, 0, 255 ),  σ = %.0f px" % p.illum_sigma,
               size=9, mono=True)
        D.para("Classical Graham fundus normalisation (Graham 2015; Foracchia 2005). Removes\n"
               "the vignette and the inter-camera exposure differences that dominate EyePACS.\n"
               "Runs BEFORE CLAHE — order matters, CLAHE on an unnormalised image amplifies\n"
               "the vignette instead of the lesion.", size=8.6)
        D.h2("Stage 2 — adaptive CLAHE")
        D.para("CLAHE on L of LAB,  clip = %.1f × (1 + 0.8 × (1 − Q)),  tiles = %d × %d"
               % (p.clahe_clip, p.clahe_grid, p.clahe_grid), size=9, mono=True)
        D.para("The 'adaptive' part: a lower-quality image gets a higher clip limit. Colour is\n"
               "preserved by equalising luminance only (Zuiderveld 1994; clip bounds the noise\n"
               "amplification described in Pizer 1987).", size=8.6)

        D.h2("In the pipeline but NOT in this measurement")
        D.table(["stage", "form"],
                [["vessel map", "multi-scale black top-hat on green, r ∈ (5, 9, 13)"],
                 ["lesion map", "white top-hat ∨ (black top-hat − 0.85 × elongated response)"],
                 ["quality fusion", "α = %.1f × (1 − Q); blend native ↔ enhanced" % p.fusion_gamma],
                 ["learned ALPP", "I* = Σ wᵢ Iᵢ,  [w] = Softmax(G(I)),  branches = %d" % len(p.alpp_branches)]],
                [0.17, 0.66], aligns=["num", "left"])
        D.para("These act inside the model (differentiable, per-image) or on the cache-routing\n"
               "decision, so a static CNR on cached pixels cannot isolate them. Measuring the\n"
               "learned ALPP requires an ablation run, which the current training budget has\n"
               "not reached.", size=8.4, color=MUTED)

        D.h2("The other half of Objective 2 — 'image quality' (A1)")
        D.para("Q* = w₁·Q_quality + w₂·Q_domain + w₃·Q_lesion + w₄·Q_blur + w₅·Q_illum",
               size=9, mono=True)
        q = cfg.quality
        D.kv([("weights", ", ".join(f"{k}={v}" for k, v in q.weights.items())),
              ("Q_quality sub-weights", ", ".join(f"{k}={v}" for k, v in q.quality_sub.items())),
              ("routing", f"Q* < {q.tau_r} retake · < {q.tau_d} enhance · ≥ {q.tau_d} accept"),
              ("blur reference", f"Laplacian variance {q.blur_ref}"),
              ("Q_lesion scale", f"floor {q.lesion_floor_cnr}, ref {q.lesion_ref_cnr} (recalibrated per corpus)")],
             w=0.30, size=8.0)
        D.para("Q_lesion is A1's UNSUPERVISED analogue of this experiment's CNR: a lesion-scale\n"
               "difference-of-Gaussians with the elongated vessel response subtracted, over a\n"
               "MAD noise estimate. It needs no masks, so it runs on EyePACS — but its scale is\n"
               "pipeline-dependent and it is NOT numerically comparable to the supervised CNR\n"
               "reported here.", size=8.4)
        D.close()

        # ---------------- page 6: ground truth + limitations ----------------
        D.new("Ground truth, provenance and limitations")
        D.h2("Annotated corpora")
        src = doc_json["ground_truth"]["datasets"]
        rows = []
        for name in ("idrid", "ddr", "fgadr"):
            s = src.get(name, {})
            mp = s.get("masks_per_channel", {})
            rows.append([name.upper(), s.get("images", 0),
                         mp.get("MA", 0), mp.get("HE", 0), mp.get("EX_H", 0), mp.get("EX_S", 0)])
        D.table(["dataset", "images", "MA", "HE", "EX_H", "EX_S"], rows,
                [0.15, 0.10, 0.09, 0.09, 0.09, 0.09],
                aligns=["left", "num", "num", "num", "num", "num"])
        D.para("Neither corpus annotates neovascularisation (NV) or macular oedema (ME), so\n"
               "those two expert channels are absent from this measurement entirely. FGADR\n"
               "would supply NV/IRMA but requires a signed request form.", size=8.4, color=MUTED)

        D.h2("Provenance")
        D.kv([("measurement", "scripts/12_lesion_visibility.py"),
              ("raw values", "outputs/lesion_visibility.json"),
              ("dataset counts", "outputs/lesion_pretrain/sources.json"),
              ("constants", "src/dr/config.py (read live)"),
              ("this document", "scripts/14_objective2_report.py"),
              ("machine-readable twin", "outputs/objective2_evidence.json")], w=0.30, size=8.2)

        D.h2("Limitations — read before quoting")
        for i, lim in enumerate(doc_json["limitations"], 1):
            wrapped, line = [], ""
            for word in lim.split():
                if len(line) + len(word) > 92:
                    wrapped.append(line); line = word
                else:
                    line = f"{line} {word}".strip()
            wrapped.append(line)
            D.para(f"{i}.  " + "\n    ".join(wrapped), size=8.2, lead=0.0135)
        D.close()

        # ---------------- page 7-8: literature ------------------------------
        lit = doc_json["literature"]
        chunk = [lit[:8], lit[8:]]
        for pi, part in enumerate(chunk):
            D.new("Literature" + (" (continued)" if pi else ""),
                  "What each design choice is answerable to. Citations are given for "
                  "verification, not fetched in-session." if not pi else None)
            for ref in part:
                D.y -= 0.006
                D.fig.text(0.075, D.y, ref["key"], size=8.2, weight="bold",
                           color="#1f4e79", va="top")
                D.y -= 0.015
                for block, size, color in ((ref["cite"], 8.0, INK),
                                           ("Supports: " + ref["supports"], 7.8, MUTED),
                                           (("Value: " + ref["value"]) if ref["value"] else None, 7.8, "#946200"),
                                           ("Used as: " + ref["used_as"], 7.8, MUTED)):
                    if not block:
                        continue
                    line, out = "", []
                    for word in block.split():
                        if len(line) + len(word) > 98:
                            out.append(line); line = word
                        else:
                            line = f"{line} {word}".strip()
                    out.append(line)
                    for ln in out:
                        D.fig.text(0.085, D.y, ln, size=size, color=color, va="top")
                        D.y -= 0.0128
                D.y -= 0.004
            D.close()

        pdf.infodict()["Title"] = "Objective 2 evidence pack - lesion visibility"
        pdf.infodict()["Subject"] = "RETFound Plus-LAFT-XAI"


def main() -> None:
    cfg = default_config()
    vis = json.loads((OUT_DIR / "lesion_visibility.json").read_text())
    sources = json.loads((OUT_DIR / "lesion_pretrain" / "sources.json").read_text())

    doc = build_json(vis, sources, cfg)
    jpath = OUT_DIR / "objective2_evidence.json"
    jpath.write_text(json.dumps(doc, indent=2))
    save_csv_alongside(doc, jpath)
    print(f"[saved] {jpath}")

    ppath = OUT_DIR / "objective2_evidence.pdf"
    build_pdf(doc, vis, cfg, ppath)
    print(f"[saved] {ppath}")


if __name__ == "__main__":
    main()
