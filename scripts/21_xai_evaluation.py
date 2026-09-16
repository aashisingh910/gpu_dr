#!/usr/bin/env python3
"""Step 21 - Objective 4: attribution faithfulness, measured properly.

The XAI numbers in evaluation.json are scored as Dice at a fixed 0.5 threshold
between a min-max-normalised Grad-CAM and the union of the cached lesion
priors.  That comparison is mis-specified, and the reason is arithmetic rather
than empirical.

Dice = 2|A n B| / (|A| + |B|).  The lesion union B covers a median of well
under one percent of the pixels; a normalised saliency map A covers tens of
percent.  Even a PERFECT explanation - one whose above-threshold region
contains every annotated lesion pixel - is then capped at

    Dice_max = 2 min(|A|,|B|) / (|A| + |B|)

which for |A| = 0.30 and |B| = 0.009 is 0.059.  A reported Dice of 0.037 is
not "the explanation is worthless"; it is 63% of everything the metric can
express.  The metric is measuring the sparsity mismatch, not the explanation.

This script therefore reports, per image:

  * Dice at 0.5                    the existing number, reproduced exactly
  * Dice_max                       the analytic ceiling given the two areas
  * normalised Dice                Dice / Dice_max, in [0,1]
  * attribution AUROC              threshold-free: saliency as a per-pixel score
  * attribution AUPRC and lift     AUPRC / prevalence, the sparse-target metric
  * energy pointing game           saliency mass inside the mask (Wang 2020)
  * concentration ratio            energy / |B|, i.e. mass relative to chance
  * pointing game                  is the saliency argmax inside a lesion

The threshold-free metrics are the standard ones for saliency evaluation
against sparse targets and none of them were being computed.

    python scripts/21_xai_evaluation.py --ckpt outputs/vits_objectives/best.pt

Writes  outputs/xai_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, RAW_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS, batch_from_rows  # noqa: E402
from dr.data.lesion_datasets import discover_all  # noqa: E402
from dr.data.lesion_priors import LesionPriorExtractor  # noqa: E402
from dr.model import build_model  # noqa: E402
from dr.modules import a1_quality, a2_preprocess  # noqa: E402
from dr.modules.a9_xai import GradCAM, counterfactual_image, pointing_game  # noqa: E402
from dr.modules.external_lesion_batches import (  # noqa: E402
    LESION_NAMES, build_idrid_ddr_sample, build_input_from_preprocessed)


# ---------------------------------------------------------------------------
# Objective 4 fix: score attribution against real ophthalmologist masks
# ---------------------------------------------------------------------------
# The original evaluation scores Grad-CAM against `ds.lesions` - the
# morphological lesion PRIORS cached at preprocessing time, not human
# annotation (EyePACS carries no pixel labels at all). Objective 2 already
# established the stronger reference: 838 IDRiD + DDR images with real
# ophthalmologist masks. Reusing that reference here removes the "maybe it's
# a weak reference, not a weak explanation" caveat entirely - the one thing
# the evidence pack names as the single change that would make this evidence
# hard to attack. `build_idrid_ddr_sample` (shared with the training-time
# auxiliary loss in scripts/03_train.py) returns per-channel masks + a
# validity vector; this script only needs the union across annotated
# channels for the localisation metrics below.

def build_idrid_ddr_union_sample(rec, cfg, device, mask_size: int):
    sample = build_idrid_ddr_sample(rec, cfg, device, mask_size)
    if sample is None:
        return None
    batch, target_masks, valid = sample
    annotated = valid.bool()
    if not bool(annotated.any()):
        return None
    union = target_masks[0][annotated.cpu()].amax(0).cpu().numpy()
    return batch, union


def to_device(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def model_inputs(b: dict) -> dict:
    return {k: b[k] for k in ("image", "crops", "anatomy", "quality_axes",
                              "hardness") if k in b}


# --------------------------------------------------------------------------
def dice_at(sal: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    a = (sal >= thr).astype(np.float32)
    b = (mask >= 0.5).astype(np.float32)
    s = a.sum() + b.sum()
    return float(2.0 * (a * b).sum() / s) if s > 0 else float("nan")


def dice_ceiling(area_a: float, area_b: float) -> float:
    """Largest Dice attainable at these two areas, i.e. maximal overlap."""
    s = area_a + area_b
    return float(2.0 * min(area_a, area_b) / s) if s > 0 else float("nan")


def energy_pointing(sal: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total saliency mass falling inside the mask (Wang et al. 2020)."""
    m = (mask >= 0.5).astype(np.float32)
    tot = float(sal.sum())
    return float((sal * m).sum() / tot) if tot > 0 else float("nan")


def per_image_metrics(sal: np.ndarray, union: np.ndarray) -> dict | None:
    m = (union >= 0.5)
    if m.sum() == 0 or m.all():
        return None
    y = m.ravel().astype(np.int8)
    s = sal.ravel().astype(np.float64)
    area_b = float(m.mean())
    area_a = float((sal >= 0.5).mean())
    d = dice_at(sal, union)
    dmax = dice_ceiling(area_a, area_b)
    e = energy_pointing(sal, union)
    ap = float(average_precision_score(y, s))
    return {
        "area_saliency_ge_0.5": area_a,
        "area_lesion_union": area_b,
        "dice_at_0.5": d,
        "dice_ceiling": dmax,
        "dice_normalised": float(d / dmax) if dmax and dmax > 0 else float("nan"),
        "attribution_auroc": float(roc_auc_score(y, s)),
        "attribution_auprc": ap,
        "auprc_prevalence_baseline": area_b,
        "auprc_lift": float(ap / area_b) if area_b > 0 else float("nan"),
        "energy_pointing_game": e,
        "concentration_ratio": float(e / area_b) if area_b > 0 else float("nan"),
        "pointing_game": float(pointing_game(sal, union)),
    }


def summarise(rows: list) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[k] = {"mean": float("nan"), "median": float("nan"), "n": 0}
            continue
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "p25": float(np.quantile(v, .25)), "p75": float(np.quantile(v, .75)),
                  "n": int(v.size)}
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=OUT_DIR / "vits_objectives" / "best.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so this can run beside a training job")
    ap.add_argument("--counterfactual", action="store_true",
                    help="also run the lesion-erasure counterfactual (2x slower)")
    ap.add_argument("--ref-masks", choices=("eyepacs", "idrid_ddr"), default="eyepacs",
                    help="'eyepacs' (default) scores against the cached "
                         "morphological lesion PRIORS, the weak reference the "
                         "evidence pack flags as a caveat. 'idrid_ddr' scores "
                         "against the 838 real ophthalmologist-annotated "
                         "images already used for Objective 2 - removes that "
                         "caveat entirely (Objective 4 fix).")
    ap.add_argument("--lesion-root", type=Path, default=RAW_DIR / "lesion",
                    help="root of the downloaded IDRiD/DDR annotations, used "
                         "only when --ref-masks idrid_ddr")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "xai_evaluation.json")
    args = ap.parse_args()

    cfg = default_config()
    cfg.device = args.device
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    for field in ("backbone", "fusion", "moe", "preproc", "head"):
        if isinstance(saved, dict) and field in saved:
            setattr(cfg, field, saved[field])

    model = build_model(cfg)
    ranks = (ck.get("gla_lora") or ck.get("allora"))["ranks"]
    from dr.modules.a4_backbone_lora import inject_lora
    inject_lora(model.backbone.vit, ranks, cfg.backbone)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"[setup] device={device}  ckpt epoch={ck.get('epoch')} stage={ck.get('stage')}")

    cam = GradCAM(model)
    rows, cf_deltas = [], []
    t0 = time.time()

    if args.ref_masks == "idrid_ddr":
        records, found = discover_all(args.lesion_root)
        if not records:
            raise SystemExit(f"no annotated records under {args.lesion_root}; "
                             f"run scripts/11_download_lesions.py first")
        rng = np.random.default_rng(1337)
        if args.n and args.n < len(records):
            records = [records[i] for i in
                      rng.choice(len(records), args.n, replace=False)]
        print(f"[data] idrid_ddr: scoring attribution on {len(records)} "
              f"ophthalmologist-annotated images from {list(found)}")

        for i, rec in enumerate(records):
            sample = build_idrid_ddr_union_sample(rec, cfg, device, cfg.moe.mask_size)
            if sample is None:
                continue
            b1, union = sample
            if union is None or union.max() < 0.5:
                continue
            sal = cam(model_inputs(b1), out_size=union.shape[-1])[0, 0].detach().cpu().numpy()
            mm = per_image_metrics(sal, union)
            if mm:
                rows.append(mm)

            if args.counterfactual:
                with torch.no_grad():
                    p0 = float(model(model_inputs(b1))["referable"])
                bgr = cv2.imread(str(rec.image), cv2.IMREAD_COLOR)
                cf = cv2.cvtColor(counterfactual_image(bgr, union), cv2.COLOR_BGR2RGB)
                # re-run A1/A2 on the counterfactual pixels so the model sees
                # a properly field-extracted image, exactly as for the original
                qrep = a1_quality.assess(cv2.cvtColor(cf, cv2.COLOR_RGB2BGR), cfg.quality)
                pre_cf = a2_preprocess.run(cv2.cvtColor(cf, cv2.COLOR_RGB2BGR),
                                          qrep, cfg.preproc)
                priors_cf = LesionPriorExtractor(out_size=cfg.moe.mask_size)(
                    pre_cf.image, pre_cf.fov_mask)
                b2 = build_input_from_preprocessed(pre_cf, priors_cf, cfg, device)
                with torch.no_grad():
                    p1 = float(model(model_inputs(b2))["referable"])
                cf_deltas.append(p0 - p1)

            if i % 20 == 0:
                print(f"    {i+1}/{len(records)}  "
                      f"{(i+1)/max(time.time()-t0,1e-6):.2f} img/s", flush=True)
    else:
        ds = CachedEyePACS(args.cache, args.split, augment=False, cfg=cfg)
        sub = list(ds.idx[:args.n])
        print(f"[data] {args.split}: scoring attribution on {len(sub)} images "
              f"(reference: cached morphological lesion priors, NOT "
              f"ophthalmologist annotation - see --ref-masks idrid_ddr)")

        for i, j in enumerate(sub):
            r = ds.row_of(j)
            les = np.asarray(ds.lesions[r]).astype(np.float32) / 255.0
            union = les.max(0)
            b1 = batch_from_rows(ds, [j], device)
            sal = cam(model_inputs(b1), out_size=union.shape[-1])[0, 0].detach().cpu().numpy()
            mm = per_image_metrics(sal, union)
            if mm:
                rows.append(mm)

            if args.counterfactual and union.max() >= 0.5:
                with torch.no_grad():
                    p0 = float(model(model_inputs(b1))["referable"])
                bgr = cv2.cvtColor(np.asarray(ds.images[r]), cv2.COLOR_RGB2BGR)
                cf = cv2.cvtColor(counterfactual_image(bgr, union), cv2.COLOR_BGR2RGB)
                b2 = batch_from_rows(ds, [j], device, pixel_override=cf[None])
                with torch.no_grad():
                    p1 = float(model(model_inputs(b2))["referable"])
                cf_deltas.append(p0 - p1)

            if i % 20 == 0:
                print(f"    {i+1}/{len(sub)}  {(i+1)/max(time.time()-t0,1e-6):.2f} img/s",
                      flush=True)
    cam.remove()

    if not rows:
        raise SystemExit("no image had a usable lesion mask")
    summ = summarise(rows)

    doc = {
        "document": {
            "title": "Objective 4 - attribution faithfulness, threshold-free",
            "generated": time.strftime("%Y-%m-%d"),
            "generator": "scripts/21_xai_evaluation.py",
            "checkpoint": str(args.ckpt),
            "split": args.split if args.ref_masks == "eyepacs" else "idrid_ddr (external)",
            "ref_masks": args.ref_masks,
            "n_images_scored": len(rows),
            "device": str(device),
        },
        "why": {
            "problem": "Dice at a fixed threshold between a dense saliency map and a "
                       "sparse lesion mask is bounded by the sparsity mismatch, not by "
                       "explanation quality.",
            "bound": "Dice_max = 2*min(|A|,|B|) / (|A| + |B|)",
            "measured_areas": {
                "saliency_ge_0.5_median": summ["area_saliency_ge_0.5"]["median"],
                "lesion_union_median": summ["area_lesion_union"]["median"],
            },
        },
        "reference_mask": (
            {
                "what": "union over the six cached lesion-prior channels "
                        "(data/cache/lesions.npy), thresholded at 0.5",
                "channels": LESION_NAMES,
                "caveat": "these are the MORPHOLOGICAL priors from lesion_priors.py, "
                          "not ophthalmologist annotations. EyePACS carries no pixel "
                          "labels, so any attribution score computed on it is scored "
                          "against a weak reference. Re-run with --ref-masks idrid_ddr "
                          "to score against the 838 real ophthalmologist-annotated "
                          "images already used for Objective 2.",
            } if args.ref_masks == "eyepacs" else {
                "what": "union over per-image ophthalmologist annotation masks "
                        "(IDRiD + DDR), aligned into the same field-extraction "
                        "geometry as the model input",
                "source": str(args.lesion_root),
                "caveat": "Objective 4 fix: this is the same 838-image real-mask "
                          "reference already used for Objective 2, replacing the "
                          "EyePACS morphological-prior reference and removing the "
                          "weak-reference caveat noted there.",
            }
        ),
        "metrics": summ,
        "counterfactual": {
            "n": len(cf_deltas),
            "mean_delta_referable": float(np.mean(cf_deltas)) if cf_deltas else None,
            "median_delta_referable": float(np.median(cf_deltas)) if cf_deltas else None,
            "definition": "P(referable | original) - P(referable | lesions inpainted). "
                          "Positive means erasing the cited lesions lowers predicted "
                          "severity, i.e. the model is using them.",
        } if args.counterfactual else None,
        "per_image": rows,
    }
    json.dump(doc, open(args.out, "w"), indent=2)
    save_csv_alongside(rows, args.out,
                       csv_path=Path(args.out).with_name(Path(args.out).stem + "_per_image.csv"))
    save_csv_alongside(summ, args.out,
                       csv_path=Path(args.out).with_name(Path(args.out).stem + "_metrics.csv"))
    print(f"\n[write] {args.out}")

    print("\n%-28s %8s %8s" % ("metric", "mean", "median"))
    for k in ("area_saliency_ge_0.5", "area_lesion_union", "dice_at_0.5",
              "dice_ceiling", "dice_normalised", "attribution_auroc",
              "attribution_auprc", "auprc_lift", "energy_pointing_game",
              "concentration_ratio", "pointing_game"):
        print("%-28s %8.4f %8.4f" % (k, summ[k]["mean"], summ[k]["median"]))
    if cf_deltas:
        print("%-28s %8.4f %8.4f" % ("counterfactual delta P",
                                     float(np.mean(cf_deltas)),
                                     float(np.median(cf_deltas))))


if __name__ == "__main__":
    main()
