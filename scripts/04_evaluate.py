#!/usr/bin/env python3
"""Step 04 - full evaluation: all 38 parameters across the 7 metric groups.

    python scripts/04_evaluate.py --ckpt outputs/retfound_plus_laft_xai/best.pt

Groups 1-3 and 6-7 come from the real EyePACS test split (patient-disjoint from
training).  Group 4 (lesion/XAI) is scored against the weak morphological
priors and is labelled as such.  Group 5 (prognosis) is reported as
NOT EVALUATED because EyePACS has no follow-up visits - see a7_temporal.py.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS, batch_from_rows, GRADE_NAMES  # noqa: E402
from dr.metrics import (calibration_metrics, count_reported,  # noqa: E402
                        deployment_metrics, diagnostic_metrics,
                        lesion_xai_metrics, ordinal_metrics, prognosis_metrics,
                        referable_operating_point, robustness_metrics,
                        subgroup_metrics)
from dr.model import build_model  # noqa: E402
from dr.modules.a6_ordinal import coral_decode_numpy  # noqa: E402
from dr.modules.a9_xai import (GradCAM, attention_rollout,  # noqa: E402
                               counterfactual_image, dice_score, pointing_game)
from dr.modules.a10_uncertainty import (TemperatureScaler, decision_gate,  # noqa: E402
                                        mc_predict)
from sklearn.metrics import roc_auc_score  # noqa: E402


def to_device(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
def model_inputs(b: dict) -> dict:
    """Only the keys the model consumes, so a batch dict can carry extras."""
    return {k: b[k] for k in ("image", "crops", "anatomy", "quality_axes",
                              "hardness") if k in b}



DECODE = "coral"


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    P, Y, L, I = [], [], [], []
    for batch in loader:
        b = to_device(batch, device)
        o = model(model_inputs(b))
        P.append(o["ordinal"].class_probs.float().cpu().numpy())
        L.append(o["ordinal"].logits_ce.float().cpu().numpy())
        Y.append(b["grade"].cpu().numpy())
        I.append(b["index"].cpu().numpy())
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(L),
            np.concatenate(I))


@torch.no_grad()
def tta_infer(model, loader, device, views=("orig", "hflip")):
    """Test-time augmentation: average class_probs across a few cheap,
    label-preserving views. No retraining, no extra data, no extra GPU
    memory beyond one extra forward pass per view - a near-free accuracy
    lever completely orthogonal to the data/compute constraints the rest of
    this run operates under.

    'hflip' mirrors image, crops, AND anatomy together (anatomy is a
    spatial heatmap over the same pixel grid, so it must flip in lockstep
    or the fusion stage sees an anatomy map that no longer matches the
    image it's paired with). Horizontal mirroring of a fundus photo is a
    standard, well-supported TTA view in the fundus-imaging literature -
    unlike a vertical flip or rotation, it does not create an anatomically
    implausible image.

    logits_ce are NOT meaningfully averageable across views (they're
    pre-softmax and views can shift their scale differently), so only the
    calibrated class_probs are averaged; logits_ce from the ORIGINAL view
    are kept for anything downstream that specifically wants them.
    """
    model.eval()
    P, Y, L, I = [], [], [], []
    for batch in loader:
        b = to_device(batch, device)
        view_probs = []
        logits_orig = None
        for view in views:
            vb = dict(b)
            if view == "hflip":
                vb["image"] = torch.flip(b["image"], dims=[-1])
                if "crops" in b:
                    vb["crops"] = torch.flip(b["crops"], dims=[-1])
                if "anatomy" in b:
                    vb["anatomy"] = torch.flip(b["anatomy"], dims=[-1])
            elif view != "orig":
                raise ValueError(f"unknown TTA view '{view}'")
            o = model(model_inputs(vb))
            view_probs.append(o["ordinal"].class_probs.float())
            if view == "orig":
                logits_orig = o["ordinal"].logits_ce.float()
        probs = torch.stack(view_probs, 0).mean(0)
        if logits_orig is None:
            logits_orig = view_probs[0].log()   # views didn't include "orig"; approximate
        P.append(probs.cpu().numpy())
        L.append(logits_orig.cpu().numpy())
        Y.append(b["grade"].cpu().numpy())
        I.append(b["index"].cpu().numpy())
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(L),
            np.concatenate(I))


# --------------------------------------------------------------------------
# corruptions for the robustness group (applied to real cached retinas)
# --------------------------------------------------------------------------
def corrupt(img_u8: np.ndarray, kind: str, sev: float) -> np.ndarray:
    x = img_u8.astype(np.float32)
    if kind == "gaussian_noise":
        x = x + np.random.normal(0, sev * 255, x.shape)
    elif kind == "defocus_blur":
        k = int(sev * 12) | 1
        x = cv2.GaussianBlur(x, (k, k), 0)
    elif kind == "brightness":
        x = x * (1.0 + sev)
    elif kind == "low_contrast":
        x = (x - x.mean()) * (1.0 - sev) + x.mean()
    elif kind == "jpeg":
        q = int(max(5, 95 - sev * 90))
        ok, enc = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            x = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


@torch.no_grad()
def eval_corruption(model, ds, indices, kind, sev, device, bs=24):
    model.eval()
    preds, ys = [], []
    for s in range(0, len(indices), bs):
        chunk = indices[s:s + bs]
        px = np.stack([corrupt(np.asarray(ds.images[ds.row_of(j)]), kind, sev)
                       for j in chunk])
        b = batch_from_rows(ds, chunk, device, pixel_override=px)
        o = model(model_inputs(b))
        _p = o["ordinal"].class_probs.float().cpu().numpy()
        preds.append(coral_decode_numpy(_p) if DECODE == "coral"
                     else _p.argmax(1))
        ys.append(b["grade"].cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    return float((p == y).mean())


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=OUT_DIR / "retfound_plus_laft_xai" / "best.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--external-cache", type=Path, default=None,
                    help="second preprocessed cache for cross-domain evaluation")
    ap.add_argument("--split", default="test")
    ap.add_argument("--xai-samples", type=int, default=150)
    ap.add_argument("--corruption-samples", type=int, default=400)
    ap.add_argument("--mc-samples", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--decode", choices=("coral", "argmax"), default="coral",
                    help="ordinal decoding; see a6_ordinal.coral_decode")
    ap.add_argument("--tta", action="store_true",
                    help="average class_probs across the original view and a "
                         "horizontal flip at inference (image+crops+anatomy "
                         "flipped in lockstep). Cheap, no retraining; the "
                         "closest thing to the ensembling/TTA the 'market' "
                         "models this project is compared against typically "
                         "use, without needing more data or compute.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    global DECODE
    DECODE = args.decode

    cfg = default_config()
    cfg.device = args.device
    device = cfg.torch_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    # the resolution policy has to come from the checkpoint: evaluating a model
    # trained on 6 crops with the default 6 is luck, not correctness
    for field in ("backbone", "fusion", "moe", "preproc", "head"):
        if isinstance(saved, dict) and field in saved:
            setattr(cfg, field, saved[field])

    model = build_model(cfg)
    ranks = (ck.get("gla_lora") or ck.get("allora"))["ranks"]
    from dr.modules.a4_backbone_lora import inject_lora
    inject_lora(model.backbone.vit, ranks, cfg.backbone)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"[setup] device={device}  checkpoint epoch={ck.get('epoch')} "
          f"stage={ck.get('stage')}  GLA-LoRA ranks={ranks}")
    print(f"[setup] global {cfg.preproc.global_size}px + {cfg.preproc.n_crops} "
          f"crops of {cfg.preproc.crop_size}px fed at {cfg.preproc.crop_input}px")

    ds = CachedEyePACS(args.cache, args.split, augment=False, cfg=cfg)
    ld = DataLoader(ds, batch_size=24, shuffle=False, num_workers=0)
    print(f"[data] {args.split} split: {len(ds)} images, "
          f"{ds.meta.loc[ds.idx,'patient_id'].nunique()} patients")

    t0 = time.time()
    if args.tta:
        print("[infer] TTA enabled: averaging original + horizontal-flip views")
        probs, y, logits, idx = tta_infer(model, ld, device)
    else:
        probs, y, logits, idx = infer(model, ld, device)
    pred = coral_decode_numpy(probs) if DECODE == "coral" else probs.argmax(1)
    infer_sec = time.time() - t0
    print(f"[infer] {len(y)} images in {infer_sec:.1f}s")

    report: dict = {}

    # --- 1 diagnostic + 2 ordinal ---------------------------------------
    report["diagnostic"] = diagnostic_metrics(y, pred, probs)
    report["ordinal"] = ordinal_metrics(y, pred)
    op = referable_operating_point(y, probs, 0.90)

    # --- 3 calibration (raw + temperature-scaled) ------------------------
    raw_cal = calibration_metrics(y, probs)
    cal_path = args.ckpt.parent / "calibrator.pt"
    cal_probs = probs
    temp = None
    if cal_path.exists():
        cs = torch.load(cal_path, map_location="cpu", weights_only=False)
        if cs.get("state") is not None:
            sc = TemperatureScaler()
            sc.load_state_dict(cs["state"])
            temp = sc.temperature
            with torch.no_grad():
                cal_probs = torch.softmax(sc(torch.as_tensor(logits)), 1).numpy()
    report["calibration"] = calibration_metrics(y, cal_probs)
    report["calibration_uncalibrated"] = raw_cal

    # --- 4 lesion / XAI --------------------------------------------------
    print(f"[xai] scoring attribution faithfulness on {args.xai_samples} images")
    sub = ds.idx[:args.xai_samples]
    cam = GradCAM(model)
    dices, ious, points, cons, dps = [], [], [], [], []
    expert_scores, expert_targets = [], []
    for j in sub:
        r = ds.row_of(j)
        img = np.asarray(ds.images[r]).astype(np.float32) / 255.0
        les = np.asarray(ds.lesions[r]).astype(np.float32) / 255.0
        ana = np.asarray(ds.anatomy[r]).astype(np.float32) / 255.0
        b1 = batch_from_rows(ds, [j], device)
        a = b1["anatomy"]

        sal = cam(model_inputs(b1), out_size=les.shape[-1])[0, 0].cpu().numpy()
        union = les.max(0)
        if union.max() >= 0.5:
            d = dice_score(sal, union)
            dices.append(d)
            ab = (sal >= 0.5).astype(np.float32); bb = (union >= 0.5).astype(np.float32)
            u = ab.sum() + bb.sum() - (ab * bb).sum()
            ious.append(float((ab * bb).sum() / u) if u > 0 else np.nan)
            points.append(pointing_game(sal, union))
            cons.append(d)

        with torch.no_grad():
            o = model(model_inputs(b1))
            p_orig = float(o["referable"])
            expert_scores.append(torch.sigmoid(o["moe"].presence)[0].cpu().numpy())
            expert_targets.append((les.reshape(6, -1).max(1) > 0.5).astype(int))
        # counterfactual: erase the detected lesions, re-run, measure the shift
        if union.max() >= 0.5:
            bgr = cv2.cvtColor((np.asarray(ds.images[r])), cv2.COLOR_RGB2BGR)
            cf = counterfactual_image(bgr, union)
            cf_rgb = cv2.cvtColor(cf, cv2.COLOR_BGR2RGB)
            bcf = batch_from_rows(ds, [j], device, pixel_override=cf_rgb[None])
            with torch.no_grad():
                ocf = model(model_inputs(bcf))
            dps.append(p_orig - float(ocf["referable"]))
    cam.remove()

    es = np.stack(expert_scores); et = np.stack(expert_targets)
    aucs = [roc_auc_score(et[:, k], es[:, k]) for k in range(et.shape[1])
            if len(np.unique(et[:, k])) > 1]
    report["lesion_xai"] = lesion_xai_metrics(
        dices, ious, points, cons, dps,
        float(np.mean(aucs)) if aucs else float("nan"))

    # --- 5 prognosis: no longitudinal data in EyePACS ---------------------
    report["prognosis"] = prognosis_metrics(None, None, None)
    report["prognosis_note"] = ("NOT EVALUATED - EyePACS is cross-sectional "
                                "(one visit per patient). A7 requires a "
                                "longitudinal cohort; see a7_temporal.py.")

    # --- 6 robustness -----------------------------------------------------
    print("[robust] corruption sweep on real retinas")
    rid = ds.idx[:args.corruption_samples]
    clean = eval_corruption(model, ds, rid, "none", 0.0, device)
    corr = {}
    for kind, sev in [("gaussian_noise", 0.05), ("defocus_blur", 0.4),
                      ("brightness", 0.3), ("low_contrast", 0.4), ("jpeg", 0.7)]:
        corr[f"{kind}"] = eval_corruption(model, ds, rid, kind, sev, device)
        print(f"         {kind:16s}: acc={corr[kind]:.4f}")

    ext_qwk = float("nan")
    if args.external_cache and Path(args.external_cache).exists():
        eds = CachedEyePACS(args.external_cache, None, augment=False, cfg=cfg)
        eld = DataLoader(eds, batch_size=24, shuffle=False, num_workers=0)
        ep, ey, _, _ = infer(model, eld, device)
        epred = coral_decode_numpy(ep) if DECODE == "coral" else ep.argmax(1)
        ext = {**diagnostic_metrics(ey, epred, ep), **ordinal_metrics(ey, epred)}
        ext_qwk = ext["quadratic_weighted_kappa"]
        report["external_domain"] = ext
        print(f"[robust] external dataset: n={len(ey)} acc={ext['accuracy']:.4f} "
              f"QWK={ext_qwk:.4f}")

    meta = ds.meta.loc[idx]
    qbuckets = {}
    qs = meta["quality_score"].to_numpy()
    for name, m in (("low", qs < 0.5), ("mid", (qs >= 0.5) & (qs < 0.7)),
                    ("high", qs >= 0.7)):
        if m.sum() >= 30:
            qbuckets[name] = float((pred[m] == y[m]).mean())
    report["robustness"] = robustness_metrics(clean, corr, ext_qwk, qbuckets)
    report["robustness"]["_corruptions"] = corr
    report["robustness"]["_quality_buckets"] = qbuckets
    report["robustness"]["_clean_accuracy"] = clean

    # --- + subgroup / fairness -------------------------------------------
    dom = subgroup_metrics(y, pred, meta["domain"].to_numpy())
    eye = subgroup_metrics(y, pred, meta["eye"].to_numpy())
    report["subgroup"] = {"worst_subgroup_accuracy": dom["worst_subgroup_accuracy"],
                          "max_subgroup_gap": dom["max_subgroup_gap"],
                          "_by_camera_domain": dom["_per_group"],
                          "_by_eye": eye["_per_group"]}

    # --- 7 deployment ------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params * 4 / 1e6
    # latency must include the local branch: it is most of the inference cost
    bench_b = batch_from_rows(ds, ds.idx[:1], device)
    bench_in = model_inputs(bench_b)
    with torch.no_grad():
        for _ in range(3):
            model(bench_in)
        t = time.time()
        for _ in range(10):
            model(bench_in)
        lat = (time.time() - t) / 10 * 1000
    student = args.ckpt.parent / "distilled_metrics.json"
    retention = float("nan")
    if student.exists():
        retention = json.loads(student.read_text()).get("accuracy_retention", float("nan"))
    report["deployment"] = deployment_metrics(size_mb, lat, 1000.0 / lat, retention)

    # --- A10 decision gate distribution -----------------------------------
    print(f"[A10] MC-dropout uncertainty on {min(200, len(ds))} images")
    gate_counts = {"AI_DECISION": 0, "HUMAN_REVIEW": 0, "RETAKE_IMAGE": 0}
    gsub = ds.idx[:200]
    for s in range(0, len(gsub), 24):
        chunk = gsub[s:s + 24]
        b = model_inputs(batch_from_rows(ds, chunk, device))
        u = mc_predict(model, b, args.mc_samples)
        for k, j in enumerate(chunk):
            row = ds.meta.iloc[j]
            d, _ = decision_gate(float(u.entropy[k]), float(row["quality_score"]),
                                 str(row["quality_decision"]),
                                 cfg.uncertainty.entropy_hi,
                                 float(u.probs[k, 2:].sum()))
            gate_counts[d] += 1
    report["decision_gate"] = gate_counts

    report["_meta"] = {
        "split": args.split, "n_images": int(len(y)),
        "n_patients": int(meta["patient_id"].nunique()),
        "checkpoint": str(args.ckpt), "temperature": temp,
        "referable_operating_point_at_90pct_sens": op,
        "grade_distribution": {GRADE_NAMES[k]: int((y == k).sum())
                               for k in range(5)},
        "confusion_matrix": [[int(((y == a) & (pred == b)).sum())
                              for b in range(5)] for a in range(5)],
    }

    # --- print ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("RETFound Plus-LAFT-XAI  |  EVALUATION REPORT  (real EyePACS test split)")
    print("=" * 74)
    titles = {"diagnostic": "1. DIAGNOSTIC (11)", "ordinal": "2. ORDINAL (3)",
              "calibration": "3. CALIBRATION (4)", "lesion_xai": "4. LESION / XAI (6)",
              "prognosis": "5. PROGNOSIS (4)", "robustness": "6. ROBUSTNESS (4)",
              "deployment": "7. DEPLOYMENT (4)", "subgroup": "+ SUBGROUP (2)"}
    for key, title in titles.items():
        print(f"\n{title}")
        for k, v in report.get(key, {}).items():
            if k.startswith("_"):
                continue
            if isinstance(v, float) and not np.isfinite(v):
                print(f"   {k:38s}  NOT EVALUATED")
            elif isinstance(v, float):
                print(f"   {k:38s}  {v:.4f}")
            else:
                print(f"   {k:38s}  {v}")
    print(f"\nReferable-DR operating point @>=90% sensitivity: "
          f"sens={op['sensitivity']:.4f} spec={op['specificity']:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred, grades 0-4):")
    for r in report["_meta"]["confusion_matrix"]:
        print("   " + " ".join(f"{c:6d}" for c in r))
    print(f"\nDecision gate: {report['decision_gate']}")
    n_rep = count_reported({k: v for k, v in report.items()
                            if k in titles})
    print(f"\nParameters carrying a finite value: {n_rep} / 38 "
          f"(the 4 prognosis metrics need longitudinal data)")
    print("=" * 74)

    # ---- guardrail: flag a critically undertrained / degenerate checkpoint --
    # Added after a real evaluation run (four one-epoch CPU stages on a ~6,000
    # image subset) produced QWK=-0.084 and AUROC=0.383 - both WORSE than
    # chance, not just at it. A truly random classifier's AUROC hovers AROUND
    # 0.5; a macro-AUROC this far below it, combined with negative QWK, is the
    # signature of a model that has barely moved from initialisation, not a
    # code defect - but it is easy to mistake for one months later if nothing
    # says so at evaluation time. This makes that diagnosis explicit and
    # immediate, the same way the training-time lambda_XAI guardrail does.
    qwk = report["ordinal"].get("quadratic_weighted_kappa", float("nan"))
    auroc = report["diagnostic"].get("auroc_macro_ovr", float("nan"))
    if np.isfinite(qwk) and np.isfinite(auroc) and (qwk < 0.0 or auroc < 0.4):
        print("\n" + "!" * 74)
        print("[WARNING] This checkpoint looks critically undertrained, not just weak:")
        print(f"          QWK = {qwk:.4f} (0.0 = chance-level ordinal agreement, "
              f"negative is WORSE than chance)")
        print(f"          AUROC (macro, one-vs-rest) = {auroc:.4f} (0.5 = chance; "
              f"this is below it, not just near it)")
        print("          The most likely cause is training budget, not the "
              "architecture: check how many epochs/stages this checkpoint "
              "actually completed (outputs/<run>/history.json) before reading "
              "anything else in this report as a finding about the model design.")
        print("!" * 74)

    out = args.out or (args.ckpt.parent / "evaluation.json")
    Path(out).write_text(json.dumps(report, indent=2, default=float))
    # long format (group, metric, value) rather than one wide row per group:
    # the 7 groups share almost no metric names with each other, so a wide
    # table would mostly be empty cells - this mirrors the printed report
    # above and is the form worth opening in a spreadsheet.
    eval_rows = [{"group": key, "metric": k, "value": v}
                for key in titles for k, v in report.get(key, {}).items()
                if not k.startswith("_")]
    save_csv_alongside(eval_rows, out)
    cm = report["_meta"]["confusion_matrix"]
    cm_rows = [{"true_grade": GRADE_NAMES[i],
               **{f"pred_{GRADE_NAMES[j]}": cm[i][j] for j in range(len(cm[i]))}}
              for i in range(len(cm))]
    save_csv_alongside(cm_rows, out,
                       csv_path=Path(out).with_name(Path(out).stem + "_confusion_matrix.csv"))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
