#!/usr/bin/env python3
"""Step 05 - run the complete A1..A10 pipeline on ONE real fundus photograph
and print the Final Clinical AI Report.

    python scripts/05_predict.py --image data/raw/eyepacs-224/.../16_left.png

Every block executes end-to-end on the actual pixels: quality gate, adaptive
preprocessing, lesion MoE, backbone + AL-LoRA, graph fusion, ordinal grading,
MC-dropout uncertainty, Grad-CAM attribution, counterfactual erasure and the
clinical decision gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import OUT_DIR, default_config  # noqa: E402
from dr.data.lesion_priors import LESION_NAMES, LesionPriorExtractor  # noqa: E402
from dr.model import build_model  # noqa: E402
from dr.modules import a1_quality, a2_preprocess  # noqa: E402
from dr.modules.a4_backbone_lora import inject_lora  # noqa: E402
from dr.modules.a9_xai import GradCAM, counterfactual_image  # noqa: E402
from dr.modules.a10_uncertainty import decision_gate, mc_predict  # noqa: E402
from dr.report import ClinicalReport, render_panel  # noqa: E402
from dr.screening import triage  # noqa: E402

def build_input(pre, priors, cfg, device):
    """Global view + local lesion crops for a single freshly-read image.

    Mirrors `CachedEyePACS.__getitem__` exactly - crops cut at the native
    retinal-field resolution first, global resize second, pixels left in [0,1]
    because the model's ALPP stage normalises internally.  Any divergence here
    would make single-image inference silently disagree with training.
    """
    from dr.data.lesion_priors import generate_lesion_crops
    pc = cfg.preproc
    rgb = cv2.cvtColor(pre.image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    centres = generate_lesion_crops(
        priors.masks, priors.anatomy, pre.fov_mask, n_crops=pc.n_crops,
        crop_frac=pc.crop_size / max(pc.cache_size, 1),
        disc_center=priors.disc_center, macula_center=priors.macula_center)

    H = rgb.shape[0]
    half = pc.crop_size // 2
    crops = []
    for cx, cy in centres:
        # generate_lesion_crops already returns centres in fov_mask's own
        # pixel grid (0..H-1 here) - not normalised 0-1 - so no extra "* H".
        x0 = int(np.clip(round(float(cx)), half, max(half, H - half)))
        y0 = int(np.clip(round(float(cy)), half, max(half, H - half)))
        win = rgb[max(0, y0 - half):y0 + half, max(0, x0 - half):x0 + half]
        if win.shape[0] != pc.crop_size or win.shape[1] != pc.crop_size:
            win = cv2.copyMakeBorder(win, 0, max(0, pc.crop_size - win.shape[0]),
                                     0, max(0, pc.crop_size - win.shape[1]),
                                     cv2.BORDER_CONSTANT, value=0)
        crops.append(cv2.resize(win, (pc.crop_input, pc.crop_input),
                                interpolation=cv2.INTER_AREA))
    g = (cv2.resize(rgb, (pc.global_size, pc.global_size),
                    interpolation=cv2.INTER_AREA) if H != pc.global_size else rgb)

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32,
                               device=device)
    return {
        "image": t(g.transpose(2, 0, 1))[None],
        "crops": t(np.stack(crops).transpose(0, 3, 1, 2))[None],
        "anatomy": t(priors.anatomy)[None],
        "quality_axes": t(np.zeros(5, np.float32))[None],
        "hardness": t(np.zeros(1, np.float32)),
    }


def load_model(ckpt_path: Path, device):
    cfg = default_config()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    for field in ("backbone", "fusion", "moe", "preproc", "head"):
        if isinstance(saved, dict) and field in saved:
            setattr(cfg, field, saved[field])
    model = build_model(cfg)
    inject_lora(model.backbone.vit, (ck.get("gla_lora") or ck.get("allora"))["ranks"], cfg.backbone)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=OUT_DIR / "retfound_plus_laft_xai" / "best.pt")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "predictions")
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg0 = default_config(); cfg0.device = args.device
    device = cfg0.torch_device()
    model, cfg = load_model(args.ckpt, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"could not read image: {args.image}")
    print(f"[input] {args.image}  {bgr.shape[1]}x{bgr.shape[0]}")

    # --- A1 + A2 + priors ------------------------------------------------
    qrep = a1_quality.assess(bgr, cfg.quality)
    print(f"[A1] Q*={qrep.score:.3f} -> {qrep.decision} (domain {qrep.domain})")
    print("[A1] axes " + "  ".join(f"{k}={v:.3f}" for k, v in qrep.axes().items()))
    pre = a2_preprocess.run(bgr, qrep, cfg.preproc)
    priors = LesionPriorExtractor(out_size=cfg.moe.mask_size)(pre.image, pre.fov_mask)
    print(f"[A2] I* {pre.image.shape}; disc={priors.disc_center} "
          f"macula={priors.macula_center}")

    batch = build_input(pre, priors, cfg, device)
    batch["quality_axes"] = torch.as_tensor(
        np.asarray([[qrep.q_quality, qrep.q_domain, qrep.q_lesion,
                     qrep.q_blur, qrep.q_illumination]], np.float32), device=device)
    a = batch["anatomy"]

    # --- A3..A8 forward ---------------------------------------------------
    with torch.no_grad():
        out = model(batch)
    probs = out["ordinal"].class_probs[0].cpu().numpy()
    grade = int(probs.argmax())
    alphas = out["moe"].alphas[0].cpu().numpy()
    pres = torch.sigmoid(out["moe"].presence)[0].cpu().numpy()
    evid = torch.sigmoid(out["moe"].evidence)[0].cpu().numpy()

    # --- A10 uncertainty ---------------------------------------------------
    u = mc_predict(model, batch, args.mc_samples)
    unc = {"entropy": float(u.entropy[0]), "aleatoric": float(u.aleatoric[0]),
           "epistemic": float(u.epistemic[0]), "std": float(u.std[0])}

    # --- A9 attribution + counterfactual -----------------------------------
    cam = GradCAM(model)
    sal = cam(batch, out_size=cfg.preproc.global_size)[0, 0].cpu().numpy()
    cam.remove()
    union = evid.max(0)
    cf_bgr = counterfactual_image(pre.image, union)
    cf_pre = type(pre)(image=cf_bgr, fov_mask=pre.fov_mask, vessels=pre.vessels,
                       lesion_boost=pre.lesion_boost)
    cf_batch = build_input(cf_pre, priors, cfg, device)
    cf_batch["quality_axes"] = batch["quality_axes"]
    with torch.no_grad():
        ocf = model(cf_batch)
    delta = float(out["referable"]) - float(ocf["referable"])

    # --- A9 lesion-grounded explanation chain -------------------------------
    from dr.modules.a9_xai import explain, explanation_report  # noqa: E402
    grounded = explain(model, batch, index=0, topk=cfg.xai.grounding_topk,
                       uncertainty=unc["entropy"])
    print()
    print(explanation_report(grounded))
    print()

    # --- screening triage (primary clinical output) -------------------------
    # Uses the sight-threatening operating point fitted on the validation split
    # by scripts/09_screening.py; falls back to a plain 0.5 cut if absent.
    triage_info = None
    screen_path = args.ckpt.parent / "screening.json"
    if screen_path.exists():
        sj = json.loads(screen_path.read_text())
        dep = sj.get("_deployable", {}).get("sight_threatening")
        if dep:
            score = float(out["sight_threatening"])
            thr = float(dep["threshold"])
            decision, reason_t = triage(score, {"refer": thr})
            triage_info = {
                "decision": decision, "task": "sight-threatening DR (grade >= 3)",
                "score": score, "threshold": thr, "reason": reason_t,
                "npv": float(dep["npv"]),
                "workload_reduction": float(dep["workload_reduction"]),
            }

    # --- decision gate ------------------------------------------------------
    dec, reason = decision_gate(unc["entropy"], qrep.score, qrep.decision,
                                cfg.uncertainty.entropy_hi, float(out["referable"]))

    rep = ClinicalReport(
        image_id=args.image.stem, grade=grade, grade_probs=probs,
        referable=float(out["referable"]),
        sight_threatening=float(out["sight_threatening"]),
        lesions={n: {"present": bool(pres[i] >= 0.5), "prob": float(pres[i]),
                     "alpha": float(alphas[i])} for i, n in enumerate(LESION_NAMES)},
        quality={"score": qrep.score, "decision": qrep.decision,
                 "gradability": qrep.gradability, "blur": qrep.blur,
                 "illumination": qrep.illumination, "artifact": qrep.artifact,
                 "fov": qrep.fov, "domain": qrep.domain},
        uncertainty=unc, decision=dec, decision_reason=reason,
        counterfactual_delta=delta, triage=triage_info,
    )
    print("\n" + rep.to_text())

    png = render_panel(pre.image, sal, union, cf_bgr,
                       str(args.out_dir / f"{args.image.stem}_report.png"))
    txt = args.out_dir / f"{args.image.stem}_report.txt"
    txt.write_text(rep.to_text())
    print(f"\n[saved] {png}\n[saved] {txt}")


if __name__ == "__main__":
    main()
