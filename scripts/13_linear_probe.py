#!/usr/bin/env python3
"""Step 13 - objective 3 without full retraining: linear probing of frozen features.

Research gap 3 says generic pretrained backbones (ImageNet/ResNet/EfficientNet)
lack retinal domain awareness.  Objective 3 says a large model can be adapted
"efficiently, enabling lightweight optimization without full retraining".

Linear probing tests both claims directly and is the standard protocol for
evaluating a foundation model:

  * every backbone weight stays frozen - nothing is retrained,
  * features are extracted once, forward-only,
  * a single linear layer is fitted on top.

The hybrid feature vector is the architecture's own: the global CLS token
concatenated with the mean of the local lesion-crop CLS tokens, so the probe
measures the hybrid extractor, not just the global view.

CAVEAT, found during a later correction pass (see the project report's
"corrections made during the study" log): a linear probe cannot measure
ADAPTABILITY, only frozen-feature linear separability. Objective 3's claim is
that a large model can be ADAPTED efficiently - a probe freezes everything and
adapts nothing, which is a different question. This matters specifically
because RETFound is MAE-pretrained, and MAE representations are documented to
probe poorly by construction relative to supervised pretraining while
matching or exceeding it under fine-tuning (He et al. 2022) - so this
instrument is biased against RETFound before a single image is scored. A
probe result showing RETFound losing to ImageNet is evidence about frozen
linear separability, not about which backbone adapts better. The correct
instrument for the actual efficiency/domain-transfer question is a MATCHED
parameter-efficient fine-tune (scripts/07_compare_models.py's `run_proposed`,
which trains real GLA-LoRA adapters on both backbones under an identical
budget) - treat this script's output as a secondary, weaker-evidence data
point alongside that, not as the answer to Objective 3's domain claim.

    python scripts/13_linear_probe.py --backbone vit_large_patch16_224 \
        --retfound data/checkpoints/RETFound_MAE/pytorch_model.bin --tag retfound
    python scripts/13_linear_probe.py --backbone vit_small_patch16_224 \
        --retfound none --tag imagenet

Objective 3 follow-up (the "one cheap experiment" from the evidence pack):
the original run found frozen RETFound features beaten by frozen ImageNet
features of the same architecture, but the cache holds A1/A2 output
(illumination-normalised, CLAHE-enhanced), not raw fundus pixels. To test
whether that preprocessing - not a lack of retinal domain knowledge - is
responsible, build a second, minimally-processed cache and re-run both arms
against it:

    python scripts/02_preprocess.py --raw-pixels --out data/cache_raw ...
    python scripts/13_linear_probe.py --cache data/cache_raw \
        --backbone vit_large_patch16_224 \
        --retfound data/checkpoints/RETFound_MAE/pytorch_model.bin --tag retfound_raw
    python scripts/13_linear_probe.py --cache data/cache_raw \
        --backbone vit_large_patch16_224 --retfound none --tag imagenet_raw

If RETFound overtakes ImageNet on raw pixels, A2 was destroying the
statistics RETFound relies on. If the ordering holds, the finding stands.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS  # noqa: E402
from dr.metrics import diagnostic_metrics, ordinal_metrics  # noqa: E402
from dr.modules.a4_backbone_lora import RETFoundPlusBackbone  # noqa: E402
from dr.screening import select_operating_point, apply_operating_point  # noqa: E402

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


@torch.no_grad()
def extract(backbone, cache, split, cfg, device, bs, limit=None) -> tuple[np.ndarray, np.ndarray]:
    ds = CachedEyePACS(cache, split, augment=False, cfg=cfg)
    if limit and limit < len(ds.idx):
        ds.idx = ds.idx[:limit]
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    F, Y = [], []
    t0 = time.time()
    for i, b in enumerate(ld):
        x = (b["image"].to(device) - mean) / std
        _, cls, _ = backbone._encode(x)                 # global CLS
        feats = [cls]
        if "crops" in b:                                # + mean of local crops
            c = b["crops"].to(device)
            B, C = c.shape[:2]
            cflat = (c.flatten(0, 1) - mean) / std
            _, ccls, _ = backbone._encode(cflat)
            feats.append(ccls.view(B, C, -1).mean(1))
        F.append(torch.cat(feats, 1).float().cpu().numpy())
        Y.append(b["grade"].numpy())
        if i % 20 == 0:
            n = (i + 1) * bs
            print(f"    {split}: {n} imgs  {n/max(time.time()-t0,1e-6):.1f} img/s",
                  flush=True)
    return np.concatenate(F), np.concatenate(Y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_large_patch16_224")
    ap.add_argument("--retfound", default="data/checkpoints/RETFound_MAE/pytorch_model.bin")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--tag", default="retfound")
    ap.add_argument("--global-size", type=int, default=224)
    ap.add_argument("--n-crops", type=int, default=6,
                    help="local crops folded into the hybrid feature. Default "
                         "matches the production pipeline (PreprocCfg.n_crops="
                         "6); the original objective-3 run used 2, which makes "
                         "the probed feature a weaker version of the "
                         "architecture's own hybrid extractor than the one "
                         "actually trained downstream.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--npv-target", type=float, default=0.985)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per split (smoke testing only)")
    args = ap.parse_args()

    cfg = default_config()
    cfg.device = args.device
    cfg.backbone.name = args.backbone
    cfg.preproc.global_size = args.global_size
    cfg.preproc.n_crops = args.n_crops
    if args.retfound.lower() == "none":
        cfg.backbone.retfound_ckpt = None
        cfg.backbone.require_retfound = False
        cfg.backbone.pretrained = True
    else:
        cfg.backbone.retfound_ckpt = args.retfound
    device = cfg.torch_device()

    backbone = RETFoundPlusBackbone(cfg.backbone, cfg.preproc.global_size)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    n_frozen = sum(p.numel() for p in backbone.parameters())
    print(f"[probe] {args.backbone} | {backbone.status}")
    print(f"[probe] frozen backbone parameters: {n_frozen:,}")
    print(f"[probe] hybrid feature = global CLS + mean of {args.n_crops} local crop CLS")

    feats = {}
    for split in ("train", "val", "test"):
        print(f"  extracting {split} ...", flush=True)
        feats[split] = extract(backbone, args.cache, split, cfg, device,
                               args.batch_size, args.limit)

    Xtr, ytr = feats["train"]; Xva, yva = feats["val"]; Xte, yte = feats["test"]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    d = Xtr.shape[1]
    print(f"[probe] feature dim {d}  train {Xtr.shape[0]}  val {Xva.shape[0]}  test {Xte.shape[0]}")

    # class_weight balanced is the loss re-weighting of objective 1, applied to
    # the only trainable component here.
    best, best_C = None, None
    for C in (0.001, 0.01, 0.1, 1.0):
        # sklearn >=1.7 dropped multi_class; lbfgs is multinomial by default.
        # class_weight="balanced" is objective 1's loss re-weighting applied to
        # the only trainable component in this experiment.
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)
        m = ordinal_metrics(yva, pv.argmax(1))
        score = m["quadratic_weighted_kappa"]
        print(f"    C={C:<7} val QWK {score:.4f}")
        if best is None or score > best:
            best, best_C, clf_best = score, C, clf
    print(f"[probe] selected C={best_C} on validation (QWK {best:.4f})")

    n_probe = d * 5 + 5
    pte = clf_best.predict_proba(Xte)
    pred = pte.argmax(1)
    m = {**diagnostic_metrics(yte, pred, pte), **ordinal_metrics(yte, pred)}
    for g in range(5):
        mask = yte == g
        m[f"recall_grade_{g}"] = float((pred[mask] == g).mean()) if mask.any() else float("nan")

    # screening operating point: fitted on val, applied unchanged to test
    pva = clf_best.predict_proba(Xva)
    k = 3
    opv = select_operating_point((yva >= k).astype(int), pva[:, k:].sum(1),
                                 "sight_threatening", "npv", args.npv_target)
    opt = apply_operating_point((yte >= k).astype(int), pte[:, k:].sum(1),
                                opv.threshold, "sight_threatening", "npv", args.npv_target)

    print(f"\n{'='*74}")
    print(f"LINEAR PROBE - {args.tag}  (backbone 100% frozen)")
    print(f"{'='*74}")
    print(f"  frozen backbone params   {n_frozen:>12,}")
    print(f"  trained probe params     {n_probe:>12,}   ({100*n_probe/n_frozen:.4f}% of backbone)")
    print(f"  test accuracy            {m['accuracy']:.4f}")
    print(f"  test balanced accuracy   {m['balanced_accuracy']:.4f}")
    print(f"  test QWK                 {m['quadratic_weighted_kappa']:.4f}")
    print(f"  test STDR AUC            {m['auc_sight_threatening']:.4f}")
    print(f"  test referable AUC       {m['auc_referable_dr']:.4f}")
    print(f"  per-grade recall         {[round(m[f'recall_grade_{g}'],3) for g in range(5)]}")
    print(f"  screening @ NPV>={args.npv_target}: cleared {opt.workload_reduction*100:.1f}% "
          f"NPV {opt.npv:.4f} missed {opt.missed} (thr fitted on val)")

    out = {"tag": args.tag, "backbone": args.backbone,
           "retfound": args.retfound if args.retfound.lower() != "none" else None,
           "frozen_backbone_params": int(n_frozen), "trained_probe_params": int(n_probe),
           "trainable_fraction_pct": float(100 * n_probe / n_frozen),
           "selected_C": best_C, "feature_dim": int(d),
           "test": m, "screening_selected_on_val": opv.as_dict(),
           "screening_applied_to_test": opt.as_dict()}
    p = OUT_DIR / f"linear_probe_{args.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    save_csv_alongside(out, p)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
