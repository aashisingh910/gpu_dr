#!/usr/bin/env python3
"""Step 07 - Model Comparison (20+ models), matching the figure's comparison block.

Families covered:
  * CNNs            ResNet-18/50, EfficientNet-B0, ConvNeXt-Tiny, DenseNet-121
  * Transformers    ViT-S/B, DeiT-S, Swin-T, DINOv2-S
  * Foundation      RETFound / RETFound Plus backbone (needs --retfound-ckpt)
  * Adaptation      full fine-tune vs linear probe vs uniform LoRA vs AL-LoRA
  * Ablations       each proposed component switched off in turn
  * Complete model  and its distilled student

Every entry is trained and evaluated under the *same* budget (epochs, subset,
sampler, seed) so the numbers are comparable to each other.  The budget is
printed in the header and stored in the JSON - a compute-limited comparison
that says so is useful; one that hides it is not.

    python scripts/07_compare_models.py --epochs 2 --max-train 6000
    python scripts/07_compare_models.py --only resnet50,vit_small,complete
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# see scripts/03_train.py for why these are set before numpy/torch import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS  # noqa: E402
from dr.metrics import (calibration_metrics, diagnostic_metrics,  # noqa: E402
                        ordinal_metrics)
from dr.model import build_model  # noqa: E402
from dr.modules.a3_lesion_moe import lesion_supervision_loss  # noqa: E402
from dr.modules.a6_ordinal import class_balanced_focal_loss, ordinal_loss  # noqa: E402
from dr.modules.a5_fusion_gnn import pathology_anatomy_consistency  # noqa: E402
from dr.modules.a6_ordinal import (boundary_loss, hierarchical_loss,  # noqa: E402
                                   dqk_loss, ordinal_contrastive_loss)
from dr.modules.a9_xai import (attribution_consistency_loss,  # noqa: E402
                               xai_weight_for_stage)


def model_inputs(b, device):
    """Only the keys the model consumes, moved to the device."""
    return {k: b[k].to(device) for k in
            ("image", "crops", "anatomy", "quality_axes", "hardness") if k in b}

# name -> (family, timm model, adaptation mode)
BASELINES = [
    ("resnet18",         "CNN",         "resnet18",                  "full"),
    ("resnet50",         "CNN",         "resnet50",                  "full"),
    ("resnext50",        "CNN",         "resnext50_32x4d",           "full"),
    ("efficientnet_b0",  "CNN",         "efficientnet_b0",           "full"),
    ("mobilenetv3_large", "CNN",        "mobilenetv3_large_100",     "full"),
    ("regnety_016",      "CNN",         "regnety_016",               "full"),
    ("convnext_tiny",    "CNN",         "convnext_tiny",             "full"),
    ("densenet121",      "CNN",         "densenet121",               "full"),
    ("vit_tiny",         "Transformer", "vit_tiny_patch16_224",      "full"),
    ("vit_small",        "Transformer", "vit_small_patch16_224",     "full"),
    ("vit_base",         "Transformer", "vit_base_patch16_224",      "full"),
    ("deit_small",       "Transformer", "deit_small_patch16_224",    "full"),
    ("swin_tiny",        "Transformer", "swin_tiny_patch4_window7_224", "full"),
    ("dinov2_small",     "Transformer", "vit_small_patch14_dinov2.lvd142m", "full"),
    ("vit_small_linear", "Adaptation",  "vit_small_patch16_224",     "linear"),
    ("vit_small_lora",   "Adaptation",  "vit_small_patch16_224",     "lora"),
]

# proposed-model variants (all use the full A1..A10 pipeline, PLUS the
# Objective 1/2/3/4 fixes and the loss-weighting/crop-evidence additions,
# unless ablated below)
PROPOSED = [
    ("complete",        ()),                       # every component + fix on
    ("abl_no_moe",      ("moe",)),                 # - A3 lesion experts
    ("abl_no_fusion",   ("fusion",)),              # - A5 lesion->anatomy edges
    ("abl_no_ordinal",  ("ordinal",)),             # - A6 CORAL (plain softmax)
    ("abl_no_hierarchical", ("hierarchical",)),    # - A6 any/referable/stdr/pdr heads
    ("abl_no_xai",      ("xai",)),                 # - A9 consistency loss
    ("abl_no_allora",   ("allora",)),              # uniform LoRA rank instead
    ("abl_no_local",    ("local",)),               # - local lesion crops entirely
    ("abl_no_alpp",     ("alpp",)),                # - learned adaptive preprocessing
    ("abl_no_bidirectional", ("bidirectional",)),  # - A5 bidirectional cross-attn
    ("abl_no_clinical", ("clinical",)),            # - A8 clinical metadata fusion
    ("abl_no_crop_moe", ("crop_moe",)),            # - lesion MoE reading local crops (novelty fix)
    ("abl_fixed_loss_weights", ("_fixed_loss_weights",)),  # hand-tuned LossCfg weights
    ("abl_no_ladder_fix", ("_no_ladder_fix",)),    # ladder at 1x LR, uniform init (Objective 1 regression check)
    ("abl_no_lesion_gate", ("lesion_gate",)),      # DR-NOVA: - LGOP direct lesion->ordinal residual
    ("abl_no_domain_adapt", ("domain_adapt",)),    # DR-NOVA: - DAFA domain-adversarial alignment
]


class BaselineNet(nn.Module):
    """timm backbone + linear grading head, with the adaptation mode applied."""

    def __init__(self, timm_name: str, mode: str, n_grades: int = 5,
                 img_size: int = 224):
        super().__init__()
        import timm
        kw = {}
        if "vit_" in timm_name or "deit" in timm_name:
            kw["img_size"] = img_size
        self.backbone = timm.create_model(timm_name, pretrained=True,
                                          num_classes=0, **kw)
        # measured, not assumed: `num_features` disagrees with the real pooled
        # width on some families (e.g. mobilenetv3 reports 576, emits 1024)
        with torch.no_grad():
            self.backbone.eval()
            d = int(self.backbone(torch.zeros(1, 3, img_size, img_size)).shape[-1])
        self.head = nn.Linear(d, n_grades)
        self.mode = mode
        if mode in ("linear", "lora"):
            for p in self.backbone.parameters():
                p.requires_grad = False
        if mode == "lora":
            from dr.config import BackboneCfg
            from dr.modules.a4_backbone_lora import inject_lora
            cfg = BackboneCfg()
            n_blocks = len(self.backbone.blocks)
            # uniform rank -> the control condition for AL-LoRA
            inject_lora(self.backbone, [8] * n_blocks, cfg)

    def forward(self, x):
        return self.head(self.backbone(x))


def make_loaders(cache, batch_size, workers, max_train, seed, max_test=None, cfg=None):
    # CachedEyePACS.cfg has no default - calling it without one (as this
    # function did before) raises TypeError the instant make_loaders() is
    # called, before a single baseline model ever runs.
    if cfg is None:
        cfg = default_config()
    tr = CachedEyePACS(cache, "train", augment=True, cfg=cfg)
    te = CachedEyePACS(cache, "test", augment=False, cfg=cfg)
    rng = np.random.default_rng(seed)
    if max_train and max_train < len(tr):
        tr.idx = rng.choice(tr.idx, max_train, replace=False)
    if max_test and max_test < len(te):
        # same fixed subset for every entry, so the ranking stays comparable
        te.idx = rng.choice(te.idx, max_test, replace=False)
    labels = tr.labels()
    counts = np.bincount(labels, minlength=5).astype(np.float64)
    counts[counts == 0] = 1
    w = np.sqrt(1.0 / counts)[labels]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    len(labels), replacement=True)
    return (DataLoader(tr, batch_size=batch_size, sampler=sampler,
                       num_workers=workers, drop_last=True,
                       pin_memory=torch.cuda.is_available()),
            DataLoader(te, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=torch.cuda.is_available()),
            torch.as_tensor(np.bincount(labels, minlength=5).astype(np.float32)))


@torch.no_grad()
def eval_baseline(model, loader, device):
    model.eval()
    P, Y = [], []
    for b in loader:
        P.append(torch.softmax(model(b["image"].to(device)), 1).float().cpu().numpy())
        Y.append(b["grade"].numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    return p, y


def run_baseline(name, timm_name, mode, args, device, loaders):
    train_ld, test_ld, counts = loaders
    model = BaselineNet(timm_name, mode, img_size=args.image_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.02)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in train_ld:
            x, y = b["image"].to(device), b["grade"].to(device)
            loss = class_balanced_focal_loss(model(x), y, counts,
                                             beta=args.cb_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    train_min = (time.time() - t0) / 60
    p, y = eval_baseline(model, test_ld, device)
    n_tr = sum(q.numel() for q in params)
    n_all = sum(q.numel() for q in model.parameters())
    return p, y, {"trainable_params": n_tr, "total_params": n_all,
                  "train_minutes": round(train_min, 2)}


def run_proposed(name, ablate, args, device, loaders):
    train_ld, test_ld, counts = loaders
    cfg = default_config()
    cfg.backbone.name = args.backbone
    cfg.preproc.global_size = args.image_size
    if "_fixed_loss_weights" in ablate:
        cfg.loss.learned_weights = False     # Objective-improvement regression check
    ladder_lr_mult = (1.0 if "_no_ladder_fix" in ablate
                      else cfg.train.ladder_lr_mult)
    real_ablate = tuple(a for a in ablate
                        if a not in ("allora", "_fixed_loss_weights", "_no_ladder_fix"))
    cfg.ablate = real_ablate
    model = build_model(cfg).to(device)

    calib = next(iter(train_ld))
    if "allora" in ablate:      # control: uniform rank across every block
        from dr.modules.a4_backbone_lora import inject_lora
        n_blocks = len(model.backbone.vit.blocks)
        r = (cfg.backbone.lora_rank_min + cfg.backbone.lora_rank_max) // 2
        info = {"ranks": [r] * n_blocks,
                "lora_params": inject_lora(model.backbone.vit, [r] * n_blocks,
                                           cfg.backbone)}
        model.adapters_ready = True
    else:
        cimg = (calib["image"].to(device) - model.norm_mean) / model.norm_std
        info = model.fit_adapters(cimg, calib["lesion_masks"].to(device))
    model.to(device)

    # Objective 1 fix: seed the CORAL ladder at the training prior instead of
    # a uniform, data-independent gap ("abl_no_ladder_fix" reverts both this
    # and the ladder's LR multiplier, to isolate the fix's contribution)
    if "_no_ladder_fix" not in ablate:
        model.init_ordinal_ladder_from_prior(counts)

    # the comparison harness trains one short stage, so it uses the LoRA stage's
    # learning rates rather than the full four-stage schedule
    stage = cfg.train.stages[1]
    model.set_stage(stage)
    groups = model.optimizer_param_groups(stage, 0.02, ladder_lr_mult)
    for g in groups:                    # honour the harness's --lr override
        g["lr"] = args.lr * (g["lr"] / stage.head_lr) if stage.head_lr else args.lr
    params = [q for g in groups for q in g["params"]]
    opt = torch.optim.AdamW(groups)
    lo = cfg.loss
    use_ordinal = "ordinal" not in ablate
    use_moe = "moe" not in ablate
    use_hier = "hierarchical" not in ablate
    w_xai = 0.0 if "xai" in ablate else xai_weight_for_stage(cfg.xai, 1, 1.0)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in train_ld:
            y = b["grade"].to(device); m = b["lesion_masks"].to(device)
            out = model(model_inputs(b, device))
            ordi = out["ordinal"]
            terms = {"cbf": class_balanced_focal_loss(
                ordi.logits_ce, y, counts, beta=args.cb_beta,
                weight_power=cfg.sampling.class_weight_power)}
            if use_ordinal:
                terms["ord"] = ordinal_loss(ordi, y)
                terms["bnd"] = boundary_loss(ordi, y, lo.boundary_gamma)
                terms["con"] = ordinal_contrastive_loss(
                    ordi.projection, y, lo.contrastive_temp)
            if use_hier:
                terms["hier"] = hierarchical_loss(
                    ordi, y, cfg.head.hierarchy_thresholds)
            if use_moe:
                terms["les"] = lesion_supervision_loss(
                    out["moe"], m, (m.amax((2, 3)) > 0.5).float())
            if "fusion" not in ablate:
                terms["pa"] = pathology_anatomy_consistency(out["fusion"])
            if getattr(lo, "use_dqk_loss", False):
                terms["dqk"] = dqk_loss(ordi.class_probs, y, ordi.class_probs.shape[1])

            if getattr(model, "loss_weighting", None) is not None:
                loss, _ = model.loss_weighting(terms)
            else:
                fixed_w = {"cbf": lo.w_cbf, "ord": lo.w_ordinal, "bnd": lo.w_boundary,
                          "con": lo.w_contrastive, "hier": lo.w_hierarchical,
                          "les": lo.w_lesion, "pa": lo.w_pa, "dqk": getattr(lo, "w_dqk", 0.5)}
                loss = sum(fixed_w[k] * v for k, v in terms.items())
            if w_xai > 0:
                loss = loss + w_xai * attribution_consistency_loss(
                    out["attribution"], m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    train_min = (time.time() - t0) / 60

    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in test_ld:
            o = model(model_inputs(b, device))
            probs = (o["ordinal"].class_probs if use_ordinal
                     else torch.softmax(o["ordinal"].logits_ce, 1))
            P.append(probs.float().cpu().numpy()); Y.append(b["grade"].numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    n_tr = sum(q.numel() for q in params)
    n_all = sum(q.numel() for q in model.parameters())
    return p, y, {"trainable_params": n_tr, "total_params": n_all,
                  "train_minutes": round(train_min, 2),
                  "lora_ranks": info["ranks"]}


def bootstrap_qwk_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                        n_boot: int = 2000, seed: int = 7) -> dict:
    """Paired bootstrap CI + two-sided p-value on QWK(a) - QWK(b), on the
    SAME test images for both models (matched pairs, not independent
    samples) - the same protocol already used for the CNR statistic in
    Objective 2's evidence pack, applied here to the model-comparison table
    so "ours is best" is a tested claim rather than a single-run point
    estimate.
    """
    from dr.metrics import ordinal_metrics
    n = len(y)
    rng = np.random.default_rng(seed)
    qwk_a = ordinal_metrics(y, pred_a)["quadratic_weighted_kappa"]
    qwk_b = ordinal_metrics(y, pred_b)["quadratic_weighted_kappa"]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)          # same resample indexes both arms - paired
        qa = ordinal_metrics(y[idx], pred_a[idx])["quadratic_weighted_kappa"]
        qb = ordinal_metrics(y[idx], pred_b[idx])["quadratic_weighted_kappa"]
        deltas[i] = qa - qb
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    # two-sided bootstrap p-value: fraction of resamples where the sign of
    # the delta flips relative to the observed point estimate
    observed = qwk_a - qwk_b
    p = float(2 * min((deltas <= 0).mean(), (deltas > 0).mean()))
    return {"qwk_a": float(qwk_a), "qwk_b": float(qwk_b),
            "delta": float(observed), "ci95": [float(lo), float(hi)],
            "p_value": p, "significant_at_0.05": bool(p < 0.05),
            "n_boot": n_boot, "n_images": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "comparison")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-test", type=int, default=None,
                    help="fixed test subset shared by all entries")
    ap.add_argument("--cb-beta", type=float, default=0.0,
                    help="class-balanced beta. The loader already rebalances "
                         "with a sqrt-inverse-frequency sampler, so the default "
                         "0.0 makes the class weights uniform and keeps only the "
                         "focal term; stacking both corrections drives short runs "
                         "to predict rare classes and collapses accuracy.")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", default=None, help="comma-separated subset of names")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = default_config(); cfg.device = args.device
    device = cfg.torch_device()
    args.out.mkdir(parents=True, exist_ok=True)
    keep = set(args.only.split(",")) if args.only else None

    loaders = make_loaders(args.cache, args.batch_size, args.workers,
                           args.max_train, args.seed, args.max_test, cfg=cfg)
    budget = {"epochs": args.epochs, "max_train": args.max_train,
              "max_test": args.max_test,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "sampler": "sqrt-inverse-frequency", "cb_beta": args.cb_beta,
              "device": str(device),
              "n_test": len(loaders[1].dataset)}
    print("=" * 78)
    print("MODEL COMPARISON - identical budget for every entry")
    print(f"  {budget}")
    print("=" * 78)

    results_path = args.out / "comparison.json"
    preds_path = args.out / "predictions.npz"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["_budget"] = budget
    raw_preds = dict(np.load(preds_path)) if preds_path.exists() else {}

    jobs = [(n, "baseline", (f, t, m)) for n, f, t, m in BASELINES]
    jobs += [(n, "proposed", (a,)) for n, a in PROPOSED]

    for name, kind, spec in jobs:
        if keep and name not in keep:
            continue
        if name in results:
            print(f"[skip] {name} already in {results_path}")
            continue
        print(f"\n[run] {name} ({kind})", flush=True)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        try:
            if kind == "baseline":
                family, timm_name, mode = spec
                p, y, extra = run_baseline(name, timm_name, mode, args, device, loaders)
            else:
                family = "Proposed"
                p, y, extra = run_proposed(name, spec[0], args, device, loaders)
            pred = p.argmax(1)
            m = {**diagnostic_metrics(y, pred, p), **ordinal_metrics(y, pred),
                 **calibration_metrics(y, p)}
            results[name] = {"family": family, **extra, **m}
            raw_preds["_y"] = y                     # same test set/order for every entry
            raw_preds[name] = pred
            print(f"      acc={m['accuracy']:.4f} QWK={m['quadratic_weighted_kappa']:.4f} "
                  f"refAUC={m['auc_referable_dr']:.4f} "
                  f"trainable={extra['trainable_params']:,} "
                  f"({extra['train_minutes']} min)")
        except Exception as e:  # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {e}")
            results[name] = {"family": kind, "error": f"{type(e).__name__}: {e}"}
        results_path.write_text(json.dumps(results, indent=2, default=float))
        save_csv_alongside({k: v for k, v in results.items() if not k.startswith("_")},
                           results_path)
        np.savez(preds_path, **raw_preds)

    # ---- leaderboard ------------------------------------------------------
    rows = [(k, v) for k, v in results.items()
            if not k.startswith("_") and "error" not in v]
    rows.sort(key=lambda kv: -kv[1].get("quadratic_weighted_kappa", -9))
    print("\n" + "=" * 100)
    print(f"{'model':22s} {'family':12s} {'acc':>7s} {'QWK':>7s} {'refAUC':>7s} "
          f"{'F1':>7s} {'ECE':>7s} {'trainable':>12s} {'min':>6s}")
    print("-" * 100)
    for k, v in rows:
        print(f"{k:22s} {v.get('family','')[:12]:12s} "
              f"{v.get('accuracy',float('nan')):7.4f} "
              f"{v.get('quadratic_weighted_kappa',float('nan')):7.4f} "
              f"{v.get('auc_referable_dr',float('nan')):7.4f} "
              f"{v.get('f1_macro',float('nan')):7.4f} "
              f"{v.get('ece',float('nan')):7.4f} "
              f"{v.get('trainable_params',0):12,d} "
              f"{v.get('train_minutes',0):6.1f}")
    print("=" * 100)
    print(f"{len(rows)} models compared | saved to {results_path}")

    # ---- comparison.csv: the leaderboard itself (QWK-sorted, one row per
    # model, errored runs excluded) - the table most people actually want to
    # open, as opposed to results.csv's unsorted full dump of every field.
    comparison_path = args.out / "comparison.csv"
    save_csv_alongside([{"model": k, **v} for k, v in rows], results_path,
                       csv_path=comparison_path)

    # ---- paired bootstrap significance: "complete" vs every other model ---
    if "complete" in raw_preds and "_y" in raw_preds:
        y_all = raw_preds["_y"]
        sig = {}
        for k, v in rows:
            if k == "complete" or k not in raw_preds:
                continue
            sig[k] = bootstrap_qwk_delta(y_all, raw_preds["complete"], raw_preds[k])
        sig_path = args.out / "significance.json"
        sig_path.write_text(json.dumps(sig, indent=2, default=float))
        save_csv_alongside(sig, sig_path)

        print("\n" + "=" * 100)
        print("PAIRED BOOTSTRAP SIGNIFICANCE - QWK(complete) - QWK(other), "
              "2000 resamples, same test images")
        print("-" * 100)
        print(f"{'vs':22s} {'QWK complete':>13s} {'QWK other':>10s} "
              f"{'delta':>8s} {'95% CI':>18s} {'p':>8s} {'sig?':>6s}")
        for k, s in sorted(sig.items(), key=lambda kv: kv[1]["delta"]):
            ci = f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"
            print(f"{k:22s} {s['qwk_a']:13.4f} {s['qwk_b']:10.4f} "
                  f"{s['delta']:+8.4f} {ci:>18s} {s['p_value']:8.4f} "
                  f"{'yes' if s['significant_at_0.05'] else 'no':>6s}")
        print("=" * 100)
        n_better = sum(1 for s in sig.values()
                      if s["delta"] > 0 and s["significant_at_0.05"])
        n_worse = sum(1 for s in sig.values()
                     if s["delta"] < 0 and s["significant_at_0.05"])
        print(f"'complete' is significantly BETTER than {n_better}/{len(sig)} "
              f"other entries, significantly WORSE than {n_worse}/{len(sig)}, "
              f"and not significantly different from the rest.")
        print(f"saved to {sig_path}")
    elif "complete" not in raw_preds:
        print("\n[significance] 'complete' has no cached predictions yet - "
              "run it (or don't --only exclude it) to get the significance table.")


if __name__ == "__main__":
    main()
