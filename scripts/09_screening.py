#!/usr/bin/env python3
"""Step 09 - Screening-triage evaluation.

Reframes the grader as a referral-safety system: thresholds are selected on the
VALIDATION split against a safety constraint, then applied unchanged to the
held-out TEST split and to the external cohort.

    python scripts/09_screening.py --external-cache data/cache_external

Reports, per task and per safety target, the referral rate, the workload the
system removes, how many diseased patients land in the cleared bucket, and the
NPV of that bucket.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS  # noqa: E402
from dr.model import build_model  # noqa: E402
from dr.modules.a4_backbone_lora import inject_lora  # noqa: E402
from dr.screening import (apply_operating_point,  # noqa: E402
                          metrics_at, select_operating_point)

TASKS = {
    "referable_dr": (2, "Referable DR (grade >= 2)"),
    "sight_threatening": (3, "Sight-threatening DR (grade >= 3)"),
}
TARGETS = [("sensitivity", 0.90), ("sensitivity", 0.95),
           ("npv", 0.98), ("npv", 0.985), ("npv", 0.99)]


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, thr: float,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 1337) -> dict:
    """Percentile CIs for the frozen-threshold operating point.

    The test split carries ~46 sight-threatening cases, so an NPV quoted to
    four decimals is a point estimate on a handful of events.  Resampling
    patients with replacement is the cheapest honest way to say how much of
    that precision is real.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    npv, cleared, sens = [], [], []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb, sb = y[b], scores[b]
        if yb.sum() == 0:                     # no positives drawn: undefined
            continue
        m = metrics_at(yb, sb, thr)
        if np.isfinite(m["npv"]):
            npv.append(m["npv"])
        cleared.append(m["workload_reduction"])
        sens.append(m["sensitivity"])
    q = lambda a: ([float(np.quantile(a, alpha / 2)),
                    float(np.quantile(a, 1 - alpha / 2))] if a else [float("nan")] * 2)
    return {"n_boot": len(cleared), "npv_ci": q(npv),
            "cleared_ci": q(cleared), "sensitivity_ci": q(sens)}


@torch.no_grad()
def infer(model, cache, split, device, cfg, bs=24):
    ds = CachedEyePACS(cache, split, augment=False, cfg=cfg)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    model.eval()
    P, Y = [], []
    for b in ld:
        o = model({k: v.to(device) for k, v in b.items()
                   if k in ("image", "crops", "anatomy", "quality_axes", "hardness")})
        P.append(o["ordinal"].class_probs.float().cpu().numpy())
        Y.append(b["grade"].numpy())
    return np.concatenate(P), np.concatenate(Y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=OUT_DIR / "retfound_plus_laft_xai" / "best.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--external-cache", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=24,
                    help="inference batch. 24 needs ~6 GB on a ViT-Large; drop "
                         "to 2-4 when the run is memory-capped.")
    ap.add_argument("--npv-target", type=float, default=None,
                    help="add an extra NPV safety target to the table")
    args = ap.parse_args()
    if args.npv_target and ("npv", args.npv_target) not in TARGETS:
        TARGETS.append(("npv", args.npv_target))
        TARGETS.sort(key=lambda t: (t[0], t[1]))

    cfg = default_config(); cfg.device = args.device
    device = cfg.torch_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    for field in ("backbone", "fusion", "moe", "preproc", "head"):
        if isinstance(saved, dict) and field in saved:
            setattr(cfg, field, saved[field])
    model = build_model(cfg)
    inject_lora(model.backbone.vit, (ck.get("gla_lora") or ck.get("allora"))["ranks"], cfg.backbone)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    print("[infer] validation split (threshold selection)")
    vp, vy = infer(model, args.cache, "val", device, cfg, args.batch_size)
    print("[infer] test split (held-out reporting)")
    tp_, ty = infer(model, args.cache, "test", device, cfg, args.batch_size)
    ext = None
    if args.external_cache and Path(args.external_cache).exists():
        print("[infer] external cohort")
        ext = infer(model, args.external_cache, None, device, cfg, args.batch_size)

    out: dict = {"_meta": {"checkpoint": str(args.ckpt),
                           "n_val": int(len(vy)), "n_test": int(len(ty)),
                           "protocol": "thresholds selected on val, applied to test"}}

    print("\n" + "=" * 100)
    print("SCREENING TRIAGE - thresholds fitted on VALIDATION, reported on held-out TEST")
    print("=" * 100)

    for key, (k, label) in TASKS.items():
        yv = (vy >= k).astype(int)
        sv = vp[:, k:].sum(1)
        yt = (ty >= k).astype(int)
        st = tp_[:, k:].sum(1)
        print(f"\n{label}   test prevalence {yt.mean()*100:.2f}%  (n={len(yt)})")
        print(f"  {'safety target':<22}{'thr':>7}{'sens':>8}{'spec':>8}"
              f"{'PPV':>8}{'NPV':>9}{'refer%':>9}{'cleared%':>10}{'missed':>8}")
        rows = []
        for constraint, target in TARGETS:
            op_val = select_operating_point(yv, sv, key, constraint, target)
            op_test = apply_operating_point(yt, st, op_val.threshold, key,
                                            constraint, target)
            name = f"{constraint} >= {target:g}"
            print(f"  {name:<22}{op_test.threshold:7.3f}{op_test.sensitivity:8.3f}"
                  f"{op_test.specificity:8.3f}{op_test.ppv:8.3f}{op_test.npv:9.4f}"
                  f"{op_test.referral_rate*100:8.1f}%{op_test.workload_reduction*100:9.1f}%"
                  f"{op_test.missed:8d}")
            ci = bootstrap_ci(yt, st, op_val.threshold)
            lo, hi = ci["npv_ci"]
            clo, chi = ci["cleared_ci"]
            print(f"  {'  95% CI (2000 boot)':<22}{'':>7}{'':>8}{'':>8}{'':>8}"
                  f"{lo:9.4f}{'':>9}{chi*100:9.1f}%")
            print(f"  {'':<22}{'':>7}{'':>8}{'':>8}{'':>8}"
                  f"{hi:9.4f}{'':>9}{clo*100:9.1f}%")
            rows.append({"selected_on_val": op_val.as_dict(),
                         "applied_to_test": op_test.as_dict(),
                         "test_bootstrap_95ci": ci})
            if ext is not None:
                ep, ey = ext
                ye = (ey >= k).astype(int)
                se = ep[:, k:].sum(1)
                op_ext = apply_operating_point(ye, se, op_val.threshold, key,
                                               constraint, target)
                rows[-1]["applied_to_external"] = op_ext.as_dict()
        out[key] = rows

    if ext is not None:
        print("\nExternal cohort (APTOS) at the same val-selected thresholds:")
        for key, (k, label) in TASKS.items():
            print(f"  {label}")
            for r in out[key]:
                e = r.get("applied_to_external")
                if e:
                    print(f"    {r['selected_on_val']['constraint']:>11} "
                          f">= {r['selected_on_val']['target']:g}:  "
                          f"sens={e['sensitivity']:.3f} NPV={e['npv']:.4f} "
                          f"refer={e['referral_rate']*100:.1f}% missed={e['missed']}")

    # the deployable configuration: safest target that still removes real work
    deploy = {}
    for key, (k, _label) in TASKS.items():
        best = None
        for r in out[key]:
            t = r["applied_to_test"]
            if t["npv"] >= 0.98 and (best is None or
                                     t["workload_reduction"] > best["workload_reduction"]):
                best = t
        if best:
            deploy[key] = {"threshold": best["threshold"], "npv": best["npv"],
                           "workload_reduction": best["workload_reduction"],
                           "sensitivity": best["sensitivity"]}
    out["_deployable"] = deploy
    path = args.ckpt.parent / "screening.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    screening_rows = [{"task": key, **r} for key, rows in out.items()
                      if not key.startswith("_") for r in rows]
    save_csv_alongside(screening_rows, path)
    if deploy:
        save_csv_alongside(deploy, path, csv_path=path.with_name("screening_deployable.csv"))
    print(f"\n[saved] {path}")
    if deploy:
        print("\nDeployable triage configuration (NPV >= 0.98 on held-out test):")
        for key, v in deploy.items():
            print(f"  {key:<20} thr={v['threshold']:.3f}  NPV={v['npv']:.4f}  "
                  f"clears {v['workload_reduction']*100:.1f}% of patients  "
                  f"(sens {v['sensitivity']:.3f})")


if __name__ == "__main__":
    main()
