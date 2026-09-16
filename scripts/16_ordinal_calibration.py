#!/usr/bin/env python3
"""Step 16 - post-hoc calibration of the CORAL decision cuts.

The canonical CORAL rule decodes with a hard 0.5 on every cumulative:

    yhat = sum_k  1[ P(Y>k) > 0.5 ]

0.5 is a default, not a fitted quantity, and under a long-tailed prior it is
the wrong cut for every k.  This script replaces it with per-threshold cuts
tau_k fitted on the VALIDATION split and then frozen and applied unchanged to
test:

    yhat = sum_k  1[ P(Y>k) > tau_k ]

That is the same protocol scripts/09_screening.py already uses for the referral
threshold - select on val, freeze, report on test - applied to the five-class
decode instead of to the binary one.  It is not test-set tuning and it is not
retraining: the network weights never move, only the decision rule that reads
them, and the rule is chosen without ever seeing the test labels.

Equivalently this is logit adjustment (Menon et al., ICLR 2021): a cut tau_k on
P(Y>k) = sigmoid(z + b_k) is exactly the ladder b_k -> b_k - logit(tau_k), so
the fitted cuts are reported here as an effective ladder alongside the raw
probabilities, and can be compared with the prior-matched ladder on page 6 of
the Objective 1 evidence pack.

    python scripts/16_ordinal_calibration.py --ckpt outputs/vits_objectives/best.pt

Writes  outputs/ordinal_calibration.json
        outputs/ordinal_calibration_probs.npz   (cached logits, for re-fitting)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, OUT_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.eyepacs import CachedEyePACS  # noqa: E402
from dr.metrics import diagnostic_metrics, ordinal_metrics  # noqa: E402
from dr.model import build_model  # noqa: E402


def to_device(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def model_inputs(b: dict) -> dict:
    """Only the keys the model consumes - mirrors scripts/04_evaluate.py."""
    return {k: b[k] for k in ("image", "crops", "anatomy", "quality_axes",
                              "hardness") if k in b}

GRADES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
MINORITY = (1, 2, 3)
WEIGHTS = {"qwk": 0.5, "macro_f1": 0.3, "minority_recall": 0.2}


# --------------------------------------------------------------------------
def cumulative(probs: np.ndarray) -> np.ndarray:
    """P(Y>k) for k = 0..K-2, from the categorical distribution."""
    rc = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]
    return rc[:, 1:]


def decode_at(cum: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return (cum > taus[None, :]).sum(1).astype(np.int64)


def score_of(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict:
    m = {**diagnostic_metrics(y, pred, probs), **ordinal_metrics(y, pred)}
    for g in range(5):
        sel = y == g
        m[f"recall_grade_{g}"] = float((pred[sel] == g).mean()) if sel.any() else float("nan")
    rec = np.nanmean([m[f"recall_grade_{g}"] for g in MINORITY])
    m["minority_recall"] = 0.0 if not np.isfinite(rec) else float(rec)
    m["selection_score"] = float(
        WEIGHTS["qwk"] * m["quadratic_weighted_kappa"]
        + WEIGHTS["macro_f1"] * m["f1_macro"]
        + WEIGHTS["minority_recall"] * m["minority_recall"])
    m["all_grades_predicted"] = bool(all((pred == g).any() for g in range(5)))
    return m


def fit_taus(y: np.ndarray, probs: np.ndarray, n_grid: int = 96,
             passes: int = 6, require_all_grades: bool = True) -> tuple:
    """Coordinate ascent on the repo's own selection score, on validation only.

    Candidates for tau_k are quantiles of the observed P(Y>k), so the search is
    over cuts that actually separate this split's scores rather than over a
    uniform grid most of which lands outside the score range.
    """
    cum = cumulative(probs)
    taus = np.full(cum.shape[1], 0.5)
    best = score_of(y, decode_at(cum, taus), probs)
    trace = [{"pass": 0, "taus": taus.tolist(), **_slim(best)}]

    for p in range(1, passes + 1):
        moved = False
        for k in range(cum.shape[1]):
            qs = np.unique(np.quantile(cum[:, k], np.linspace(0.005, 0.995, n_grid)))
            for cand in qs:
                trial = taus.copy()
                trial[k] = float(cand)
                m = score_of(y, decode_at(cum, trial), probs)
                if require_all_grades and not m["all_grades_predicted"]:
                    continue
                if m["selection_score"] > best["selection_score"] + 1e-9:
                    best, taus, moved = m, trial, True
        trace.append({"pass": p, "taus": taus.tolist(), **_slim(best)})
        if not moved:
            break
    return taus, best, trace


def _slim(m: dict) -> dict:
    return {"selection_score": m["selection_score"],
            "qwk": m["quadratic_weighted_kappa"],
            "f1_macro": m["f1_macro"],
            "minority_recall": m["minority_recall"],
            "accuracy": m["accuracy"],
            "all_grades_predicted": m["all_grades_predicted"]}


def report(name: str, y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict:
    m = score_of(y, pred, probs)
    return {
        "rule": name,
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "f1_macro": m["f1_macro"],
        "quadratic_weighted_kappa": m["quadratic_weighted_kappa"],
        "adjacent_accuracy": m["adjacent_accuracy"],
        "auc_sight_threatening": m["auc_sight_threatening"],
        "auc_referable_dr": m["auc_referable_dr"],
        "per_grade_recall": [m[f"recall_grade_{g}"] for g in range(5)],
        "minority_recall": m["minority_recall"],
        "selection_score": m["selection_score"],
        "all_grades_predicted": m["all_grades_predicted"],
        "n_predicted_as": [int((pred == g).sum()) for g in range(5)],
        "confusion_matrix": [[int(((y == i) & (pred == j)).sum()) for j in range(5)]
                             for i in range(5)],
    }


# --------------------------------------------------------------------------
@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        b = to_device(batch, device)
        o = model(model_inputs(b))
        P.append(o["ordinal"].class_probs.float().cpu().numpy())
        Y.append(b["grade"].cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=OUT_DIR / "vits_objectives" / "best.pt")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-probs", type=Path,
                    default=OUT_DIR / "ordinal_calibration_probs.npz")
    ap.add_argument("--refit-only", action="store_true",
                    help="skip inference and refit from the cached probabilities")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "ordinal_calibration.json")
    args = ap.parse_args()

    if args.refit_only and args.cache_probs.exists():
        z = np.load(args.cache_probs)
        pv, yv, pt, yt = z["pv"], z["yv"], z["pt"], z["yt"]
        thr = z["thresholds"]
        print(f"[cache] reusing {args.cache_probs}")
    else:
        cfg = default_config()
        cfg.device = args.device
        device = cfg.torch_device()
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
        print(f"[setup] device={device}  ckpt epoch={ck.get('epoch')} "
              f"stage={ck.get('stage')}")

        import torch.nn.functional as F
        sd = ck["model"]
        b0 = sd["ordinal.b0"].float().reshape(1)
        thr = torch.cat([b0, b0 - torch.cumsum(F.softplus(sd["ordinal.deltas"].float()), 0)]).numpy()

        out = {}
        for split in ("val", "test"):
            ds = CachedEyePACS(args.cache, split, augment=False, cfg=cfg)
            ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            t0 = time.time()
            p, y = infer(model, ld, device)
            print(f"[infer] {split}: {len(y)} images in {time.time()-t0:.1f}s")
            out[split] = (p, y)
        (pv, yv), (pt, yt) = out["val"], out["test"]
        np.savez_compressed(args.cache_probs, pv=pv, yv=yv, pt=pt, yt=yt, thresholds=thr)
        print(f"[write] {args.cache_probs}")

    cum_v, cum_t = cumulative(pv), cumulative(pt)
    default = np.full(cum_v.shape[1], 0.5)

    print("\n[fit] coordinate ascent on validation only ...")
    taus, val_best, trace = fit_taus(yv, pv)
    print(f"[fit] tau = {np.round(taus, 4).tolist()}")

    eff_ladder = (thr - np.log(taus / (1.0 - taus))).tolist()

    rows = {
        "val": {
            "baseline_coral_0.5": report("CORAL, cut 0.5", yv, decode_at(cum_v, default), pv),
            "argmax": report("argmax", yv, pv.argmax(1), pv),
            "calibrated": report("CORAL, cuts fitted on val", yv, decode_at(cum_v, taus), pv),
        },
        "test": {
            "baseline_coral_0.5": report("CORAL, cut 0.5", yt, decode_at(cum_t, default), pt),
            "argmax": report("argmax", yt, pt.argmax(1), pt),
            "calibrated": report("CORAL, val cuts applied unchanged", yt,
                                 decode_at(cum_t, taus), pt),
        },
    }

    doc = {
        "document": {
            "title": "Post-hoc CORAL cut calibration",
            "generated": time.strftime("%Y-%m-%d"),
            "generator": "scripts/16_ordinal_calibration.py",
            "checkpoint": str(args.ckpt),
            "protocol": "cuts selected on the validation split by coordinate ascent on "
                        "0.5*QWK + 0.3*MacroF1 + 0.2*minority recall, then FROZEN and applied "
                        "unchanged to test. No weight is retrained; the test labels are never "
                        "seen by the fit.",
            "equivalence": "a cut tau_k on P(Y>k) = sigmoid(z + b_k) is the ladder shift "
                           "b_k -> b_k - logit(tau_k) (logit adjustment, Menon et al. 2021).",
        },
        "fitted_cuts": {
            "tau": taus.tolist(),
            "default_cut": 0.5,
            "learned_ladder_b_k": thr.tolist(),
            "effective_ladder_after_calibration": eff_ladder,
            "constraint": "candidate rejected unless all five grades are predicted",
        },
        "n": {"val": int(len(yv)), "test": int(len(yt))},
        "results": rows,
        "fit_trace": trace,
    }
    json.dump(doc, open(args.out, "w"), indent=2)
    results_rows = [{"split": split, "rule": rule, **metrics}
                    for split, rules in rows.items() for rule, metrics in rules.items()]
    save_csv_alongside(results_rows, args.out,
                       csv_path=Path(args.out).with_name(Path(args.out).stem + "_results.csv"))
    print(f"[write] {args.out}")

    hdr = f"{'rule':34s} " + " ".join(f"{g:>8s}" for g in GRADES) + "   acc    QWK   all5"
    for split in ("val", "test"):
        print(f"\n=== {split} (n={len(yv) if split=='val' else len(yt)}) ===")
        print(hdr)
        for key in ("baseline_coral_0.5", "argmax", "calibrated"):
            r = rows[split][key]
            rec = " ".join(f"{x:8.3f}" for x in r["per_grade_recall"])
            print(f"{r['rule'][:34]:34s} {rec}  {r['accuracy']:.3f}  "
                  f"{r['quadratic_weighted_kappa']:.3f}  {r['all_grades_predicted']}")


if __name__ == "__main__":
    main()
