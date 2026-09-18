#!/usr/bin/env python3
"""Step 10 - pretrain the A3 lesion experts on REAL lesion annotations.

    IDRiD / DDR / FGADR (pixel-level masks)
             |
             v
    A3 Lesion MoE  <- supervised Dice + weighted BCE + presence
             |
             v
    transfer the stem + experts into the EyePACS grading model

Why this step exists
--------------------
EyePACS has no lesion masks.  Supervising the experts with the morphological
priors teaches them to reproduce a top-hat filter, including its mistakes: a
vessel crossing looks like a microaneurysm to a top-hat, and so it does to an
expert trained on one.  Real annotations are the only way the MA expert learns
the distinction that separates Mild from No-DR.

Only the stem and the six experts transfer.  The adaptive gate does not: it
conditions on Q* and on hard-example history, neither of which exists in the
segmentation sets, so it is re-learned on EyePACS where those signals are real.

Magnification matters more than it looks.  Feeding a whole 4288x2848 IDRiD image
to a 448 px network puts a 30 px microaneurysm at 3 px and its 112 px evidence
cell at 0.8 px - below one cell, so the MA expert cannot learn and its Dice pins
at zero.  Pretraining therefore runs on *windows at the same field fraction the
DR model's local branch uses*, which is the view the experts will actually be
asked to run on.  `--whole-image` restores the old behaviour for comparison.

Channels a dataset does not annotate are masked out of the loss rather than
treated as negative - see `lesion_datasets` for why that distinction matters.

    python scripts/10_lesion_pretrain.py --epochs 8
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import OUT_DIR, RAW_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.lesion_datasets import (CachedLesionCrops,  # noqa: E402
                                     LesionSegmentationDataset,
                                     build_crop_cache, discover_all,
                                     split_records)
from dr.data.lesion_priors import LESION_NAMES  # noqa: E402
from dr.modules.a3_lesion_moe import LesionMoE, lesion_supervision_loss  # noqa: E402


def to_device(b: dict, dev) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Per-channel Dice against the human masks, on annotated channels only."""
    model.eval()
    inter = np.zeros(len(LESION_NAMES)); denom = np.zeros(len(LESION_NAMES))
    seen = np.zeros(len(LESION_NAMES))
    for batch in loader:
        b = to_device(batch, device)
        out = model(b["image"])
        p = (torch.sigmoid(out.evidence) > 0.5).float()
        t = b["masks"]
        if t.shape[-2:] != p.shape[-2:]:
            t = torch.nn.functional.interpolate(t, size=p.shape[-2:],
                                                mode="nearest")
        t = (t > 0.5).float()
        v = b["valid"]
        inter += ((p * t).sum((2, 3)) * v).sum(0).cpu().numpy()
        denom += ((p.sum((2, 3)) + t.sum((2, 3))) * v).sum(0).cpu().numpy()
        seen += v.sum(0).cpu().numpy()
    dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1e-6), np.nan)
    out = {f"dice_{n}": float(d) for n, d in zip(LESION_NAMES, dice)}
    out["dice_mean"] = float(np.nanmean(dice[seen > 0])) if (seen > 0).any() else float("nan")
    out["_annotated_per_channel"] = seen.astype(int).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=RAW_DIR / "lesion")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "lesion_pretrain")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=None,
                    help="network input size (crop_input when cropping)")
    ap.add_argument("--whole-image", action="store_true",
                    help="train on downsampled whole images instead of "
                         "magnification-matched windows (small lesions vanish)")
    ap.add_argument("--crops-per-image", type=int, default=6)
    ap.add_argument("--negative-frac", type=float, default=0.3,
                    help="share of windows drawn at random rather than on a lesion")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--extract-workers", type=int, default=6)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = default_config()
    cfg.device = args.device
    epochs = args.epochs or cfg.train.lesion_pretrain_epochs
    lr = args.lr or cfg.train.lesion_pretrain_lr
    image_size = args.image_size or cfg.preproc.global_size
    device = cfg.torch_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    records, summary = discover_all(args.root)
    print(f"[data] scanning {args.root}")
    for name, s in summary.items():
        print(f"       {name:6s}: {s['images']:5d} annotated images  "
              f"{s['masks_per_channel']}")
    if not records:
        sys.exit(
            f"No lesion annotations found under {args.root}.\n"
            "Expected data/raw/lesion/{idrid,ddr,fgadr}/... with one directory "
            "per lesion class.\nIDRiD and DDR segmentation are on Kaggle; FGADR "
            "requires a signed request form.\nWithout them the A3 experts fall "
            "back to morphological pseudo-labels - run 03_train.py without "
            "--lesion-encoder.")

    train_recs, val_recs = split_records(records, seed=cfg.train.seed)
    cache_root = Path(args.out) / "crop_cache"

    if args.whole_image:
        train_ds = LesionSegmentationDataset(train_recs, image_size,
                                             cfg.moe.mask_size, augment=True)
        val_ds = LesionSegmentationDataset(val_recs, image_size, cfg.moe.mask_size,
                                           augment=False)
        print("[data] WHOLE-IMAGE mode: small lesions are sub-cell in the "
              "evidence map, so MA Dice will stay near zero")
    else:
        # the field fraction the DR model's local branch sees
        ff = cfg.preproc.crop_size / max(cfg.preproc.cache_size, 1)
        print(f"[data] window mode: {ff:.1%} of the retinal field per crop, fed at "
              f"{image_size}px - the same magnification as the DR local branch")
        for name, recs in (("train", train_recs), ("val", val_recs)):
            d = cache_root / name
            if (d / "crops_x.npy").exists() and not args.rebuild_cache:
                print(f"[crop-cache] reusing {d}")
                continue
            print(f"[crop-cache] extracting {len(recs) * args.crops_per_image} "
                  f"{name} windows from {len(recs)} annotated images")
            info = build_crop_cache(
                recs, d, field_fraction=ff, crop_input=image_size,
                mask_size=cfg.moe.mask_size, crops_per_image=args.crops_per_image,
                negative_frac=args.negative_frac, workers=args.extract_workers,
                seed=cfg.train.seed)
            print(f"[crop-cache] {info}")
        train_ds = CachedLesionCrops(cache_root / "train", augment=True,
                                     seed=cfg.train.seed)
        val_ds = CachedLesionCrops(cache_root / "val", augment=False,
                                   seed=cfg.train.seed + 1)

    print(f"[data] {len(train_ds)} train / {len(val_ds)} val crops "
          f"from {len(train_recs)}/{len(val_recs)} annotated images")
    print(f"[data] channel coverage (train): {train_ds.channel_coverage()}")
    unann = [c for c, n in train_ds.channel_coverage().items() if n == 0]
    if unann:
        print(f"[data] NOTE: {unann} have no annotations in these sets and are "
              "masked out of the loss; those experts stay at their init and are "
              "learned from the weak priors during DR training.")
    if hasattr(train_ds, "lesion_prevalence"):
        # a channel whose target is almost always empty cannot produce a
        # meaningful Dice - printing this turns "Dice 0.00" into a fact.
        # `lesion_prevalence` is a @property (already a dict keyed by the
        # ANNOTATED_CHANNELS present in these crops, not all of LESION_NAMES -
        # NV/ME are never annotated by any of IDRiD/DDR/FGADR), so use its
        # items directly rather than calling it or zipping against LESION_NAMES.
        prev = train_ds.lesion_prevalence
        print("[data] lesion pixel fraction in the evidence grid: "
              + " ".join(f"{n}={v:.4f}" for n, v in prev.items()))

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=len(train_ds) > args.batch_size,
                          pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))

    model = LesionMoE(cfg.moe).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] LesionMoE {n_tr:,} trainable parameters, "
          f"evidence maps at {cfg.moe.mask_size}px")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * max(len(train_ld), 1)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    start_epoch = 0

    # resume: this is a short job, but the machine running it may not stay up
    # for even that long, so it checkpoints the same way scripts/03_train.py
    # does - re-running the exact same command continues after the last
    # completed epoch instead of retraining from scratch.
    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists():
        resume = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        history = resume["history"]
        best = resume["best"]
        start_epoch = resume["epoch"] + 1
        torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
        np.random.set_state(resume["numpy_rng_state"])
        print(f"[resume] found {ckpt_path}, continuing at epoch "
              f"{start_epoch+1}/{epochs} (best Dice so far {best:.4f})")
    t0 = time.time()

    for ep in range(start_epoch, epochs):
        model.train()
        tot, n = 0.0, 0
        te = time.time()
        for step, batch in enumerate(train_ld):
            b = to_device(batch, device)
            out = model(b["image"])
            # CachedLesionCrops/LesionSegmentationDataset yield "masks"/"valid"
            # (see dr.data.lesion_datasets) - there is no separate "presence"
            # label, it's the image-level fact implied by the pixel masks: a
            # channel is present in a crop iff its mask has any positive pixel.
            masks = b["masks"]
            presence = (masks.sum(dim=(-2, -1)) > 0).float()
            loss = lesion_supervision_loss(out, masks, presence, valid=b["valid"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            bs = b["image"].shape[0]; tot += float(loss) * bs; n += bs
            if step % 20 == 0:
                print(f"  e{ep+1} step {step}/{len(train_ld)} "
                      f"loss={tot/max(n,1):.4f} {n/max(time.time()-te,1e-6):.1f} img/s",
                      flush=True)

        vm = evaluate(model, val_ld, device)
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1),
               "epoch_sec": round(time.time() - te, 1), **vm}
        history.append(rec)
        per = {k.replace("dice_", ""): round(v, 4) for k, v in vm.items()
               if k.startswith("dice_") and k != "dice_mean" and np.isfinite(v)}
        print(f"[epoch {ep+1}/{epochs}] loss={rec['train_loss']:.4f} "
              f"val Dice mean={vm['dice_mean']:.4f} {per} ({rec['epoch_sec']}s)",
              flush=True)

        if np.isfinite(vm["dice_mean"]) and vm["dice_mean"] > best:
            best = vm["dice_mean"]
            torch.save({"moe": model.state_dict(), "best_dice": best,
                        "epoch": ep + 1, "sources": summary,
                        "channel_coverage": train_ds.channel_coverage(),
                        "image_size": image_size, "mask_size": cfg.moe.mask_size,
                        "mode": "whole" if args.whole_image else "window",
                        "field_fraction": (None if args.whole_image else
                                           cfg.preproc.crop_size / cfg.preproc.cache_size)},
                       out_dir / "lesion_encoder.pt")
            print(f"          saved lesion encoder (mean Dice {best:.4f})")

        ckpt_payload = {
            "epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "history": history, "best": best,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (torch.cuda.get_rng_state_all()
                                   if torch.cuda.is_available() else None),
            "numpy_rng_state": np.random.get_state(),
        }
        tmp_ckpt = out_dir / "checkpoint.pt.tmp"
        torch.save(ckpt_payload, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_path)

    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    save_csv_alongside(history, history_path)
    sources_path = out_dir / "sources.json"
    sources_path.write_text(json.dumps(summary, indent=2))
    save_csv_alongside(summary, sources_path)
    print(f"\n[done] best val Dice {best:.4f} vs REAL annotations "
          f"| {(time.time()-t0)/60:.1f} min | {out_dir/'lesion_encoder.pt'}")
    print("[next] scripts/03_train.py --lesion-encoder "
          f"{out_dir/'lesion_encoder.pt'}")


if __name__ == "__main__":
    main()
