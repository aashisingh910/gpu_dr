#!/usr/bin/env python3
"""Step 02 - run A1 (quality gate) + A2 (lesion-preserving preprocessing) +
the weak lesion/anatomy priors over the real EyePACS images, once, into a
memory-mapped cache.

These stages are pure OpenCV on full-resolution pixels and cost ~60-100 ms per
image; doing them inside the training loop would dominate every epoch.  Running
them once here means training reads uint8 memmaps at full speed.

The cache now stores the retinal field at `--image-size` (1024 by default, the
"1024x1024 global image" of the architecture) together with, per image, the
centres of `--n-crops` local lesion windows.  Those centres are computed here
because they depend on the lesion priors and the disc/macula landmarks, all of
which are already being derived at this step - recomputing them per epoch in
the dataloader would cost more than the pixels do.

    python scripts/02_preprocess.py --root data/raw/eyepacs-hi --workers 8
    python scripts/02_preprocess.py --root data/raw/eyepacs-hi --limit 8000
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import CACHE_DIR, RAW_DIR, default_config  # noqa: E402
from dr.data.eyepacs import (build_manifest, split_by_patient,  # noqa: E402
                             subsample_stratified)
from dr.data.lesion_priors import (LesionPriorExtractor,  # noqa: E402
                                   generate_lesion_crops)
from dr.modules import a1_quality, a2_preprocess  # noqa: E402

cv2.setNumThreads(1)   # each worker stays single-threaded; the pool parallelises

_CFG = None
_EXTRACTOR = None


def _init_worker(image_size: int, mask_size: int, n_crops: int, crop_size: int,
                 qref: dict, raw_pixels: bool = False):
    global _CFG, _EXTRACTOR
    _CFG = default_config()
    _CFG.preproc.cache_size = image_size
    _CFG.preproc.n_crops = n_crops
    _CFG.preproc.crop_size = crop_size
    _CFG.preproc.raw_pixels = raw_pixels
    _CFG.moe.mask_size = mask_size
    # the Q* references are corpus-dependent (see a1_quality.calibrate_...)
    for key in ("lesion_floor_cnr", "lesion_ref_cnr", "blur_ref"):
        if key in qref:
            setattr(_CFG.quality, key, qref[key])
    if "domain_ref" in qref:
        _CFG.quality.domain_ref = qref["domain_ref"]
    _EXTRACTOR = LesionPriorExtractor(out_size=mask_size)


def _process_one(args: tuple[int, str]):
    idx, path = args
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return ("unreadable", idx, path)
    try:
        report = a1_quality.assess(bgr, _CFG.quality)               # A1
        pre = a2_preprocess.run(bgr, report, _CFG.preproc)          # A2
        priors = _EXTRACTOR(pre.image, pre.fov_mask)                # weak labels
    except Exception as e:                                          # noqa: BLE001
        return ("error", idx, f"{type(e).__name__}: {e}")

    rgb = cv2.cvtColor(pre.image, cv2.COLOR_BGR2RGB)
    lesions = (np.clip(priors.masks, 0, 1) * 255).astype(np.uint8)
    anatomy = (np.clip(priors.anatomy, 0, 1) * 255).astype(np.uint8)
    crops = generate_lesion_crops(
        priors.masks, priors.anatomy, pre.fov_mask,
        n_crops=_CFG.preproc.n_crops,
        crop_frac=_CFG.preproc.crop_size / max(_CFG.preproc.cache_size, 1),
        disc_center=priors.disc_center, macula_center=priors.macula_center)
    meta = {
        "index": idx,
        "quality_score": round(report.score, 4),
        "legacy_quality_score": round(report.legacy_score, 4),
        "q_quality": round(report.q_quality, 4),
        "q_domain": round(report.q_domain, 4),
        "q_lesion": round(report.q_lesion, 4),
        "q_blur": round(report.q_blur, 4),
        "q_illumination": round(report.q_illumination, 4),
        "q_gradability": round(report.gradability, 4),
        "q_artifact": round(report.artifact, 4),
        "q_fov": round(report.fov, 4),
        "quality_decision": report.decision,
        "domain": report.domain,
        "disc_x": priors.disc_center[0], "disc_y": priors.disc_center[1],
        "macula_x": priors.macula_center[0], "macula_y": priors.macula_center[1],
    }
    for name, burden in zip(("MA", "HE", "EX_H", "EX_S", "NV", "ME"), priors.burden()):
        meta[f"burden_{name}"] = round(float(burden), 6)
    return ("ok", idx, rgb, lesions, anatomy, crops, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=RAW_DIR / "eyepacs-224")
    ap.add_argument("--out", type=Path, default=CACHE_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="stratified subsample of N images (default: all)")
    ap.add_argument("--image-size", type=int, default=1024,
                    help="retinal field size written to the cache")
    ap.add_argument("--mask-size", type=int, default=112)
    ap.add_argument("--n-crops", type=int, default=6)
    ap.add_argument("--crop-size", type=int, default=320,
                    help="local window size, in cache pixels")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--calibration-sample", type=int, default=400,
                    help="images used to anchor the Q* references to this corpus")
    ap.add_argument("--quality-reference", type=Path, default=None,
                    help="reuse a quality_reference.json instead of recalibrating "
                         "(required when caching an external/domain-shift set, so "
                         "its Q_domain is measured against the *training* domain)")
    ap.add_argument("--raw-pixels", action="store_true",
                    help="Objective 3 ablation: cache field-extraction only - no "
                         "illumination normalisation, no CLAHE, no lesion-boost "
                         "fusion. Build a second cache with this flag (e.g. into "
                         "--out data/cache_raw) to re-run the linear probe "
                         "(scripts/13_linear_probe.py --cache data/cache_raw) and "
                         "test whether A1/A2 preprocessing, tuned for lesion "
                         "contrast rather than for RETFound's own pretraining "
                         "statistics, is what suppresses RETFound's frozen-"
                         "feature advantage over generic ImageNet backbones.")
    args = ap.parse_args()

    print(f"[manifest] scanning {args.root}")
    df = build_manifest(args.root)
    print(f"[manifest] {len(df)} labelled images, "
          f"{df.patient_id.nunique()} patients")
    print("[manifest] grade distribution:")
    for g, c in df.grade.value_counts().sort_index().items():
        print(f"           grade {g}: {c:6d}  ({100*c/len(df):5.2f}%)")

    if args.limit:
        df = subsample_stratified(df, args.limit, args.seed)
        print(f"[manifest] subsampled to {len(df)} images "
              f"({df.patient_id.nunique()} patients)")

    df = split_by_patient(df, seed=args.seed).reset_index(drop=True)
    print("[split] " + ", ".join(f"{k}={v}" for k, v in
                                 df.split.value_counts().items()))
    # guard against the classic leak
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        overlap = set(df[df.split == a].patient_id) & set(df[df.split == b].patient_id)
        assert not overlap, f"patient leak between {a} and {b}: {len(overlap)}"
    print("[split] verified: no patient appears in two splits")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- anchor the A1 Q* axes to this corpus ---------------------------
    if args.quality_reference:
        qref = json.loads(Path(args.quality_reference).read_text())
        print(f"[A1 calib] reusing references from {args.quality_reference}")
    else:
        print(f"[A1 calib] sampling {args.calibration_sample} images to anchor "
              "Q_lesion / Q_blur / Q_domain")
        qref = a1_quality.calibrate_quality_reference(
            df["path"].tolist(), sample=args.calibration_sample, seed=args.seed)
        (out / "quality_reference.json").write_text(json.dumps(qref, indent=2))
    if qref:
        print(f"[A1 calib] lesion CNR floor={qref['lesion_floor_cnr']:.2f} "
              f"ref={qref['lesion_ref_cnr']:.2f} | blur_ref={qref['blur_ref']:.0f} "
              f"| n={qref['n_sampled']}")
        print(f"[A1 calib] domain reference: "
              + ", ".join(f"{k}={v:.3f}" for k, v in qref["domain_ref"].items()))

    n, S, M, K = len(df), args.image_size, args.mask_size, args.n_crops
    shapes = {"images.npy": (n, S, S, 3, np.uint8), "lesions.npy": (n, 6, M, M, np.uint8),
              "anatomy.npy": (n, 5, M, M, np.uint8), "crops.npy": (n, K, 2, None, np.float32)}
    progress_path = out / "preprocess_progress.csv"
    run_args_path = out / "preprocess_run_args.json"
    this_args = {"root": str(args.root), "limit": args.limit, "image_size": args.image_size,
                 "mask_size": args.mask_size, "n_crops": args.n_crops,
                 "crop_size": args.crop_size, "raw_pixels": args.raw_pixels}

    # --- resume: reuse existing cache files + progress log instead of wiping
    # them, so a run interrupted (crash, reboot, shutdown) part-way through
    # only has to redo the images it hadn't finished yet.  This is keyed on
    # the cache files existing with the exact shape this invocation implies -
    # anything else (first run, or a genuinely different --image-size/--limit)
    # falls through to the normal fresh-start path below.
    existing = all((out / name).exists() for name in shapes)
    metas: dict[int, dict] = {}
    resuming = False
    if existing and progress_path.exists():
        try:
            probe = np.load(out / "images.npy", mmap_mode="r")
            if probe.shape == (n, S, S, 3):
                resuming = True
            probe = None
        except Exception:                                       # noqa: BLE001
            resuming = False
    if resuming and run_args_path.exists():
        prev_args = json.loads(run_args_path.read_text())
        if prev_args != this_args:
            print(f"[resume] WARNING: this run's args differ from the ones that "
                  f"built the existing cache ({prev_args} vs {this_args}) - "
                  f"resuming anyway since the array shapes match, but this may "
                  f"mix inconsistent preprocessing settings across rows.")

    if resuming:
        with open(progress_path, newline="") as f:
            for row in csv.DictReader(f):
                idx = int(row["index"])
                for k2, v in row.items():
                    if k2 != "index" and k2 not in ("quality_decision", "domain"):
                        row[k2] = float(v)
                row["index"] = idx
                metas[idx] = row
        print(f"[resume] {len(metas)}/{n} images already cached in {out} - "
              f"continuing with the remainder")
        mode = "r+"
    else:
        for stale in (progress_path, run_args_path, out / "preprocess_failed.json"):
            stale.unlink(missing_ok=True)
        need_gb = n * (S * S * 3 + 11 * M * M) / 1e9
        free_gb = shutil.disk_usage(out.parent).free / 1e9
        print(f"[cache] {n} images at {S}px needs ~{need_gb:.1f} GB; "
              f"{free_gb:.1f} GB free")
        if need_gb > free_gb * 0.92:
            sys.exit(f"Refusing to start: the cache needs ~{need_gb:.1f} GB but only "
                     f"{free_gb:.1f} GB is free. Lower --image-size or --limit.")
        mode = "w+"
    run_args_path.write_text(json.dumps(this_args, indent=2))

    images = np.lib.format.open_memmap(out / "images.npy", mode=mode,
                                       dtype=np.uint8, shape=(n, S, S, 3))
    lesions = np.lib.format.open_memmap(out / "lesions.npy", mode=mode,
                                        dtype=np.uint8, shape=(n, 6, M, M))
    anatomy = np.lib.format.open_memmap(out / "anatomy.npy", mode=mode,
                                        dtype=np.uint8, shape=(n, 5, M, M))
    crops = np.lib.format.open_memmap(out / "crops.npy", mode=mode,
                                      dtype=np.float32, shape=(n, K, 2))

    fieldnames = ["index", "quality_score", "legacy_quality_score", "q_quality",
                  "q_domain", "q_lesion", "q_blur", "q_illumination", "q_gradability",
                  "q_artifact", "q_fov", "quality_decision", "domain", "disc_x", "disc_y",
                  "macula_x", "macula_y", "burden_MA", "burden_HE", "burden_EX_H",
                  "burden_EX_S", "burden_NV", "burden_ME"]
    progress_f = open(progress_path, "a" if resuming else "w", newline="")
    progress_w = csv.DictWriter(progress_f, fieldnames=fieldnames)
    if not resuming:
        progress_w.writeheader()
        progress_f.flush()

    # Permanently-unreadable/errored images must not be retried forever (that
    # would make the resumable version never converge) but also must not be
    # silently invisible to the resume logic (that would just retry them
    # forever anyway) - persist them separately from the successfully-cached
    # `metas` so `done_indices` covers "processed OR permanently given up on".
    failed_path = out / "preprocess_failed.json"
    failed: dict[int, str] = {}
    if resuming and failed_path.exists():
        failed = {int(k): v for k, v in json.loads(failed_path.read_text()).items()}
        print(f"[resume] {len(failed)} previously-failed images will stay excluded "
              f"(not retried) - see {failed_path}")

    done_indices = set(metas) | set(failed)
    tasks = [(idx, path) for idx, path in enumerate(df["path"].tolist())
             if idx not in done_indices]
    errors: list[str] = []
    t0 = time.time()
    n_remaining = len(tasks)
    print(f"[preproc] {n_remaining}/{n} images left to process")
    if args.raw_pixels:
        print("[A2] --raw-pixels: caching field-extraction only - no "
              "illumination normalisation, no CLAHE, no lesion-boost fusion "
              "(Objective 3 ablation)")
    if n_remaining:
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.image_size, args.mask_size, args.n_crops,
                               args.crop_size, qref, args.raw_pixels)) as pool:
            for k, res in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=16)):
                if res[0] in ("error", "unreadable"):
                    idx = res[1]
                    reason = res[2] if res[0] == "error" else f"unreadable: {res[2]}"
                    errors.append(f"idx {idx}: {reason}")
                    failed[idx] = reason
                    continue
                _, idx, rgb, les, ana, crp, meta = res
                images[idx] = rgb
                lesions[idx] = les
                anatomy[idx] = ana
                crops[idx] = crp
                metas[idx] = meta
                progress_w.writerow(meta)
                # A 1024px cache is tens of GB of dirty memmap pages.  On a
                # machine with less RAM than the cache, letting them
                # accumulate pushes the box into swap and throughput
                # collapses to near zero - flushing on a fixed cadence bounds
                # the dirty set instead, and also bounds how much work an
                # unplanned shutdown (this machine's normal failure mode, not
                # an edge case) can throw away: at most ~200 images'-worth,
                # never the whole run.
                if k % 200 == 199:
                    images.flush(); lesions.flush(); anatomy.flush(); crops.flush()
                    progress_f.flush(); os.fsync(progress_f.fileno())
                    failed_path.write_text(json.dumps(failed))
                if k % 500 == 0:
                    el = time.time() - t0
                    rate = (k + 1) / max(el, 1e-6)
                    done_now = len(done_indices) + k + 1
                    print(f"[preproc] {done_now}/{n}  {rate:.1f} img/s (this run)  "
                          f"eta {(n_remaining-k-1)/max(rate,1e-6)/60:.1f} min", flush=True)

    images.flush(); lesions.flush(); anatomy.flush(); crops.flush()
    progress_f.flush(); progress_f.close()
    failed_path.write_text(json.dumps(failed))
    ok = sorted(metas)
    print(f"[preproc] completed {len(ok)}/{n} total "
          f"({n_remaining} processed this run in {(time.time()-t0)/60:.1f} min)")
    if failed:
        sample = list(failed.items())[:3]
        print(f"[preproc] {len(failed)} permanently unreadable/errored images "
              f"excluded from the dataset, e.g. {sample}")
    if len(ok) + len(failed) < n:
        missing = n - len(ok) - len(failed)
        sys.exit(f"[preproc] stopped with {missing}/{n} images not yet processed "
                 f"(interrupted run) - re-run this exact command to resume; "
                 f"already-done images will be skipped.")

    meta_df = pd.DataFrame([metas[i] for i in ok]).set_index("index")
    full = df.join(meta_df, how="inner")
    full["cache_row"] = full.index
    full.to_csv(out / "meta.csv.tmp", index=False)
    os.replace(out / "meta.csv.tmp", out / "meta.csv")

    print("\n[A1 gate] routing decisions across the real dataset:")
    for k, v in full.quality_decision.value_counts().items():
        print(f"          {k:8s}: {v:6d} ({100*v/len(full):5.2f}%)")
    print(f"[A1 gate] mean Q* = {full.quality_score.mean():.3f} "
          f"(legacy single-axis Q = {full.legacy_quality_score.mean():.3f})")
    print("[A1 gate] mean of each Q* axis:")
    for c in ("q_quality", "q_domain", "q_lesion", "q_blur", "q_illumination"):
        print(f"          {c:16s}: {full[c].mean():.3f}")
    print("[A1 gate] mean lesion-visibility Q_lesion by DR grade "
          "(should rise with severity if the axis is meaningful):")
    print(full.groupby("grade")["q_lesion"].mean().round(4).to_string())
    print(f"[A1 gate] {full.domain.nunique()} camera/domain fingerprints; top:")
    for k, v in full.domain.value_counts().head(4).items():
        print(f"          {k:24s}: {v:6d}")
    print("\n[priors] mean lesion burden by DR grade (weak morphological labels):")
    cols = [c for c in full.columns if c.startswith("burden_")]
    print(full.groupby("grade")[cols].mean().round(5).to_string())
    print(f"\n[cache] written to {out}")


if __name__ == "__main__":
    main()
