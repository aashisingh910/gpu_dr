#!/usr/bin/env python3
"""Step 12 - does the multistage preprocessing actually make lesions visible?

Objective 2 claims the pipeline "enhances retinal image quality and lesion
visibility".  Quality is measured elsewhere (A1 Q*); *visibility* never was.
This script measures it directly, against real IDRiD / DDR lesion annotations,
using contrast-to-noise ratio on the green channel:

    CNR = |mean(lesion px) - mean(local background px)| / std(local background px)

"Local background" is an annular ring around each lesion, so the number is a
local detectability measure rather than a global contrast statistic.

Two independent claims are tested separately:

  RESOLUTION  a 320 px window cut at native cache resolution and fed at 224 px
              samples a lesion at 224/320 = 0.70 output px per cache px, against
              224/1024 = 0.22 for a whole-image 224 px view - 3.2x linear.  Does
              that survive as measured CNR?

  ENHANCEMENT illumination normalisation + CLAHE, the multistage step itself.

    python scripts/12_lesion_visibility.py --limit 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dr.config import OUT_DIR, RAW_DIR, default_config  # noqa: E402
from dr.csv_export import save_csv_alongside  # noqa: E402
from dr.data.lesion_datasets import discover_all  # noqa: E402
from dr.modules.a2_preprocess import (adaptive_clahe, normalise_illumination,  # noqa: E402
                                      local_contrast_normalize, protect_large_lesions)

CACHE_PX, CROP_PX, CROP_IN, GLOBAL_PX = 1024, 320, 224, 224


def fov_mask(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (g > 12).astype(np.uint8)


def cnr(green: np.ndarray, mask: np.ndarray, ring: int = 6) -> float | None:
    """CNR of the masked lesion against an annular local background."""
    m = mask > 0
    if m.sum() < 4:
        return None
    k = np.ones((ring * 2 + 1, ring * 2 + 1), np.uint8)
    bg = (cv2.dilate(mask, k) > 0) & (~m)
    if bg.sum() < 8:
        return None
    lv, bv = green[m].astype(np.float64), green[bg].astype(np.float64)
    sd = bv.std()
    if sd < 1e-6:
        return None
    return float(abs(lv.mean() - bv.mean()) / sd)


def enhance(bgr: np.ndarray, cfg) -> np.ndarray:
    """The multistage step: illumination normalisation then local enhancement.

    Objective 2 fix: soft exudates were the one lesion class CLAHE made
    measurably worse (paired median CNR ratio 0.945x). Two mitigations were
    tried; only one survived benchmarking:
      - `clahe_protect_large_lesions` (default on): blends CLAHE's output
        back toward native pixels in large, already-bright regions.
        VALIDATED - restores blob contrast without touching CLAHE's (fast)
        global mechanism.
      - `enhancement_algorithm='local_contrast'`: replacing CLAHE outright
        with a tile-free division-normalisation. REJECTED after
        benchmarking - measured 3-17x slower than CLAHE and, at any sigma
        large enough to matter, WORSE soft-exudate contrast than doing
        nothing (see `local_contrast_normalize`'s docstring for the
        numbers). Kept only as an explicit `--algo local_contrast` option
        so the comparison is reproducible; 'clahe' is the default.
    """
    m = fov_mask(bgr)
    out = normalise_illumination(bgr, m, cfg.illum_sigma)
    algo = getattr(cfg, "enhancement_algorithm", "clahe")
    if algo == "clahe":
        clahe = adaptive_clahe(out, cfg.clahe_clip, cfg.clahe_grid)
    else:
        clahe = local_contrast_normalize(out, cfg.local_contrast_sigma,
                                         cfg.local_contrast_strength)
    if getattr(cfg, "clahe_protect_large_lesions", False):
        clahe = protect_large_lesions(out, clahe, cfg.large_lesion_kernel,
                                      cfg.large_lesion_protect_strength)
    return clahe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion-root", type=Path, default=RAW_DIR / "lesion")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--algo", choices=("clahe", "local_contrast"), default=None,
                    help="override PreprocCfg.enhancement_algorithm for this "
                         "run. Default (clahe) is the validated choice; "
                         "local_contrast is a rejected alternative kept only "
                         "so the benchmark that rejected it is reproducible "
                         "(see local_contrast_normalize's docstring).")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "lesion_visibility.json")
    args = ap.parse_args()

    cfg = default_config().preproc
    if args.algo:
        cfg.enhancement_algorithm = args.algo
    records, found = discover_all(args.lesion_root)
    if not records:
        print(f"no annotated records under {args.lesion_root}"); sys.exit(1)
    rng = np.random.default_rng(1337)
    if args.limit and args.limit < len(records):
        records = [records[i] for i in rng.choice(len(records), args.limit, replace=False)]
    print(f"[data] {len(records)} annotated images from {list(found)}")

    # channel -> condition -> list of CNR
    acc: dict[str, dict[str, list]] = {}

    for n, rec in enumerate(records):
        bgr = cv2.imread(str(rec.image))
        if bgr is None:
            continue
        field = cv2.resize(bgr, (CACHE_PX, CACHE_PX), interpolation=cv2.INTER_AREA)
        field_en = enhance(field, cfg)
        # whole-image view the legacy pipeline feeds the network
        glob = cv2.resize(field, (GLOBAL_PX, GLOBAL_PX), interpolation=cv2.INTER_AREA)
        glob_en = cv2.resize(field_en, (GLOBAL_PX, GLOBAL_PX), interpolation=cv2.INTER_AREA)

        for ch, mpath in rec.masks.items():
            mk = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                continue
            mk = (cv2.resize(mk, (CACHE_PX, CACHE_PX), interpolation=cv2.INTER_NEAREST) > 0
                  ).astype(np.uint8)
            if mk.sum() < 4:
                continue
            d = acc.setdefault(ch, {k: [] for k in
                                    ("global224", "global224_enh", "crop224", "crop224_enh")})

            # --- whole-image 224 view (0.22 output px per cache px) ---
            mk_g = (cv2.resize(mk, (GLOBAL_PX, GLOBAL_PX), interpolation=cv2.INTER_NEAREST) > 0
                    ).astype(np.uint8)
            for key, img in (("global224", glob), ("global224_enh", glob_en)):
                v = cnr(img[:, :, 1], mk_g, ring=2)
                if v is not None:
                    d[key].append(v)

            # --- 320 px native-resolution window fed at 224 (0.70 px per cache px) ---
            ys, xs = np.nonzero(mk)
            cy, cx = int(ys.mean()), int(xs.mean())
            h = CROP_PX // 2
            cy = int(np.clip(cy, h, CACHE_PX - h)); cx = int(np.clip(cx, h, CACHE_PX - h))
            sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
            mk_c = cv2.resize(mk[sl], (CROP_IN, CROP_IN), interpolation=cv2.INTER_NEAREST)
            for key, src in (("crop224", field), ("crop224_enh", field_en)):
                win = cv2.resize(src[sl], (CROP_IN, CROP_IN), interpolation=cv2.INTER_AREA)
                v = cnr(win[:, :, 1], mk_c.astype(np.uint8), ring=6)
                if v is not None:
                    d[key].append(v)
        if (n + 1) % 15 == 0:
            print(f"  {n+1}/{len(records)} images", flush=True)

    # ---- paired statistics ------------------------------------------------
    def paired(a: list, b: list) -> dict:
        """Bootstrap CI on the paired median ratio b/a (same lesions, both views)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ok = a > 1e-6
        if ok.sum() < 5:
            return {"median_ratio": float("nan"), "ci": [float("nan")] * 2,
                    "n_pairs": int(ok.sum()), "wins": float("nan")}
        r = b[ok] / a[ok]
        rng2 = np.random.default_rng(7)
        boots = [float(np.median(rng2.choice(r, len(r)))) for _ in range(2000)]
        return {"median_ratio": float(np.median(r)),
                "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                "n_pairs": int(ok.sum()),
                "wins": float((b[ok] > a[ok]).mean())}

    # ---- report ----------------------------------------------------------
    LABEL = {"MA": "microaneurysm", "HE": "haemorrhage",
             "EX_H": "hard exudate", "EX_S": "soft exudate"}
    print(f"\n{'='*86}")
    print("LESION VISIBILITY - green-channel CNR against annotated ground truth")
    print(f"{'='*86}")
    print(f"{'lesion':<16}{'n':>6}{'whole-224':>12}{'+enhance':>11}"
          f"{'crop-224':>11}{'+enhance':>11}{'resolution':>13}{'enhance':>10}")
    out = {}
    for ch, d in sorted(acc.items()):
        if not d["global224"] or not d["crop224"]:
            continue
        mean = {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()}
        res_gain = mean["crop224"] / mean["global224"] if mean["global224"] else float("nan")
        enh_gain = mean["crop224_enh"] / mean["crop224"] if mean["crop224"] else float("nan")
        print(f"{LABEL.get(ch, ch):<16}{len(d['global224']):>6}"
              f"{mean['global224']:>12.3f}{mean['global224_enh']:>11.3f}"
              f"{mean['crop224']:>11.3f}{mean['crop224_enh']:>11.3f}"
              f"{res_gain:>12.2f}x{enh_gain:>9.2f}x")
        pr = paired(d["global224"], d["crop224"])           # resolution effect
        pe = paired(d["crop224"], d["crop224_enh"])          # enhancement effect
        # CNR < 1 means the lesion sits inside the noise of its own background
        sub = {k: float(np.mean(np.asarray(v) < 1.0)) if v else float("nan")
               for k, v in d.items()}
        out[ch] = {"n": len(d["global224"]), **mean,
                   "resolution_gain": res_gain, "enhancement_gain": enh_gain,
                   "resolution_paired": pr, "enhancement_paired": pe,
                   "frac_below_noise": sub}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"_enhancement_algorithm": cfg.enhancement_algorithm,
         "_clahe_protect_large_lesions": cfg.clahe_protect_large_lesions,
         **out}, indent=2))
    save_csv_alongside(out, args.out)
    print(f"\n[saved] {args.out}  (enhancement_algorithm={cfg.enhancement_algorithm})")
    print("\nresolution = crop-224 / whole-224 (does the native-resolution window help?)")
    print("enhance    = crop-224+enhance / crop-224 (does the multistage step help?)")

    print(f"\n{'='*86}")
    print("PAIRED EFFECT (same lesion, both views) + DETECTABILITY")
    print(f"{'='*86}")
    print(f"{'lesion':<16}{'resolution x':>16}{'95% CI':>18}{'win rate':>10}"
          f"{'  below noise (CNR<1)':>26}")
    for ch, o in sorted(out.items()):
        pr = o["resolution_paired"]; sub = o["frac_below_noise"]
        print(f"{LABEL.get(ch, ch):<16}{pr['median_ratio']:>15.2f}x"
              f"{('[%.2f-%.2f]' % tuple(pr['ci'])):>18}{pr['wins']*100:>9.0f}%"
              f"   whole-224 {sub['global224']*100:>3.0f}%  ->  crop+enh {sub['crop224_enh']*100:>3.0f}%")


if __name__ == "__main__":
    main()
