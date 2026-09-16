"""Build model-ready batches from raw IDRiD/DDR images + ophthalmologist masks.

Two consumers:

  * scripts/21_xai_evaluation.py (--ref-masks idrid_ddr) - scores Grad-CAM
    attribution against real annotation instead of EyePACS's weak
    morphological priors (Objective 4 fix).

  * scripts/03_train.py (--aux-lesion-masks) - periodically folds a small
    batch of these real-mask images into EyePACS training as an auxiliary
    lesion-supervision + attribution-consistency loss (Objective 4/2
    follow-up: `lesion_supervision_loss` already accepts a `valid` mask
    specifically so real annotations and weak priors can be mixed in the
    same batch without the pseudo-labels being scored as ground truth - this
    module is what actually supplies the real half of that mix).

Kept out of scripts/ and out of src/dr/data/ (missing from this checkout) so
both call sites share one implementation instead of drifting apart.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from ..modules import a1_quality, a2_preprocess

LESION_NAMES = ("MA", "HE", "EX_H", "EX_S", "NV", "ME")
# IDRiD/DDR annotate these four; NV and ME are never available from them and
# must be excluded from any loss computed against this reference (see
# `lesion_supervision_loss`'s `valid` argument).
ANNOTATED_CHANNELS = ("MA", "HE", "EX_H", "EX_S")


def resize_like_field(img: np.ndarray, fov: np.ndarray, size: int,
                      interp: int) -> np.ndarray:
    """Apply the exact crop/pad/resize geometry of
    `a2_preprocess.extract_retinal_field` to an arbitrary image or mask,
    given the same field-of-view mask used to cache the photograph. This is
    what keeps an external annotation mask pixel-aligned with `pre.image`.
    """
    ys, xs = np.where(fov > 0)
    if ys.size == 0:
        return cv2.resize(img, (size, size), interpolation=interp)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    shape = (side, side, crop.shape[2]) if crop.ndim == 3 else (side, side)
    canvas = np.zeros(shape, crop.dtype)
    canvas[top:top + h, left:left + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=interp)


def build_input_from_preprocessed(pre, priors, cfg, device):
    """Global view + local lesion crops for one freshly-preprocessed image.

    Mirrors scripts/05_predict.py:build_input exactly, so single-image
    inference/training here matches CachedEyePACS.__getitem__.
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
        # pixel grid (0..H-1 here, since fov_mask/pre.image/rgb are all the
        # same cache_size x cache_size frame) - NOT normalised 0-1, so no
        # extra "* H" - that previously pushed every centre out of range and
        # collapsed all crops onto the same clipped corner regardless of
        # where the annotated lesion actually was.
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


def build_idrid_ddr_sample(rec, cfg, device, mask_size: int):
    """Batch + per-channel annotated masks + validity vector for one record.

    Returns (batch, target_masks, valid) where target_masks is
    (1, len(LESION_NAMES), mask_size, mask_size) float32 in {0,1} and valid
    is (len(LESION_NAMES),) - 1.0 for MA/HE/EX_H/EX_S (IDRiD/DDR annotate
    these), 0.0 for NV/ME (never annotated by either corpus, so must not be
    scored as confirmed-absent). Returns None if the image can't be read or
    carries no lesion pixels at all.
    """
    from dr.data.lesion_priors import LesionPriorExtractor

    bgr = cv2.imread(str(rec.image), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    fov = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 12).astype(np.uint8)
    qrep = a1_quality.assess(bgr, cfg.quality)
    pre = a2_preprocess.run(bgr, qrep, cfg.preproc)
    priors = LesionPriorExtractor(out_size=cfg.moe.mask_size)(pre.image, pre.fov_mask)
    batch = build_input_from_preprocessed(pre, priors, cfg, device)
    batch["quality_axes"] = torch.as_tensor(
        np.asarray([[qrep.q_quality, qrep.q_domain, qrep.q_lesion,
                     qrep.q_blur, qrep.q_illumination]], np.float32), device=device)

    target = np.zeros((len(LESION_NAMES), mask_size, mask_size), np.float32)
    valid = np.zeros(len(LESION_NAMES), np.float32)
    any_pixel = False
    for ch, mpath in rec.masks.items():
        if ch not in LESION_NAMES:
            continue
        mk = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mk is None:
            continue
        mk_aligned = resize_like_field(mk, fov, mask_size, cv2.INTER_NEAREST)
        mk_aligned = (mk_aligned > 0).astype(np.float32)
        idx = LESION_NAMES.index(ch)
        target[idx] = mk_aligned
        valid[idx] = 1.0
        any_pixel = any_pixel or bool(mk_aligned.max() > 0)
    if not any_pixel:
        return None
    target_t = torch.as_tensor(target[None], device=device)          # (1,n,S,S)
    valid_t = torch.as_tensor(valid, device=device)                  # (n,)
    return batch, target_t, valid_t
