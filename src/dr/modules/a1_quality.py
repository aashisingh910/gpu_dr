"""A1 - Adaptive Quality & Domain Gate  (upgraded gate).

The gate is no longer "one blended quality number -> accept/reject".  Five
*independent* axes are scored on the raw fundus photograph and pooled into an
adaptive quality score

    Q* = w1 Q_quality + w2 Q_domain + w3 Q_lesion + w4 Q_blur + w5 Q_illum

  Q_quality  composite gradability / artifact / field-of-view completeness
  Q_domain   how typical this camera+colour fingerprint is of the training
             domain (an out-of-domain image is *usable* but should be routed
             through enhancement, not trusted blind)
  Q_lesion   lesion visibility: contrast-to-noise of lesion-scale round blobs.
             This is the axis that actually predicts whether a microaneurysm
             survives the pipeline, and it is the one a plain quality score
             misses entirely - a sharp, well-exposed, but hazy image scores
             high on every classical metric and still hides every MA.
  Q_blur     variance of the Laplacian inside the field
  Q_illum    exposure + illumination uniformity

Routing:

    Q* < tau_r           -> RETAKE
    tau_r <= Q* < tau_d  -> ENHANCE   (high-resolution enhancement pathway)
    Q* >= tau_d          -> ACCEPT    (normal high-resolution pathway)

The camera/domain recogniser fingerprints the acquisition device from the
image geometry + colour statistics, which is what the downstream cross-domain
evaluation uses to group test images.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from ..config import QualityCfg


@dataclass
class QualityReport:
    # --- raw probes -------------------------------------------------------
    gradability: float
    blur: float
    illumination: float
    artifact: float
    fov: float
    # --- the five Q* axes -------------------------------------------------
    q_quality: float         # composite gradability/artifact/fov
    q_domain: float          # typicality w.r.t. the training domain
    q_lesion: float          # lesion visibility (CNR of lesion-scale blobs)
    q_blur: float
    q_illumination: float
    score: float             # Q*
    legacy_score: float      # the old single-axis Q, kept for comparability
    decision: str            # retake | enhance | accept
    domain: str              # camera/domain fingerprint
    fov_mask: np.ndarray | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("fov_mask")
        return d

    def axes(self) -> dict:
        return {"quality": self.q_quality, "domain": self.q_domain,
                "lesion": self.q_lesion, "blur": self.q_blur,
                "illumination": self.q_illumination}


def retinal_fov_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the illuminated retinal circle (the camera aperture)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # The retina is everything meaningfully brighter than the black surround.
    thr = max(7, int(gray.mean() * 0.18))
    mask = (gray > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:  # keep the largest blob only, drops flash artefacts at the border
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == largest).astype(np.uint8)
    return mask


def _blur_score(gray: np.ndarray, mask: np.ndarray, blur_ref: float) -> float:
    """Variance of the Laplacian inside the FOV, squashed to [0, 1]."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    vals = lap[mask > 0]
    if vals.size == 0:
        return 0.0
    return float(np.clip(vals.var() / blur_ref, 0.0, 1.0))


def _illumination_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """Penalises both global over/under-exposure and strong illumination drift."""
    vals = gray[mask > 0].astype(np.float32)
    if vals.size == 0:
        return 0.0
    mean = vals.mean() / 255.0
    # ideal mean exposure ~0.45; falls off smoothly either side
    exposure = float(np.exp(-((mean - 0.45) ** 2) / (2 * 0.16 ** 2)))
    # uniformity: compare the mean of the 4 quadrants of the FOV
    h, w = gray.shape
    quads = [gray[:h // 2, :w // 2], gray[:h // 2, w // 2:],
             gray[h // 2:, :w // 2], gray[h // 2:, w // 2:]]
    qmask = [mask[:h // 2, :w // 2], mask[:h // 2, w // 2:],
             mask[h // 2:, :w // 2], mask[h // 2:, w // 2:]]
    means = [q[m > 0].mean() for q, m in zip(quads, qmask) if (m > 0).sum() > 32]
    if len(means) < 2:
        return exposure
    drift = float(np.std(means) / (np.mean(means) + 1e-6))
    uniformity = float(np.clip(1.0 - drift * 2.2, 0.0, 1.0))
    return float(0.6 * exposure + 0.4 * uniformity)


def _artifact_score(bgr: np.ndarray, mask: np.ndarray) -> float:
    """1.0 = clean.  Detects dust specks, flare and saturated blowouts."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inside = mask > 0
    if inside.sum() == 0:
        return 0.0
    # saturated (blown-out) pixels: flare / reflection
    blown = float((gray[inside] >= 250).mean())
    # dead pixels
    dead = float((gray[inside] <= 3).mean())
    # small bright specks = dust on the optics; top-hat isolates them
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    specks = float((tophat[inside] > 60).mean())
    penalty = 6.0 * blown + 3.0 * dead + 8.0 * specks
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def _gradability_score(bgr: np.ndarray, mask: np.ndarray) -> float:
    """Can a grader actually see retinal structure?

    Uses green-channel contrast (vessels live in green) plus the density of
    detectable vessel-like ridges.  A hazy/cataract image scores low even when
    it is sharp and correctly exposed.
    """
    green = bgr[:, :, 1]
    inside = mask > 0
    if inside.sum() < 1000:
        return 0.0
    vals = green[inside].astype(np.float32)
    contrast = float(np.clip(vals.std() / 46.0, 0.0, 1.0))
    # vessel ridge density via black-top-hat on the green channel
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bth = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k)
    ridges = (bth > max(8, int(bth[inside].mean() + 2 * bth[inside].std())))
    density = float(ridges[inside].mean())
    vessel = float(np.clip(density / 0.055, 0.0, 1.0))
    return float(0.5 * contrast + 0.5 * vessel)


def _fov_score(mask: np.ndarray) -> float:
    """How complete is the retinal disc?  Clipped/partial fields score low."""
    h, w = mask.shape
    area = float(mask.sum())
    if area < 1000:
        return 0.0
    coverage = area / float(h * w)
    # a full 45-degree fundus circle inscribed in the frame covers ~pi/4 = 0.785
    completeness = float(np.clip(coverage / 0.70, 0.0, 1.0))
    # circularity: partial fields are not round
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        if per > 0:
            circ = float(np.clip(4 * np.pi * cv2.contourArea(c) / (per ** 2), 0, 1))
    return float(0.6 * completeness + 0.4 * circ)


def raw_lesion_cnr(bgr: np.ndarray, mask: np.ndarray) -> float:
    """Unnormalised lesion-scale contrast-to-noise ratio.

        CNR = p99(|lesion-scale residual|) / sigma_noise

    The residual is a difference-of-Gaussians tuned to lesion size, with the
    elongated vessel response subtracted out so vessels cannot masquerade as
    haemorrhages.  sigma_noise is a robust MAD estimate of the high-frequency
    residual, dominated by sensor and compression noise.
    """
    inside = mask > 0
    if inside.sum() < 1000:
        return 0.0
    green = bgr[:, :, 1].astype(np.float32)
    h, w = green.shape
    # lesion scale ~ 0.6% of the field diameter for an MA, ~2% for a haemorrhage
    s_small = max(1.0, 0.006 * max(h, w))
    s_large = max(s_small + 1.0, 0.020 * max(h, w))
    dog = cv2.GaussianBlur(green, (0, 0), s_small) - cv2.GaussianBlur(green, (0, 0), s_large)

    k = max(3, int(2 * round(s_large) + 1))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    vessel = np.zeros_like(dog)
    for ang in (0, 45, 90, 135):
        M = cv2.getRotationMatrix2D((k / 2, 0), ang, 1.0)
        kk = (cv2.warpAffine(kern.astype(np.float32), M, (k, k)) > 0.5).astype(np.uint8)
        if kk.sum() < 3:
            continue
        vessel = np.maximum(
            vessel, cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kk).astype(np.float32))
    resid = np.abs(dog) - 0.6 * (vessel / (vessel.max() + 1e-6)) * np.abs(dog).max()
    vals = resid[inside]
    if vals.size == 0:
        return 0.0

    hf = green - cv2.GaussianBlur(green, (0, 0), 1.0)
    sigma = 1.4826 * float(np.median(np.abs(hf[inside] - np.median(hf[inside])))) + 1e-6
    return float(np.percentile(vals, 99.0) / sigma)


def lesion_visibility_score(bgr: np.ndarray, mask: np.ndarray,
                            ref_cnr: float, floor_cnr: float = 0.0) -> float:
    """Q_lesion - can a lesion-sized blob still be told apart from the noise?

    Every classical quality metric is blind to this.  A hazy-media image can be
    perfectly sharp (high Laplacian variance off the vessel edges), perfectly
    exposed and perfectly framed, and still bury every microaneurysm in the
    background scatter.

    The raw CNR is not comparable across acquisition pipelines - a denoised,
    downsampled mirror of EyePACS has a far lower noise floor than raw camera
    output, so the same retina scores several times higher.  `floor_cnr` and
    `ref_cnr` are therefore *calibrated on the dataset* by
    `calibrate_quality_reference`, not hard-coded: they map that corpus's 5th
    and 85th CNR percentiles onto 0 and 1 so the axis actually discriminates
    instead of saturating at 1.0 for every image.
    """
    cnr = raw_lesion_cnr(bgr, mask)
    span = max(ref_cnr - floor_cnr, 1e-6)
    return float(np.clip((cnr - floor_cnr) / span, 0.0, 1.0))


def calibrate_quality_reference(images: list, sample: int = 400,
                                seed: int = 1337) -> dict:
    """Derive the dataset-specific Q_lesion and Q_domain references.

    `images` is a list of file paths (or BGR arrays).  Returns a dict that can
    be spliced straight into QualityCfg, so the gate is anchored to the corpus
    it will actually run on rather than to a constant that happens to suit one
    camera.
    """
    rng = np.random.default_rng(seed)
    items = list(images)
    if len(items) > sample:
        items = [items[i] for i in rng.choice(len(items), sample, replace=False)]

    cnrs, rgs, fills, sats, blurs = [], [], [], [], []
    for it in items:
        bgr = cv2.imread(str(it), cv2.IMREAD_COLOR) if isinstance(it, (str, bytes)) \
            or hasattr(it, "__fspath__") else it
        if bgr is None:
            continue
        m = retinal_fov_mask(bgr)
        inside = m > 0
        if inside.sum() < 1000:
            continue
        cnrs.append(raw_lesion_cnr(bgr, m))
        lap = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F, ksize=3)
        blurs.append(float(lap[inside].var()))
        g = float(bgr[:, :, 1][inside].mean()) + 1e-6
        rgs.append(float(bgr[:, :, 2][inside].mean()) / g)
        fills.append(float(inside.mean()))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[:, :, 1][inside].mean()) / 255.0)

    if not cnrs:
        return {}
    c = np.asarray(cnrs)
    b = np.asarray(blurs)
    return {
        "lesion_floor_cnr": float(np.percentile(c, 5)),
        "lesion_ref_cnr": float(np.percentile(c, 85)),
        # same reasoning as the CNR bounds: Laplacian variance is not comparable
        # across resolutions or denoising pipelines, so anchor it to this corpus
        "blur_ref": float(np.percentile(b, 85)),
        "domain_ref": {
            "rg_ratio": float(np.mean(rgs)), "rg_sd": float(np.std(rgs) + 1e-3),
            "fill": float(np.mean(fills)), "fill_sd": float(np.std(fills) + 1e-3),
            "sat": float(np.mean(sats)), "sat_sd": float(np.std(sats) + 1e-3),
        },
        "n_sampled": len(cnrs),
        "cnr_percentiles": {str(q): float(np.percentile(c, q))
                            for q in (1, 5, 25, 50, 75, 95, 99)},
    }


def domain_typicality_score(bgr: np.ndarray, mask: np.ndarray, ref: dict) -> float:
    """Q_domain - how close is this acquisition to the training distribution?

    A camera the model has never seen is not a *bad* image, so this must not
    behave like a quality penalty.  It is a z-distance on three cheap,
    illumination-robust fingerprint statistics (red/green balance, field fill
    fraction, saturation), squashed through a Gaussian so that a typical image
    scores ~1 and a 3-sigma outlier scores ~0.1 - enough to push the image down
    the enhancement pathway without triggering a retake on its own.
    """
    inside = mask > 0
    if inside.sum() == 0:
        return 0.0
    b, g, r = [float(bgr[:, :, i][inside].mean()) + 1e-6 for i in range(3)]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    stats = {
        "rg_ratio": r / g,
        "fill": float(inside.mean()),
        "sat": float(hsv[:, :, 1][inside].mean()) / 255.0,
    }
    z2 = 0.0
    for key in ("rg_ratio", "fill", "sat"):
        mu = float(ref.get(key, stats[key]))
        sd = max(float(ref.get(f"{key.split('_')[0]}_sd", 0.2)), 1e-3)
        z2 += ((stats[key] - mu) / sd) ** 2
    return float(np.exp(-0.5 * z2 / 3.0))


def recognise_domain(bgr: np.ndarray, mask: np.ndarray) -> str:
    """Cheap camera fingerprint from aspect ratio + colour balance.

    EyePACS is a multi-site set captured on a mix of cameras; this label is
    what `evaluate.py` uses to slice cross-domain performance.
    """
    h, w = mask.shape
    ar = w / max(h, 1)
    inside = mask > 0
    if inside.sum() == 0:
        return "unknown"
    b, g, r = [float(bgr[:, :, i][inside].mean()) + 1e-6 for i in range(3)]
    rg = r / g
    ys, xs = np.where(inside)
    fill = inside.sum() / float(h * w)
    shape = "wide" if ar > 1.45 else ("square" if ar < 1.15 else "std")
    tone = "warm" if rg > 2.1 else ("neutral" if rg > 1.6 else "cool")
    crop = "full" if fill > 0.72 else "cropped"
    return f"{shape}-{tone}-{crop}"


def assess(bgr: np.ndarray, cfg: QualityCfg) -> QualityReport:
    """Run the full upgraded A1 gate on one BGR uint8 image."""
    mask = retinal_fov_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    grad = _gradability_score(bgr, mask)
    blur = _blur_score(gray, mask, cfg.blur_ref)
    illum = _illumination_score(gray, mask)
    arte = _artifact_score(bgr, mask)
    fov = _fov_score(mask)

    # --- the five Q* axes ------------------------------------------------
    qs = cfg.quality_sub
    denom = sum(qs.values()) or 1.0
    q_quality = (qs["gradability"] * grad + qs["artifact"] * arte
                 + qs["fov"] * fov) / denom
    q_domain = domain_typicality_score(bgr, mask, cfg.domain_ref)
    q_lesion = lesion_visibility_score(bgr, mask, cfg.lesion_ref_cnr,
                                       cfg.lesion_floor_cnr)

    w = cfg.weights
    wsum = sum(w.values()) or 1.0
    score = (w["quality"] * q_quality + w["domain"] * q_domain
             + w["lesion"] * q_lesion + w["blur"] * blur
             + w["illumination"] * illum) / wsum
    score = float(np.clip(score, 0.0, 1.0))

    # the pre-upgrade single-axis score, kept so the two gates stay comparable
    legacy = float(np.clip(0.30 * grad + 0.25 * blur + 0.20 * illum
                           + 0.15 * arte + 0.10 * fov, 0.0, 1.0))

    if score < cfg.tau_r:
        decision = "retake"
    elif score < cfg.tau_d:
        decision = "enhance"
    else:
        decision = "accept"

    return QualityReport(
        gradability=grad, blur=blur, illumination=illum, artifact=arte, fov=fov,
        q_quality=float(q_quality), q_domain=float(q_domain),
        q_lesion=float(q_lesion), q_blur=float(blur), q_illumination=float(illum),
        score=score, legacy_score=legacy, decision=decision,
        domain=recognise_domain(bgr, mask), fov_mask=mask,
    )
