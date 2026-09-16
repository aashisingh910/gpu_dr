"""A2 - Adaptive Lesion-Preserving Preprocessing.

Pipeline, in the order drawn in the architecture figure:

    retinal field extraction -> illumination normalisation -> adaptive CLAHE
    -> vessel enhancement -> lesion enhancement -> quality-aware fusion -> I*

The fusion step is what makes it *adaptive*: the A1 quality score decides how
much of the aggressively enhanced image is blended into the output, so a clean
image is left close to native (enhancement can destroy microaneurysms) while a
poor image is pushed hard.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import PreprocCfg
from .a1_quality import QualityReport


@dataclass
class PreprocOutput:
    image: np.ndarray          # I* , BGR uint8, square, cfg.image_size
    fov_mask: np.ndarray       # uint8 {0,1}
    vessels: np.ndarray        # float32 [0,1] vessel-likelihood map
    lesion_boost: np.ndarray   # float32 [0,1] lesion-contrast map


def extract_retinal_field(bgr: np.ndarray, mask: np.ndarray,
                          size: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the bounding box of the retinal circle and pad to a square."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        return img, np.ones((size, size), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = bgr[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side, 3), crop.dtype)
    cmask = np.zeros((side, side), np.uint8)
    canvas[top:top + h, left:left + w] = crop
    cmask[top:top + h, left:left + w] = m
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(cmask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, out_mask


def normalise_illumination(bgr: np.ndarray, mask: np.ndarray,
                           sigma: float) -> np.ndarray:
    """Subtract a large-scale Gaussian background estimate, per channel.

    This is the classical Graham fundus normalisation; it removes the vignette
    and inter-camera exposure differences that dominate EyePACS.
    """
    img = bgr.astype(np.float32)
    bg = cv2.GaussianBlur(img, (0, 0), sigma)
    out = img - bg + 128.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[mask == 0] = 0
    return out


def adaptive_clahe(bgr: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """CLAHE on the L channel of LAB - boosts local contrast, keeps colour.

    Default enhancement operator (PreprocCfg.enhancement_algorithm='clahe').
    See `local_contrast_normalize` for an alternative that was benchmarked
    and rejected - kept only so that comparison is reproducible.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


def local_contrast_normalize(bgr: np.ndarray, sigma: float,
                             strength: float) -> np.ndarray:
    """Division-based local contrast normalisation - an ALTERNATIVE to CLAHE
    that was tried as a full replacement and REJECTED after benchmarking.
    Kept only so the comparison is reproducible; do not use as the default
    (see PreprocCfg.enhancement_algorithm).

    The theoretical case for it seemed sound: no tile boundaries, so it
    should not suffer CLAHE's per-tile histogram redistribution that
    suppresses large lesions. Measured, it does not hold up:

      - At a sigma large enough to normalise meaningfully, the local mean
        is computed over a neighbourhood comparable to or larger than a
        soft exudate's own radius, so the blob pulls its OWN local mean
        toward itself - the same "lesion gets normalised against itself"
        failure as a CLAHE tile, just via a Gaussian kernel instead of a
        histogram. Measured on a synthetic soft-exudate-scale blob: CNR
        0.24-0.56 across sigma in [8, 60], all WORSE than doing nothing
        (native 0.79) and worse than CLAHE (0.79, effectively unchanged).
      - Two full-resolution float Gaussian blurs (mean, mean-of-squares) at
        a large sigma are 3-17x SLOWER than OpenCV's tuned CLAHE (measured:
        60-310ms vs 17.6ms per 1024px image), because OpenCV's CLAHE is a
        highly optimised tiled-histogram C++ implementation, not something
        a general Gaussian-blur-based op beats by default.

    The lesson: "no tile boundaries" is not sufficient to avoid the
    large-lesion-suppression failure mode, and "fewer named stages" is not
    sufficient to be faster than a library implementation that has already
    been optimised for exactly this operation. `protect_large_lesions`
    (targeted, small extra cost on top of CLAHE) remains the validated fix.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    mu = cv2.GaussianBlur(lf, (0, 0), sigma)
    mu_sq = cv2.GaussianBlur(lf * lf, (0, 0), sigma)
    sd = np.sqrt(np.clip(mu_sq - mu * mu, 0.0, None))
    norm = (lf - mu) / (sd + 8.0)                 # +8 caps flat-region noise gain
    norm = np.clip(norm * 32.0 + 128.0, 0, 255)   # rescale z-score back to L range
    out_l = np.clip(strength * norm + (1.0 - strength) * lf, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([out_l, a, b]), cv2.COLOR_LAB2BGR)


def large_bright_blob_mask(bgr: np.ndarray, kernel: int) -> np.ndarray:
    """Detect lesions that are already large AND already high-contrast.

    A large-kernel morphological OPENING of the "brighter than broad local
    background" map keeps only bright regions that are at least `kernel`
    pixels wide - exactly a soft exudate's scale - while removing small
    bright features (reflections, small exudates, noise) that cannot contain
    a structuring element that size. (A top-hat, by contrast, would do the
    opposite: it responds to features SMALLER than the kernel and is near
    zero in the interior of a large uniform blob, which is not what we want
    here.) Returns a float32 [0,1] map, high where a large bright lesion is
    likely to sit.
    """
    green = bgr[:, :, 1].astype(np.float32)
    sigma = max(3.0, kernel / 2.0)
    bg = cv2.GaussianBlur(green, (0, 0), sigma)
    excess = np.clip(green - bg, 0.0, None)          # brighter than broad surroundings
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    large = cv2.morphologyEx(excess, cv2.MORPH_OPEN, k)
    large = cv2.GaussianBlur(large, (5, 5), 0)
    lo, hi = float(large.min()), float(large.max())
    if hi - lo < 1e-6:
        return np.zeros_like(large)
    return np.clip((large - lo) / (hi - lo), 0.0, 1.0)


def protect_large_lesions(native: np.ndarray, clahe_out: np.ndarray,
                          kernel: int, strength: float) -> np.ndarray:
    """Blend CLAHE output back toward native pixels wherever a large, already
    high-contrast lesion is detected.

    Objective 2's evidence: soft exudates are the one lesion class where
    enhancement HURTS (paired median CNR ratio 0.945x, CI 0.897-1.039, win
    rate 43%) - they are large and already high-contrast, and CLAHE's
    tile-local clip limit can suppress a lesion that fills or exceeds its
    tile. `strength=0` reproduces the old, unprotected behaviour;
    `strength=1` fully reverts CLAHE inside detected large-bright regions.
    """
    if strength <= 0:
        return clahe_out
    mask = large_bright_blob_mask(native, kernel)[..., None] * float(strength)
    return (clahe_out.astype(np.float32) * (1.0 - mask)
            + native.astype(np.float32) * mask).astype(np.uint8)


def enhance_vessels(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multi-scale black-top-hat on the green channel -> vessel likelihood."""
    green = bgr[:, :, 1]
    acc = np.zeros_like(green, np.float32)
    for r in (5, 9, 13):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        acc = np.maximum(acc, cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k).astype(np.float32))
    acc = cv2.GaussianBlur(acc, (3, 3), 0)
    m = acc[mask > 0]
    if m.size:
        acc = (acc - m.min()) / (np.ptp(m) + 1e-6)
    acc = np.clip(acc, 0, 1).astype(np.float32)
    acc[mask == 0] = 0
    return acc


def enhance_lesions(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Contrast map that favours lesion-scale blobs over vessels.

    Bright lesions (exudates) come from a white top-hat on the green channel;
    dark lesions (microaneurysms, haemorrhages) from a black top-hat that is
    then *penalised* wherever the elongated vessel response is strong, since
    vessels and haemorrhages share intensity but not shape.
    """
    green = bgr[:, :, 1]
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, k_small).astype(np.float32)
    dark = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k_small).astype(np.float32)
    # long structuring element responds to vessels, not to round lesions
    k_long = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    vessel_like = np.zeros_like(dark)
    for ang in range(0, 180, 30):
        M = cv2.getRotationMatrix2D((10, 0), ang, 1.0)
        kk = cv2.warpAffine(k_long.astype(np.float32), M, (21, 21)) > 0.5
        kk = kk.astype(np.uint8)
        if kk.sum() < 3:
            continue
        vessel_like = np.maximum(
            vessel_like, cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kk).astype(np.float32))
    dark_round = np.clip(dark - 0.85 * vessel_like, 0, None)
    out = np.maximum(bright, dark_round)
    inside = out[mask > 0]
    if inside.size:
        out = (out - inside.min()) / (np.ptp(inside) + 1e-6)
    out = np.clip(out, 0, 1).astype(np.float32)
    out[mask == 0] = 0
    return out


def quality_aware_fusion(native: np.ndarray, enhanced: np.ndarray,
                         q: float, gamma: float) -> np.ndarray:
    """Blend native and enhanced images with a weight driven by A1's score.

    alpha = gamma * (1 - Q): a pristine image (Q~1) keeps its native pixels,
    a marginal image (Q~0.4) is pulled strongly toward the enhanced version.
    """
    alpha = float(np.clip(gamma * (1.0 - q), 0.0, 1.0))
    return cv2.addWeighted(enhanced, alpha, native, 1.0 - alpha, 0.0)


def run(bgr: np.ndarray, report: QualityReport, cfg: PreprocCfg) -> PreprocOutput:
    """Cache-time A2: retinal-field extraction at `cache_size` + A1 routing.

    What lands in the cache changed with the adaptive ALPP.  The aggressive
    fixed enhancement used to be baked into the cached pixels, which meant the
    learned fusion downstream could only ever re-mix an already-damaged image.
    Now:

      Q* high    (accept)  -> the *native* retinal field is cached
      Q* medium  (enhance) -> the high-resolution enhancement pathway runs and
                              its output is cached instead
      Q* low     (retake)  -> still cached and flagged, so the retake rate can
                              be measured, but the record carries decision=retake

    Either way the learned ALPP does the per-image enhancement mixing at train
    time, on GPU, differentiably.  The vessel/lesion maps are still returned
    because the weak-prior extractor and the lesion crop generator consume them.
    """
    mask = report.fov_mask if report.fov_mask is not None else np.ones(bgr.shape[:2], np.uint8)
    field, fmask = extract_retinal_field(bgr, mask, cfg.cache_size)

    if getattr(cfg, "raw_pixels", False):
        # Objective 3's decisive experiment: field extraction only, no
        # illumination normalisation, no CLAHE, no lesion-boost fusion. Lets
        # a linear probe test whether RETFound's frozen-feature disadvantage
        # vs generic ImageNet backbones is about missing retinal domain
        # knowledge, or about A1/A2 preprocessing not matching the statistics
        # RETFound was itself pretrained on.
        vessels = enhance_vessels(field, fmask)
        lesions = enhance_lesions(field, fmask)
        out = field.copy()
        out[fmask == 0] = 0
        return PreprocOutput(image=out, fov_mask=fmask, vessels=vessels,
                             lesion_boost=lesions)

    illum = normalise_illumination(field, fmask, cfg.illum_sigma)
    algo = getattr(cfg, "enhancement_algorithm", "clahe")
    if algo == "clahe":
        # a lower-quality image gets a higher CLAHE clip limit -> the "adaptive" part
        clip = cfg.clahe_clip * (1.0 + 0.8 * (1.0 - report.score))
        enhanced_l = adaptive_clahe(illum, clip, cfg.clahe_grid)
    else:
        # same adaptive schedule (lower quality -> stronger correction),
        # applied to the tile-free local-contrast operator instead
        strength = min(1.0, cfg.local_contrast_strength * (1.0 + 0.8 * (1.0 - report.score)))
        enhanced_l = local_contrast_normalize(illum, cfg.local_contrast_sigma, strength)
    if getattr(cfg, "clahe_protect_large_lesions", False):
        enhanced_l = protect_large_lesions(illum, enhanced_l, cfg.large_lesion_kernel,
                                           cfg.large_lesion_protect_strength)
    clahe = enhanced_l   # name kept for the rest of this function / downstream callers

    vessels = enhance_vessels(clahe, fmask)
    lesions = enhance_lesions(clahe, fmask)

    if report.decision == "enhance":
        # high-resolution enhancement pathway: fold the lesion contrast map back
        # in so lesion pixels survive the downstream resize instead of being
        # averaged away, then blend by how far below tau_d the image sits.
        boost = (lesions[..., None] * np.float32([40, 40, 40]))
        enhanced = np.clip(clahe.astype(np.float32) + boost, 0, 255).astype(np.uint8)
        enhanced[fmask == 0] = 0
        out = quality_aware_fusion(field, enhanced, report.score, cfg.fusion_gamma)
    else:
        out = field.copy()
    out[fmask == 0] = 0
    return PreprocOutput(image=out, fov_mask=fmask, vessels=vessels,
                         lesion_boost=lesions)


# ===========================================================================
#  Adaptive ALPP  -  learned per-image fusion of the four enhancement branches
# ===========================================================================
"""The OpenCV stages above run once, at cache time, on full-resolution pixels.
They are fixed: every image gets the same recipe, steered only by the scalar
A1 score.  That is the part being replaced here.

`ALPP` re-implements the same four branches as differentiable torch ops and
fuses them with weights predicted per image:

    I* = w1 I_norm + w2 I_contrast + w3 I_vessel + w4 I_lesion
    [w1, w2, w3, w4] = Softmax(G(I))

so a hazy macula-centred image can ask for contrast + lesion enhancement while
a clean disc-centred one asks for almost pure illumination normalisation.  The
branch operators stay classical (no free parameters, so they cannot drift into
destroying microaneurysms); only the mixing is learned.
"""

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402
import torch.nn.functional as F   # noqa: E402

from ..config import PreprocCfg as _PreprocCfg   # noqa: E402


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    r = max(1, int(3.0 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable depthwise Gaussian; fixed kernel, fully differentiable in x."""
    if sigma <= 0:
        return x
    C = x.shape[1]
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kh, groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv, groups=C)


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def black_tophat(x: torch.Tensor, k: int) -> torch.Tensor:
    """Morphological closing minus the image: isolates dark structures."""
    return _erode(_dilate(x, k), k) - x


def white_tophat(x: torch.Tensor, k: int) -> torch.Tensor:
    return x - _dilate(_erode(x, k), k)


def _unit(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max to [0,1]; bounds detached (scale, not signal)."""
    f = x.flatten(1)
    lo = f.min(1, keepdim=True).values.detach()
    hi = f.max(1, keepdim=True).values.detach()
    return ((f - lo) / (hi - lo + 1e-6)).clamp(0, 1).view_as(x)


def branch_illumination(x: torch.Tensor, sigma_frac: float = 0.06) -> torch.Tensor:
    """I_norm - Graham background subtraction, the vignette/exposure remover."""
    sigma = max(1.0, sigma_frac * x.shape[-1])
    return (x - gaussian_blur(x, sigma) + 0.5).clamp(0, 1)


def branch_local_contrast(x: torch.Tensor, sigma_frac: float = 0.02,
                          strength: float = 0.8,
                          large_lesion_sigma_frac: float = 0.06,
                          protect_strength: float = 0.6) -> torch.Tensor:
    """I_contrast - CLAHE-equivalent local contrast normalisation.

    (x - mu_local) / (sigma_local + eps) is the continuous analogue of CLAHE;
    `strength` blends back toward the original so flat regions do not have
    their noise amplified the way an unclipped normalisation would.

    Mathematically the same family as `a2_preprocess.local_contrast_normalize`
    (division normalisation via Gaussian local moments), used here because it
    needs to be differentiable for the learned ALPP gate - NOT because it was
    shown to beat CLAHE; benchmarked at cache time, the classical version of
    this operator was measurably worse than CLAHE on both speed and
    soft-exudate contrast (see that function's docstring) and is not used as
    the cache-time default. The `protect_large_lesions` mitigation below
    applies to this branch's output too, for the same reason.

    Objective 2 fix: this local normalisation shares CLAHE's failure mode on
    soft exudates (large, already high-contrast blobs get suppressed rather
    than boosted). A large-scale bright-blob estimate is used to blend the
    normalised output back toward the original wherever a big bright region
    is detected, mirroring `protect_large_lesions` in the classical (cache-
    time) chain, so the learned ALPP mixing does not re-introduce the same
    regression.
    """
    sigma = max(1.0, sigma_frac * x.shape[-1])
    mu = gaussian_blur(x, sigma)
    var = (gaussian_blur(x * x, sigma) - mu * mu).clamp_min(0.0)
    sd = var.sqrt()
    norm = (x - mu) / (sd + 0.08)
    norm = (norm * 0.25 + 0.5).clamp(0, 1)
    out = (strength * norm + (1 - strength) * x).clamp(0, 1)

    if protect_strength > 0:
        g = x[:, 1:2]
        big_sigma = max(1.0, large_lesion_sigma_frac * x.shape[-1])
        large_bright = (g - gaussian_blur(g, big_sigma)).clamp_min(0.0)
        large_bright = _unit(large_bright) * float(protect_strength)
        out = out * (1 - large_bright) + x * large_bright
    return out.clamp(0, 1)


def branch_vessels(x: torch.Tensor) -> torch.Tensor:
    """I_vessel - multi-scale black-top-hat on green, broadcast to 3 channels."""
    g = x[:, 1:2]
    acc = None
    for k in (5, 9, 13):
        r = black_tophat(g, k)
        acc = r if acc is None else torch.maximum(acc, r)
    v = _unit(acc).expand_as(x)
    return (0.5 * x + 0.5 * v).clamp(0, 1)


def branch_lesions(x: torch.Tensor) -> torch.Tensor:
    """I_lesion - round-blob enhancement with the elongated response removed.

    Bright lesions from a white top-hat, dark lesions from a black top-hat that
    is penalised wherever a long structuring element also fires, since vessels
    and haemorrhages share intensity but not shape.
    """
    g = x[:, 1:2]
    bright = white_tophat(g, 9)
    dark = black_tophat(g, 9)
    # a 1xK max-pool is the separable elongated erosion/dilation pair
    long_dark = -F.max_pool2d(-F.max_pool2d(g, (1, 21), stride=1, padding=(0, 10)),
                              (1, 21), stride=1, padding=(0, 10)) - g
    long_dark = torch.maximum(
        long_dark,
        -F.max_pool2d(-F.max_pool2d(g, (21, 1), stride=1, padding=(10, 0)),
                      (21, 1), stride=1, padding=(10, 0)) - g)
    round_dark = (dark - 0.85 * long_dark).clamp_min(0)
    lesion = _unit(torch.maximum(bright, round_dark))
    return (x + 0.6 * lesion.expand_as(x)).clamp(0, 1)


class ALPP(nn.Module):
    """Algorithm 2, made adaptive: fixed branches, learned per-image mixing."""

    BRANCHES = {"norm": branch_illumination, "contrast": branch_local_contrast,
                "vessel": branch_vessels, "lesion": branch_lesions}

    def __init__(self, cfg: _PreprocCfg):
        super().__init__()
        self.cfg = cfg
        self.names = tuple(cfg.alpp_branches)
        n = len(self.names)
        d = cfg.alpp_gate_dim
        # G(I): a tiny conv net on a heavily downsampled image.  It only has to
        # recognise "hazy", "dark", "vessel-poor" - it must not be big enough to
        # start doing the classification itself.
        self.gate = nn.Sequential(
            nn.Conv2d(3, d // 2, 3, 2, 1), nn.GroupNorm(4, d // 2), nn.SiLU(),
            nn.Conv2d(d // 2, d, 3, 2, 1), nn.GroupNorm(4, d), nn.SiLU(),
            nn.Conv2d(d, d, 3, 2, 1), nn.GroupNorm(4, d), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(d + 5, n)          # +5 = the A1 Q* axes
        nn.init.zeros_(self.head.weight)
        prior = list(cfg.alpp_prior)[:n] + [0.0] * max(0, n - len(cfg.alpp_prior))
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(prior, dtype=torch.float32))

    def weights(self, x01: torch.Tensor, q_axes: torch.Tensor | None) -> torch.Tensor:
        small = F.interpolate(x01, size=(128, 128), mode="bilinear",
                              align_corners=False)
        h = self.gate(small)
        if q_axes is None:
            q_axes = torch.zeros(x01.shape[0], 5, device=x01.device, dtype=h.dtype)
        return torch.softmax(self.head(torch.cat([h, q_axes.to(h.dtype)], 1)), -1)

    def forward(self, x01: torch.Tensor, q_axes: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """x01: (B,3,H,W) in [0,1].  Returns (I*, per-image branch weights)."""
        w = self.weights(x01, q_axes)                        # (B, n)
        out = None
        for i, nm in enumerate(self.names):
            b = self.BRANCHES[nm](x01)
            wi = w[:, i].view(-1, 1, 1, 1)
            out = wi * b if out is None else out + wi * b
        return out.clamp(0, 1), w
