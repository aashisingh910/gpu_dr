"""A9 - Lesion-Grounded Explainability Engine.

XAI here is not a heat-map bolted onto a finished prediction.  It is a chain
that has to close, link by link, before an explanation is emitted:

    prediction
        v
    attention map          where the model looked
        v
    lesion map             which lesion classes that attention overlaps
        v
    anatomical region      where those lesions sit (macula? periphery?)
        v
    severity contribution  how much each region/lesion pair moved the grade,
                           measured by counterfactual inpainting
        v
    confidence             how well the chain agrees with itself

`explain()` walks that chain and returns a structured `GroundedExplanation`
rather than a picture.  The grounding is *measured*, not asserted: when real
lesion annotations are available (IDRiD / DDR / FGADR) the same object carries
Dice and pointing-game scores against human masks, so "the model looked at the
right place" becomes a number instead of a claim.

Four supporting components, matching the architecture figure:

  1. Foundation attribution  - Grad-CAM over the frozen ViT patch tokens plus
     attention rollout, giving "where did the backbone look".
  2. Lesion evidence         - the A3 expert evidence maps, i.e. "what did it
     think it saw, and of which lesion class".
  3. Attribution-lesion consistency loss
        L_XAI = 1 - Dice(A, M_L)
     applied during training on a *differentiable* attribution (the A5
     global<->lesion cross-attention), which avoids the double-backprop cost of
     differentiating through Grad-CAM while still tying the model's spatial
     evidence to lesion locations.
  4. Counterfactual explanation - digitally remove the detected lesions by
     inpainting and re-run the model; Delta P is the causal contribution of
     those pixels to the predicted risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Attribution:
    saliency: np.ndarray          # (H,W) float32 in [0,1]
    method: str


def _minmax(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max rescale to [0, 1].

    The bounds are **detached on purpose**: this is a scale normalisation, and
    gradient should not flow through *which* pixel happened to be extremal.
    Detaching also removes the min/max backward, whose scatter over the
    flattened map is a known MPS failure mode (it emits index -1 on ties/NaN).
    """
    flat = x.flatten(1)
    lo = flat.min(1, keepdim=True).values.detach()
    hi = flat.max(1, keepdim=True).values.detach()
    out = (flat - lo) / (hi - lo + 1e-8)
    return out.clamp(0.0, 1.0).nan_to_num(0.0).view_as(x)


def differentiable_attribution(attn_g2l: torch.Tensor, grid: int,
                               out_size: int, n_spatial: int | None = None
                               ) -> torch.Tensor:
    """Spatial attribution from the A5 cross-attention, kept in the graph.

    attn_g2l: (B, heads, N_global, N_lesion). Averaging over heads and lesion
    keys gives, for each global patch, how strongly it engaged lesion evidence.

    With the local branch on, the query axis carries `grid*grid` grid-aligned
    tokens followed by one token per lesion crop.  Only the leading grid tokens
    can be laid back out spatially, so `n_spatial` selects them; the crop
    tokens have no image position and would otherwise corrupt the reshape.
    """
    a = attn_g2l.mean(1).sum(-1)                       # (B, N_global)
    if n_spatial is not None:
        a = a[:, :n_spatial]
    B, N = a.shape
    a = a.view(B, 1, grid, grid)
    a = F.interpolate(a, size=(out_size, out_size), mode="bilinear",
                      align_corners=False)
    return _minmax(a)                                  # (B,1,S,S) in [0,1]


def attribution_consistency_loss(attribution: torch.Tensor,
                                 lesion_masks: torch.Tensor) -> torch.Tensor:
    """L_XAI = 1 - Dice(A, M_L) with M_L = any-lesion union mask.

    Images with no detected lesion are skipped: forcing the attribution to
    match an empty mask would push the map to zero everywhere and destroy the
    signal for healthy retinas.
    """
    m = lesion_masks.amax(1, keepdim=True).to(attribution.dtype)
    if attribution.shape[-2:] != m.shape[-2:]:
        m = F.interpolate(m, size=attribution.shape[-2:], mode="bilinear",
                          align_corners=False)
    has = (m.flatten(1).sum(1) > 1e-3)
    if has.sum() == 0:
        return attribution.sum() * 0.0
    a, m = attribution[has], m[has]
    inter = (a * m).flatten(1).sum(1)
    dice = (2 * inter + 1.0) / (a.flatten(1).sum(1) + m.flatten(1).sum(1) + 1.0)
    return (1.0 - dice).mean()


class GradCAM:
    """Grad-CAM on a ViT's patch tokens (used at inference / reporting time)."""

    def __init__(self, model, target_block=None):
        self.model = model
        self.acts = None
        self.grads = None
        blocks = model.backbone.vit.blocks
        self.block = target_block if target_block is not None else blocks[-1]
        self._h1 = self.block.register_forward_hook(self._save_act)
        self._h2 = self.block.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, o):
        self.acts = o if isinstance(o, torch.Tensor) else o[0]

    def _save_grad(self, _m, _gi, go):
        self.grads = go[0]

    def remove(self):
        self._h1.remove(); self._h2.remove()

    def __call__(self, batch: dict, out_size: int = 224) -> torch.Tensor:
        """Returns (B,1,out,out) saliency for the model's expected DR grade."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(batch)
        score = out["ordinal"].expected_grade.sum()
        score.backward(retain_graph=False)

        acts, grads = self.acts, self.grads
        npx = getattr(self.model.backbone.vit, "num_prefix_tokens", 1)
        a, g = acts[:, npx:], grads[:, npx:]
        weights = g.mean(1, keepdim=True)                     # channel importance
        cam = F.relu((a * weights).sum(-1))                   # (B,N)
        B, N = cam.shape
        grid = int(N ** 0.5)
        cam = cam.view(B, 1, grid, grid)
        cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear",
                            align_corners=False)
        return _minmax(cam).detach()


@torch.no_grad()
def attention_rollout(model, images: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """Abnar & Zuidema rollout across the ViT blocks (identity-augmented)."""
    attns: list[torch.Tensor] = []
    hooks = []

    def make_hook(block):
        def hook(_m, inp, _out):
            x = inp[0]
            B, N, C = x.shape
            attn_mod = block.attn
            qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads,
                                          C // attn_mod.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            a = (q @ k.transpose(-2, -1)) * attn_mod.scale
            attns.append(a.softmax(-1).detach())
        return hook

    for blk in model.backbone.vit.blocks:
        hooks.append(blk.attn.register_forward_hook(make_hook(blk)))
    model.backbone(images)
    for h in hooks:
        h.remove()
    if not attns:
        return torch.zeros(images.shape[0], 1, out_size, out_size, device=images.device)

    result = None
    for a in attns:
        a = a.mean(1)                                        # average heads
        a = a + torch.eye(a.shape[-1], device=a.device).unsqueeze(0)
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    npx = getattr(model.backbone.vit, "num_prefix_tokens", 1)
    cam = result[:, 0, npx:]                                  # CLS -> patches
    B, N = cam.shape
    grid = int(N ** 0.5)
    cam = cam.view(B, 1, grid, grid)
    cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear",
                        align_corners=False)
    return _minmax(cam)


def counterfactual_image(bgr: np.ndarray, lesion_mask: np.ndarray,
                         radius: int = 5) -> np.ndarray:
    """Digitally erase the detected lesions by inpainting from surrounding retina."""
    h, w = bgr.shape[:2]
    m = cv2.resize(lesion_mask.astype(np.float32), (w, h),
                   interpolation=cv2.INTER_LINEAR)
    m = (m > 0.35).astype(np.uint8)
    if m.sum() == 0:
        return bgr.copy()
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, m, radius, cv2.INPAINT_TELEA)


def dice_score(a: np.ndarray, b: np.ndarray, thresh: float = 0.5) -> float:
    """Dice between a continuous attribution map and a binary lesion mask."""
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.float32), a.shape[::-1],
                       interpolation=cv2.INTER_LINEAR)
    ab = (a >= thresh).astype(np.float32)
    bb = (b >= 0.5).astype(np.float32)
    s = ab.sum() + bb.sum()
    if s == 0:
        return float("nan")
    return float(2.0 * (ab * bb).sum() / s)


def pointing_game(attribution: np.ndarray, lesion_mask: np.ndarray) -> float:
    """1.0 if the attribution's peak falls inside a lesion region."""
    if lesion_mask.max() < 0.5:
        return float("nan")
    m = cv2.resize(lesion_mask.astype(np.float32), attribution.shape[::-1],
                   interpolation=cv2.INTER_LINEAR)
    idx = int(np.argmax(attribution))
    y, x = divmod(idx, attribution.shape[1])
    return float(m[y, x] >= 0.5)


# ===========================================================================
#  Lesion-grounded explanation chain
# ===========================================================================
from dataclasses import field           # noqa: E402
from ..data.lesion_priors import ANATOMY_NAMES, LESION_NAMES   # noqa: E402


@dataclass
class LesionCitation:
    """One link of the chain: a lesion class, where it is, what it cost."""
    lesion: str
    anatomy: str
    attention_overlap: float     # share of the attention mass on this lesion
    evidence: float              # the expert's own confidence in it
    severity_contribution: float # dgrade when these pixels are inpainted away
    annotated_dice: float | None = None   # vs real masks, when available


@dataclass
class GroundedExplanation:
    predicted_grade: int
    expected_grade: float
    referable_prob: float
    citations: list                       # list[LesionCitation], ranked
    region_severity: dict                 # anatomy -> severity contribution
    attention_on_lesion: float            # fraction of attention inside lesions
    chain_confidence: float               # [0,1] agreement across the chain
    uncertainty: float | None = None
    note: str = ""


def _to_np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def attention_lesion_overlap(attribution: np.ndarray, lesion_maps: np.ndarray
                             ) -> np.ndarray:
    """Per-lesion share of the attention mass. attribution (H,W), maps (L,H,W)."""
    a = attribution.astype(np.float32)
    if a.sum() <= 0:
        return np.zeros(len(lesion_maps), np.float32)
    out = []
    for m in lesion_maps:
        mm = m
        if mm.shape != a.shape:
            mm = cv2.resize(mm.astype(np.float32), a.shape[::-1],
                            interpolation=cv2.INTER_LINEAR)
        out.append(float((a * (mm > 0.5)).sum() / (a.sum() + 1e-8)))
    return np.asarray(out, np.float32)


def dominant_region(lesion_map: np.ndarray, anatomy_maps: np.ndarray) -> tuple[str, float]:
    """Which anatomical region does this lesion class mostly sit in?"""
    m = (lesion_map > 0.5).astype(np.float32)
    if m.sum() < 1:
        return "none", 0.0
    scores = []
    for a in anatomy_maps:
        aa = a
        if aa.shape != m.shape:
            aa = cv2.resize(aa.astype(np.float32), m.shape[::-1],
                            interpolation=cv2.INTER_LINEAR)
        scores.append(float((m * aa).sum() / (m.sum() + 1e-8)))
    i = int(np.argmax(scores))
    return ANATOMY_NAMES[i], float(scores[i])


@torch.no_grad()
def severity_contribution(model, batch: dict, lesion_maps: torch.Tensor,
                          per_class: bool = True) -> np.ndarray:
    """Counterfactual dgrade per lesion class.

    The pixels of one lesion class are blurred out of the *input tensor* (a
    differentiable-free stand-in for inpainting that works on the normalised
    batch the model actually sees) and the model is re-run.  The drop in the
    expected grade is that class's causal contribution.  This is the only link
    of the chain that measures effect rather than correlation.
    """
    base = model(batch)["ordinal"].expected_grade
    img = batch["image"]
    L = lesion_maps.shape[1]
    out = np.zeros((img.shape[0], L), np.float32)
    if not per_class:
        return out
    for k in range(L):
        m = lesion_maps[:, k:k + 1].float()
        m = F.interpolate(m, size=img.shape[-2:], mode="bilinear", align_corners=False)
        m = (m > 0.5).to(img.dtype)
        if float(m.sum()) == 0:
            continue
        blurred = F.avg_pool2d(img, 9, stride=1, padding=4)
        cf = dict(batch)
        cf["image"] = img * (1 - m) + blurred * m
        alt = model(cf)["ordinal"].expected_grade
        out[:, k] = _to_np(base - alt)
    return out


def explain(model, batch: dict, index: int = 0, topk: int = 3,
            annotated_masks: np.ndarray | None = None,
            uncertainty: float | None = None) -> GroundedExplanation:
    """Walk the full grounding chain for one sample of a batch."""
    with torch.no_grad():
        out = model(batch)
    ordi = out["ordinal"]
    attribution = _to_np(out["attribution"])[index, 0]
    evidence = torch.sigmoid(out["moe"].evidence)
    lesion_maps = _to_np(evidence)[index]
    anatomy = _to_np(batch["anatomy"])[index]

    overlap = attention_lesion_overlap(attribution, lesion_maps)
    contrib = severity_contribution(model, batch, (evidence > 0.5).float())[index]

    citations = []
    region_sev: dict = {n: 0.0 for n in ANATOMY_NAMES}
    for k, name in enumerate(LESION_NAMES):
        ev = float(lesion_maps[k].max())
        if ev < 0.5 and overlap[k] < 1e-3:
            continue
        region, frac = dominant_region(lesion_maps[k], anatomy)
        dice = None
        if annotated_masks is not None and k < len(annotated_masks):
            dice = dice_score(lesion_maps[k], annotated_masks[k])
        citations.append(LesionCitation(
            lesion=name, anatomy=region, attention_overlap=float(overlap[k]),
            evidence=ev, severity_contribution=float(contrib[k]),
            annotated_dice=dice))
        if region in region_sev:
            region_sev[region] += float(contrib[k])
    citations.sort(key=lambda c: abs(c.severity_contribution), reverse=True)

    # union, not the per-class sum: the six evidence maps overlap (an ME region
    # is by construction made of HE/EX/MA pixels), so summing their shares can
    # report more than 100% of the attention mass
    union = lesion_maps.max(0)
    attn_on_lesion = float(attention_lesion_overlap(attribution, union[None])[0])
    # chain confidence: the explanation is only trustworthy when the model's
    # attention, its lesion evidence and its causal contributions point the
    # same way.  Disagreement anywhere collapses it.
    cited = citations[:topk]
    if not cited:
        chain_conf = 0.0
        note = "no lesion evidence above threshold; grade is not lesion-grounded"
    else:
        agree = float(np.mean([
            min(1.0, c.evidence) * min(1.0, 4 * c.attention_overlap)
            * (1.0 if c.severity_contribution > 0 else 0.3) for c in cited]))
        chain_conf = float(np.clip(0.5 * agree + 0.5 * min(1.0, 3 * attn_on_lesion), 0, 1))
        note = ""

    return GroundedExplanation(
        predicted_grade=int(_to_np(ordi.class_probs)[index].argmax()),
        expected_grade=float(_to_np(ordi.expected_grade)[index]),
        referable_prob=float(_to_np(ordi.cum_probs)[index, 1]),
        citations=cited, region_severity=region_sev,
        attention_on_lesion=attn_on_lesion, chain_confidence=chain_conf,
        uncertainty=uncertainty, note=note)


def explanation_report(exp: GroundedExplanation) -> str:
    """Render the chain as the text a clinician would actually read."""
    lines = [f"Predicted grade {exp.predicted_grade} "
             f"(continuous severity {exp.expected_grade:.2f}, "
             f"P(referable) = {exp.referable_prob:.3f})"]
    if exp.uncertainty is not None:
        lines.append(f"Predictive uncertainty: {exp.uncertainty:.3f}")
    lines.append(f"Attention inside lesion evidence: {exp.attention_on_lesion:.1%}")
    lines.append(f"Explanation chain confidence:     {exp.chain_confidence:.3f}")
    if exp.note:
        lines.append(f"NOTE: {exp.note}")
    if exp.citations:
        lines.append("")
        lines.append("Grounded evidence, ranked by causal contribution:")
        for c in exp.citations:
            d = "" if c.annotated_dice is None else f", Dice vs annotation {c.annotated_dice:.3f}"
            lines.append(
                f"  - {c.lesion:5s} in {c.anatomy:12s} | attention {c.attention_overlap:6.1%}"
                f" | evidence {c.evidence:.2f} | dgrade {c.severity_contribution:+.3f}{d}")
    nz = {k: v for k, v in exp.region_severity.items() if abs(v) > 1e-4}
    if nz:
        lines.append("")
        lines.append("Severity contribution by anatomical region:")
        for k, v in sorted(nz.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  - {k:12s} {v:+.3f}")
    return "\n".join(lines)


def xai_weight_for_stage(cfg_xai, stage_idx: int, progress: float = 1.0) -> float:
    """lambda_XAI over the staged schedule: 0 -> 0 -> 0.02 -> 0.05.

    Held at zero while the high-resolution encoder and the lesion-pretrained
    MoE settle - an attribution-consistency term applied to a model whose
    attention is still random just teaches it to match noise.
    """
    sched = list(cfg_xai.weight_schedule)
    if not sched:
        return 0.0
    w = sched[min(max(stage_idx, 0), len(sched) - 1)]
    if not cfg_xai.ramp_within_stage or w == 0.0:
        return float(w)
    prev = sched[min(max(stage_idx - 1, 0), len(sched) - 1)]
    return float(prev + (w - prev) * min(max(progress, 0.0), 1.0))
