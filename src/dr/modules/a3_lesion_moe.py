"""A3 - Lesion Specialist Mixture-of-Experts, with an adaptive gate.

Six specialist branches (MA, HE, EX-H, EX-S, NV, ME) share a shallow stem and
each emit
    * a lesion evidence map  (1, S, S)  - supervised by real lesion annotations
      when the encoder has been lesion-pretrained, by the weak morphological
      priors otherwise
    * a descriptor vector    (D,)       - pooled expert feature

What changed is the *routing*.  The gate used to see only the stem features,
so it produced essentially the same mixture for every image of a given
appearance.  It now conditions on four contexts:

    alpha_k = Softmax( G(Z_G, Z_L, Q, H) )
    Z_MoE   = sum_k alpha_k E_k(I)

  Z_G  global retinal features from the RETFound backbone - lets the gate know
       what kind of retina this is before deciding which specialist to trust
  Z_L  the pooled lesion-stem features
  Q    the five A1 quality axes - on a low lesion-visibility image the gate
       should lean on the large-scale experts (HE, NV, ME) rather than pretend
       it can see microaneurysms
  H    the sample's hard-example history, so images this model keeps getting
       wrong get a mixture chosen with that knowledge instead of the average

The expert set and the expert architecture are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MoECfg


@dataclass
class MoEOutput:
    z_lesion: torch.Tensor        # (B, lesion_dim)
    alphas: torch.Tensor          # (B, n_experts)
    evidence: torch.Tensor        # (B, n_experts, S, S) logits
    presence: torch.Tensor        # (B, n_experts) logits
    expert_tokens: torch.Tensor   # (B, n_experts, expert_dim)


def _conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class LesionExpert(nn.Module):
    """One lesion specialist: dilated convs keep small lesions resolvable."""

    def __init__(self, cin: int, dim: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, dim, 3, 1, dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.seg = nn.Conv2d(dim, 1, 1)          # lesion evidence map
        self.cls = nn.Linear(dim, 1)             # image-level presence

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(x)
        evid = self.seg(h)                        # (B,1,S,S) logits
        # attention-pool the feature using the expert's own evidence map
        w = torch.sigmoid(evid)
        feat = (h * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-6)
        return evid, feat, self.cls(feat)


class AdaptiveGate(nn.Module):
    """G(Z_G, Z_L, Q, H) -> mixture weights over the lesion experts."""

    def __init__(self, cfg: MoECfg, n_experts: int, stem_dim: int,
                 global_dim: int, n_quality: int = 5):
        super().__init__()
        self.cfg = cfg
        ctx = set(cfg.gate_context)
        self.use_global = "global" in ctx
        self.use_quality = "quality" in ctx
        self.use_hard = "hardness" in ctx

        d_in = stem_dim                                    # Z_L, always present
        self.g_proj = None
        if self.use_global:
            self.g_proj = nn.Sequential(nn.Linear(global_dim, cfg.gate_hidden),
                                        nn.SiLU(inplace=True))
            d_in += cfg.gate_hidden
        if self.use_quality:
            d_in += n_quality
        if self.use_hard:
            d_in += 1

        self.mlp = nn.Sequential(
            nn.Linear(d_in, cfg.gate_hidden), nn.SiLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(cfg.gate_hidden, n_experts),
        )

    def forward(self, z_stem: torch.Tensor, z_global: torch.Tensor | None,
                quality: torch.Tensor | None, hardness: torch.Tensor | None
                ) -> torch.Tensor:
        B = z_stem.shape[0]
        parts = [z_stem]
        if self.use_global:
            if z_global is None:
                z_global = z_stem.new_zeros(B, self.g_proj[0].in_features)
            parts.append(self.g_proj(z_global.to(z_stem.dtype)))
        if self.use_quality:
            q = quality if quality is not None else z_stem.new_zeros(B, 5)
            parts.append(q.to(z_stem.dtype))
        if self.use_hard:
            h = hardness if hardness is not None else z_stem.new_zeros(B, 1)
            parts.append(h.to(z_stem.dtype).view(B, 1))
        logits = self.mlp(torch.cat(parts, dim=-1))
        return torch.softmax(logits / max(self.cfg.gate_temperature, 1e-3), dim=-1)


class LesionMoE(nn.Module):
    def __init__(self, cfg: MoECfg, in_ch: int = 3, global_dim: int = 1024):
        super().__init__()
        self.cfg = cfg
        self.names = tuple(cfg.experts)
        n = len(self.names)
        d = cfg.expert_dim

        self.stem = nn.Sequential(
            _conv_block(in_ch, 32, stride=2),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 96),
        )
        # different dilations = different receptive fields per lesion scale:
        # MA is a few pixels, NV/ME are broad patterns
        dilations = {"MA": 1, "HE": 2, "EX_H": 1, "EX_S": 2, "NV": 3, "ME": 4}
        self.experts = nn.ModuleList(
            [LesionExpert(96, d, dilations.get(nm, 1)) for nm in self.names])

        self.gate = AdaptiveGate(cfg, n, stem_dim=96, global_dim=global_dim)
        self.proj = nn.Sequential(
            nn.Linear(d + n, cfg.lesion_dim), nn.LayerNorm(cfg.lesion_dim),
            nn.SiLU(inplace=True),
        )

    def encoder_parameters(self):
        """Stem + experts: the part transferred from lesion pretraining."""
        return list(self.stem.parameters()) + list(self.experts.parameters())

    def forward(self, x: torch.Tensor, z_global: torch.Tensor | None = None,
                quality: torch.Tensor | None = None,
                hardness: torch.Tensor | None = None) -> MoEOutput:
        s = self.stem(x)
        if s.shape[-1] != self.cfg.mask_size:
            s = F.interpolate(s, size=(self.cfg.mask_size, self.cfg.mask_size),
                              mode="bilinear", align_corners=False)
        evid, feats, pres = [], [], []
        for ex in self.experts:
            e, f, p = ex(s)
            evid.append(e); feats.append(f); pres.append(p)
        evidence = torch.cat(evid, 1)                      # (B,n,S,S)
        tokens = torch.stack(feats, 1)                     # (B,n,D)
        presence = torch.cat(pres, 1)                      # (B,n)

        z_stem = s.mean((2, 3))                            # (B,96) = Z_L context
        alphas = self.gate(z_stem, z_global, quality, hardness)
        pooled = (tokens * alphas.unsqueeze(-1)).sum(1)    # (B,D)
        z = self.proj(torch.cat([pooled, alphas], dim=-1))
        return MoEOutput(z_lesion=z, alphas=alphas, evidence=evidence,
                         presence=presence, expert_tokens=tokens)


def lesion_supervision_loss(out: MoEOutput, target_masks: torch.Tensor,
                            target_presence: torch.Tensor,
                            valid: torch.Tensor | None = None) -> torch.Tensor:
    """Dice + positive-weighted BCE on the maps, BCE on image-level presence.

    Lesion pixels are a tiny minority, so the Dice term is what actually drives
    learning; plain BCE alone collapses to an all-background prediction.

    `valid` is a (B,) or (B,n) mask marking which supervision entries are real.
    With mixed batches - real IDRiD/DDR/FGADR annotations for some images, weak
    morphological priors for others - this is what stops a pseudo-label from
    being scored as if it were ground truth.
    """
    prob = torch.sigmoid(out.evidence)
    t = target_masks.to(prob.dtype)
    if t.shape[-2:] != prob.shape[-2:]:
        t = F.interpolate(t, size=prob.shape[-2:], mode="bilinear", align_corners=False)
    dims = (2, 3)
    inter = (prob * t).sum(dims)
    dice = 1.0 - (2 * inter + 1.0) / (prob.sum(dims) + t.sum(dims) + 1.0)

    # Positive-weighted BCE keeps the rare lesion pixels from being ignored.
    # The cap matters more than it looks: microaneurysms occupy ~0.15% of the
    # evidence grid, so the balancing weight should be ~665.  A flat clamp at 50
    # under-weights MA positives by an order of magnitude and the channel never
    # leaves the all-background solution, while HE (~1.5%, ideal weight ~67) sits
    # just above the cap and does learn.  Capping by lesion *scale* instead lets
    # the rare channels carry their true weight without letting a single stray
    # annotated pixel produce an unbounded gradient.
    pos = t.sum(dims, keepdim=True)
    tot = t.shape[-1] * t.shape[-2]
    pw = ((tot - pos) / (pos + 1.0)).clamp(1.0, 1000.0)
    bce = F.binary_cross_entropy_with_logits(out.evidence, t, weight=pw,
                                             reduction="none").mean(dims)

    pres = F.binary_cross_entropy_with_logits(
        out.presence, target_presence.to(prob.dtype), reduction="none")

    per_sample = dice + 0.5 * bce + pres                    # (B, n)
    if valid is None:
        return per_sample.mean()
    v = valid.to(per_sample.dtype)
    if v.dim() == 1:
        v = v.unsqueeze(1).expand_as(per_sample)
    return (per_sample * v).sum() / v.sum().clamp_min(1.0)
