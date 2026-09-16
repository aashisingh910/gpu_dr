"""A8 - Multimodal Clinical Fusion.

    Z_F (image)  --.
                   +--> cross-attention / gated fusion --> Z_MC
    C (clinical) --'

Clinical features are age, diabetes duration, HbA1c, systolic/diastolic BP,
treatment (insulin/oral/none) and prior-DR flag.  The gate is what makes this
safe when clinical data is missing: a learned scalar g in [0,1] decides how
much clinical evidence to admit, and a `valid` mask forces g -> 0 for records
with no metadata, so the model degrades to image-only rather than consuming a
vector of imputed zeros as if it were real measurement.

DATA CAVEAT: public EyePACS ships no clinical metadata.  Training on EyePACS
runs with `valid=0` for every record (image-only path).  Supply a CSV via
`--clinical-csv` keyed on image id to activate this block for real.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

CLINICAL_FIELDS = ("age", "duration_years", "hba1c", "bp_systolic",
                   "bp_diastolic", "on_insulin", "on_oral_agent", "prior_dr")


@dataclass
class MultimodalOutput:
    z_mc: torch.Tensor        # (B, D) joint representation
    gate: torch.Tensor        # (B, 1) how much clinical signal was admitted


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class MultimodalFusion(nn.Module):
    def __init__(self, img_dim: int, clinical_dim: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.enc = ClinicalEncoder(clinical_dim, img_dim, dropout)
        self.attn = nn.MultiheadAttention(img_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(img_dim)
        self.gate = nn.Sequential(nn.Linear(img_dim * 2, img_dim), nn.GELU(),
                                  nn.Linear(img_dim, 1), nn.Sigmoid())
        self.out = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, img_dim),
                                 nn.GELU())

    def forward(self, z_img: torch.Tensor, clinical: torch.Tensor,
                valid: torch.Tensor) -> MultimodalOutput:
        """z_img (B,D); clinical (B,F); valid (B,1) 1.0 if metadata is real."""
        c = self.enc(clinical).unsqueeze(1)                  # (B,1,D)
        q = z_img.unsqueeze(1)
        a, _ = self.attn(q, c, c, need_weights=False)
        a = self.norm(a.squeeze(1))
        g = self.gate(torch.cat([z_img, a], dim=-1)) * valid  # hard-zero if absent
        return MultimodalOutput(z_mc=self.out(z_img + g * a), gate=g)
