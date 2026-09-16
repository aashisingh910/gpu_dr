"""A7 - Temporal Prognostic Engine (RETFound Plus Temporal).

Longitudinal visits of the same patient (I_1..I_T with times t_1..t_T) are
encoded by the shared image encoder, given a continuous time embedding, run
through a temporal transformer with a causal mask, and read out into
progression-risk logits at 6 / 12 / 24 / 60 months plus a discrete-time
survival head.

    Loss = L_BCE (per-horizon risk) + L_survival (discrete-time hazard NLL)

DATA CAVEAT
-----------
EyePACS is **cross-sectional**: one visit per patient (two images, left/right
eye), with no follow-up grades.  There is therefore no real longitudinal
supervision available in this dataset, and this module is deliberately kept
inert during EyePACS training - `train.py` never fires the prognosis loss and
`evaluate.py` reports the prognosis metric group as NOT EVALUATED rather than
inventing numbers.

To train it for real, point `--longitudinal-csv` at a cohort with
(patient_id, visit_date, image_path, grade) rows - e.g. an internal PACS
export or the AREDS/DRCR longitudinal sets.  The code path is complete and
unit-tested via `scripts/08_selftest.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalOutput:
    risk_logits: torch.Tensor      # (B, H) progression risk per horizon
    hazard_logits: torch.Tensor    # (B, H) discrete-time hazards
    survival: torch.Tensor         # (B, H) S(t_h) = prod (1 - hazard)
    sequence_repr: torch.Tensor    # (B, D)


class TimeEmbedding(nn.Module):
    """Continuous-time sinusoidal embedding (months since baseline)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, -6, half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t_months: torch.Tensor) -> torch.Tensor:
        x = t_months.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class TemporalPrognosticEngine(nn.Module):
    def __init__(self, dim: int, horizons: tuple[int, ...] = (6, 12, 24, 60),
                 n_layers: int = 2, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.horizons = horizons
        self.time_emb = TimeEmbedding(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.risk = nn.Linear(dim, len(horizons))
        self.hazard = nn.Linear(dim, len(horizons))

    def forward(self, visit_feats: torch.Tensor, times_months: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> TemporalOutput:
        """visit_feats (B,T,D), times_months (B,T), pad_mask (B,T) True=pad."""
        x = visit_feats + self.time_emb(times_months)
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        if pad_mask is not None:                      # take the last real visit
            lengths = (~pad_mask).sum(1).clamp_min(1) - 1
            last = h[torch.arange(h.shape[0], device=h.device), lengths]
        else:
            last = h[:, -1]

        hz = self.hazard(last)
        surv = torch.cumprod(1.0 - torch.sigmoid(hz), dim=1)
        return TemporalOutput(risk_logits=self.risk(last), hazard_logits=hz,
                              survival=surv, sequence_repr=last)


def progression_loss(out: TemporalOutput, progressed: torch.Tensor,
                     event_horizon_idx: torch.Tensor,
                     observed: torch.Tensor) -> torch.Tensor:
    """L_BCE + L_survival.

    progressed        (B,H) 1 if the patient had progressed by horizon h
    event_horizon_idx (B,)  index of the horizon at which the event occurred
    observed          (B,)  1 = event observed, 0 = right-censored
    """
    bce = F.binary_cross_entropy_with_logits(out.risk_logits, progressed.float())

    # discrete-time survival NLL: hazard at the event bin, survival before it
    B, H = out.hazard_logits.shape
    idx = torch.arange(H, device=out.hazard_logits.device).unsqueeze(0)
    ev = event_horizon_idx.unsqueeze(1)
    log_h = F.logsigmoid(out.hazard_logits)
    log_1mh = F.logsigmoid(-out.hazard_logits)
    before = (idx < ev).float() * log_1mh
    at = (idx == ev).float() * (observed.unsqueeze(1) * log_h
                                + (1 - observed.unsqueeze(1)) * log_1mh)
    surv_nll = -(before + at).sum(1).mean()
    return bce + surv_nll
