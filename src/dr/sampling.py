"""Class re-balancing and hard-example mining.

The previous setup stacked two aggressive corrections on top of each other:
full inverse-frequency sampling *and* class-balanced focal weighting.  Together
they over-correct - grade 4, which is ~2% of EyePACS, ends up dominating the
effective gradient, the model over-predicts severity, and the middle grades
(Mild / Moderate) collapse because nothing is specifically asking for them.

This module replaces that with a moderate prior plus a signal that actually
targets the failing boundaries:

    P_i  proportional to  n_{class(i)}^power  *  min( (1 + H_i)^eta, clip )

  power = -0.5   n^-0.5 instead of n^-1: still lifts the minority classes, but
                 leaves grade 0 with real presence in every epoch
  H_i = L_i / (mean L + eps)   per-sample loss, normalised by the epoch mean, so
                 the scale is stable as training progresses
  eta = 0.5      how hard the mining pulls

On top of the loss-based term, two targeted bonuses:

  * boundary errors - a prediction one grade off the truth, i.e. exactly the
    No-DR/Mild, Mild/Moderate, Moderate/Severe, Severe/PDR confusions the model
    keeps making, gets extra weight
  * low-confidence samples - a small margin between the top two classes means
    the model is on the fence, and those samples carry the most information

H is kept as an EMA across epochs so a single noisy batch cannot spike a
sample's sampling probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from .config import SamplingCfg


@dataclass
class HardExampleState:
    """Per-sample mining state, indexed by dataset meta row."""
    hardness: np.ndarray        # EMA of L_i / mean(L)
    seen: np.ndarray            # how many times each sample has been scored
    boundary: np.ndarray        # EMA of "was an adjacent-class error"
    lowconf: np.ndarray         # EMA of "was a low-margin prediction"

    @classmethod
    def empty(cls, n: int) -> "HardExampleState":
        return cls(np.ones(n, np.float32), np.zeros(n, np.int32),
                   np.zeros(n, np.float32), np.zeros(n, np.float32))


def update_hardness(state: HardExampleState, meta_idx: np.ndarray,
                    losses: np.ndarray, preds: np.ndarray, targets: np.ndarray,
                    margins: np.ndarray, cfg: SamplingCfg) -> HardExampleState:
    """Fold one epoch's per-sample statistics into the mining state.

    The sampler draws **with replacement**, so a hard sample legitimately
    appears several times in one epoch.  Plain fancy-index assignment would
    keep only the last occurrence - the noisiest possible estimate of that
    sample's loss - and `seen[idx] += 1` would count the whole run of
    duplicates as a single observation.  Duplicates are therefore averaged with
    `np.add.at`, which accumulates instead of overwriting.
    """
    meta_idx = np.asarray(meta_idx)
    losses = np.asarray(losses, np.float64)
    mean_l = float(losses.mean()) if losses.size else 0.0

    err = np.abs(np.asarray(preds) - np.asarray(targets))
    raw = {
        "hardness": losses / (mean_l + cfg.hard_eps),
        "boundary": (err == 1).astype(np.float64),
        "lowconf": (np.asarray(margins) < 0.15).astype(np.float64),
    }

    n = len(state.hardness)
    counts = np.zeros(n, np.float64)
    np.add.at(counts, meta_idx, 1.0)
    touched = counts > 0
    first = touched & (state.seen == 0)

    a = cfg.hard_ema
    for name, values in raw.items():
        acc = np.zeros(n, np.float64)
        np.add.at(acc, meta_idx, values)
        mean_new = np.divide(acc, counts, out=np.zeros(n, np.float64), where=touched)
        arr = getattr(state, name)
        blended = a * arr + (1 - a) * mean_new
        # a sample scored for the first time takes the new value outright;
        # blending it against the initialisation would wash out the signal
        arr[touched] = np.where(first[touched], mean_new[touched], blended[touched])

    state.seen += counts.astype(np.int32)
    return state


def sampling_weights(labels: np.ndarray, meta_idx: np.ndarray,
                     state: HardExampleState | None, cfg: SamplingCfg,
                     n_grades: int = 5) -> np.ndarray:
    """P_i proportional to n_class^power * (1 + H_i)^eta, with the two bonuses."""
    counts = np.bincount(np.asarray(labels), minlength=n_grades).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts[labels] ** cfg.power

    if state is not None:
        h = state.hardness[meta_idx].astype(np.float64)
        mine = np.minimum((1.0 + h) ** cfg.hard_eta, cfg.hard_clip)
        mine *= 1.0 + cfg.boundary_bonus * state.boundary[meta_idx]
        mine *= 1.0 + cfg.lowconf_bonus * state.lowconf[meta_idx]
        w = w * mine
    return (w / w.sum()).astype(np.float64)


def make_sampler(labels: np.ndarray, meta_idx: np.ndarray,
                 state: HardExampleState | None, cfg: SamplingCfg,
                 num_samples: int | None = None) -> WeightedRandomSampler:
    w = sampling_weights(labels, meta_idx, state, cfg)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=num_samples or len(labels),
                                 replacement=True)


def class_loss_weights(counts: np.ndarray, power: float,
                       n_grades: int = 5) -> torch.Tensor:
    """Moderate per-class loss weighting, normalised to mean 1."""
    c = np.asarray(counts, np.float64).copy()
    c[c == 0] = 1.0
    w = c ** power
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32)


def mining_report(state: HardExampleState, labels: np.ndarray,
                  meta_idx: np.ndarray, n_grades: int = 5) -> dict:
    """What the miner is currently pushing, so the effect is visible in logs."""
    h = state.hardness[meta_idx]
    out = {"mean_hardness": float(h.mean()),
           "p90_hardness": float(np.percentile(h, 90)) if h.size else 0.0,
           "boundary_rate": float(state.boundary[meta_idx].mean()),
           "lowconf_rate": float(state.lowconf[meta_idx].mean())}
    for g in range(n_grades):
        m = np.asarray(labels) == g
        out[f"hardness_grade_{g}"] = float(h[m].mean()) if m.any() else float("nan")
    return out
