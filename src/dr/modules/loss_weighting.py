"""Learned multi-task loss weighting (Kendall, Gal & Cipolla 2018).

`LossCfg` currently hand-fixes seven weights (w_ordinal=1, w_hierarchical=0.5,
w_boundary=0.2, w_contrastive=0.2, w_lesion=0.3, w_pa=0.1, w_cbf=0.3). These
were picked once and never re-derived, and the loss terms live on very
different natural scales (a Dice-based term saturates near 1, a contrastive
InfoNCE term can be an order of magnitude larger early in training), so a
fixed weight is implicitly over- or under-emphasising whichever term happens
to be large at initialisation.

Homoscedastic uncertainty weighting replaces each fixed w_i with a learned
per-task log-variance s_i:

    L = sum_i [ exp(-s_i) * L_i + s_i ]

exp(-s_i) shrinks automatically for a task whose loss is already large/noisy
(the model "gives up" trying to force it down further) and grows for a task
that is small/easy to reduce further, with the `+ s_i` term preventing the
degenerate solution of driving every weight to zero. This removes a whole
axis of manual tuning and is the standard fix for exactly the "which lambda
do I pick" problem LossCfg's fixed weights represent.

lambda_XAI is deliberately NOT included here: it is phase-gated (zero in
stages 1-2 by design, see XaiCfg.weight_schedule / xai_weight_for_stage), and
that gating is a scheduling decision, not a scale-balancing one - learning it
would fight the schedule.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HomoscedasticLossWeighting(nn.Module):
    """One learned log-variance per named loss term.

    Usage:
        hw = HomoscedasticLossWeighting(["ord", "hier", "bnd", "con", "les", "pa", "cbf"])
        total, weights = hw(terms)   # terms: dict[str, scalar tensor]
        total = total + w_xai * l_xai   # xai stays on its own schedule
    """

    def __init__(self, names: tuple[str, ...] | list[str]):
        super().__init__()
        self.names = tuple(names)
        # init at 0 -> exp(-0) = 1, i.e. reproduces the old "all weights = 1"
        # starting point before anything has been learned
        self.log_vars = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, losses: dict) -> tuple[torch.Tensor, dict]:
        total = 0.0
        effective_weights = {}
        for i, name in enumerate(self.names):
            if name not in losses:
                continue
            s = self.log_vars[i]
            w = torch.exp(-s)
            total = total + w * losses[name] + s
            effective_weights[name] = float(w.detach())
        return total, effective_weights

    def as_dict(self) -> dict:
        """Current effective weight per term, for logging/checkpointing."""
        with torch.no_grad():
            return {n: float(torch.exp(-self.log_vars[i]))
                   for i, n in enumerate(self.names)}
