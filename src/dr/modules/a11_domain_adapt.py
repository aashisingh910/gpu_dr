"""A11 - DAFA: Domain-Adversarial Feature Alignment.

DR-NOVA addition, attacking the domain-generalisation weakness measured
directly in the project report: EyePACS QWK 0.5944 drops to APTOS QWK 0.3088
under external-domain shift - a substantial, measured collapse, not a
hypothetical concern.

Builds on Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation by
Backpropagation" (DANN): a gradient-reversal layer (GRL) sits between the
fused representation and a small domain classifier. The domain classifier is
trained normally to tell EyePACS and external-domain (e.g. APTOS) images
apart; the GRL flips the sign of that gradient on the way back into the
fusion representation, so the FEATURES are pushed to become domain-invariant
even though the classifier itself is trained to succeed.

The modification for this project: DANN is domain-agnostic by design (it
only asks "which domain is this from"); DAFA gates the adversarial signal by
the same per-image quality/domain-typicality score A1 already computes
(Q_domain), so a low-quality or atypical image - where the domain label
itself is least reliable - contributes less to the alignment gradient rather
than being weighted identically to a clean, well-characterised image.

Usage is opt-in and degrades to a complete no-op when no domain label is
available: `batch["domain_id"]` (0 = EyePACS / in-domain, 1 = external, ...)
must be supplied by the training loop for this module to receive gradient at
all. Mixing in a labelled external corpus (e.g. a stratified APTOS subset)
during training is what actually exercises it - training on EyePACS alone
leaves this module present but idle.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class DomainAdversarialHead(nn.Module):
    """Small MLP domain classifier fed through a gradient-reversal layer.

    `n_domains`: 2 is the minimum useful case (in-domain vs. one external
    corpus); pass more if training spans several labelled sources at once.
    """

    def __init__(self, in_dim: int, n_domains: int = 2, hidden: int = 128,
                dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, z_fused: torch.Tensor, lambd: float,
               domain_weight: torch.Tensor | None = None) -> torch.Tensor:
        """Returns domain logits (B, n_domains). `domain_weight` (B,) in
        [0,1], typically A1's Q_domain, scales each sample's contribution to
        the reversed gradient by rescaling its activations before the GRL -
        a near-zero-weight sample still gets a forward pass (so batch norm /
        loss bookkeeping stay simple) but contributes almost no gradient.
        """
        z = z_fused
        if domain_weight is not None:
            z = z * domain_weight.clamp(0, 1).unsqueeze(-1)
        return self.net(grad_reverse(z, lambd))


def dafa_loss(domain_logits: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy on the domain classifier's own output. The
    domain-invariance effect comes entirely from the GRL's backward pass,
    not from anything unusual in this loss itself.
    """
    import torch.nn.functional as F
    return F.cross_entropy(domain_logits, domain_id.long())


def dafa_lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """The DANN paper's standard ramp: lambda rises smoothly from 0 to 1
    across training progress in [0,1], so the adversarial signal doesn't
    dominate before the backbone has learned anything useful to align.
    lambda(p) = 2 / (1 + exp(-gamma*p)) - 1
    """
    import math
    p = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
