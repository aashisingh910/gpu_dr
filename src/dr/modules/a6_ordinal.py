"""A6 - Hierarchical Ordinal DR Grading + boundary + contrastive objectives.

DR severity is ordinal (0 No DR < 1 Mild < 2 Moderate < 3 Severe < 4 PDR), so
a plain 5-way softmax throws away the ordering and treats "predict 0 when the
truth is 4" as no worse than "predict 3 when the truth is 4".

The CORAL rank head is unchanged: one shared projection with monotonically
decreasing thresholds, giving

    P(Y>0) >= P(Y>1) >= P(Y>2) >= P(Y>3)      (guaranteed by construction)

Monotonicity is enforced by parameterising the thresholds as a cumulative sum
of soft-plus increments, so the derived categorical distribution
P(Y=k) = P(Y>k-1) - P(Y>k) can never go negative.

What is added on top:

  * Hierarchical clinical heads.  The four cumulative probabilities are also
    supervised *as the clinical decisions they are* - any DR, referable DR,
    sight-threatening DR, PDR - each with its own head and its own loss.  A
    single 5-class objective spends most of its gradient separating No-DR from
    Moderate, which is easy, and almost none on Mild-vs-Moderate, which is the
    boundary screening actually turns on.

  * Boundary loss.  L_boundary = |y - yhat|^gamma on the expected grade, where
    yhat is the continuous severity sum_k k P(Y=k).  Cross-entropy is flat in
    the distance of the error; this term is not, so a two-grade miss costs
    strictly more than a one-grade miss.

  * Supervised contrastive loss on a projection of Z_MC.  Pulls same-grade
    samples together and pushes different-grade samples apart *in proportion
    to their grade distance*, which gives the minority grades a training
    signal that does not depend on them winning an argmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OrdinalOutput:
    cum_logits: torch.Tensor     # (B, K-1)  logits of P(Y>k)
    cum_probs: torch.Tensor      # (B, K-1)  monotone non-increasing
    class_probs: torch.Tensor    # (B, K)    derived categorical distribution
    logits_ce: torch.Tensor      # (B, K)    parallel softmax head (for L_CBF)
    expected_grade: torch.Tensor # (B,)      sum_k k*P(Y=k), a continuous severity
    hier_logits: torch.Tensor    # (B, H)    the explicit hierarchical heads
    projection: torch.Tensor     # (B, proj) L2-normalised contrastive embedding


class OrdinalHead(nn.Module):
    def __init__(self, dim: int, n_grades: int = 5, dropout: float = 0.1,
                 hierarchy_thresholds: tuple = (1, 2, 3, 4), proj_dim: int = 128):
        super().__init__()
        self.K = n_grades
        self.thresholds_h = tuple(hierarchy_thresholds)
        self.pre = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout))
        self.rank = nn.Linear(dim, 1, bias=False)      # shared CORAL projection
        self.b0 = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.full((n_grades - 2,), 0.5))
        self.ce = nn.Linear(dim, n_grades)
        # one head per clinical decision: any DR / referable / STDR / PDR
        self.hier = nn.Linear(dim, len(self.thresholds_h))
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, proj_dim))

    def thresholds(self) -> torch.Tensor:
        """b_0 > b_1 > ... > b_{K-2}, strictly decreasing."""
        steps = F.softplus(self.deltas)
        return torch.cat([self.b0, self.b0 - torch.cumsum(steps, 0)])

    @torch.no_grad()
    def init_from_prior(self, class_counts: torch.Tensor) -> torch.Tensor:
        """Re-initialise the ladder at the training class prior instead of a
        data-independent uniform spacing.

        Objective 1's evidence pack found that with the default init
        (b0=0, deltas=0.5 -> a uniform ladder of gap ~0.974), the thresholds
        barely move during training (drift ~1e-4, three orders of magnitude
        below what recovering Mild/Severe recall would require), so every
        interior grade is boxed into the same narrow, symmetric slice of the
        shared rank z regardless of how rare that grade actually is.

        This sets b_k = logit(P(Y>k)) on the *training* class frequencies
        (Menon et al. 2021's logit adjustment, applied as an initialisation
        rather than a post-hoc correction): the ladder starts already shaped
        to the prior, so Mild's tiny share of z is a good starting guess
        instead of something a barely-moving gradient has to discover.

        `class_counts`: (K,) counts per grade on the training split, in
        grade order [n_0, n_1, ..., n_{K-1}].
        Returns the new threshold vector (for logging).
        """
        counts = class_counts.to(torch.float64).clamp_min(1.0)
        total = counts.sum()
        K = counts.shape[0]
        # P(Y>k) for k = 0..K-2  =  sum of counts strictly above k, over total
        tail = torch.tensor([counts[k + 1:].sum() for k in range(K - 1)],
                            dtype=torch.float64, device=counts.device)
        p_gt = (tail / total).clamp(1e-4, 1 - 1e-4)
        b_target = torch.log(p_gt / (1 - p_gt))          # logit(P(Y>k)), strictly decreasing

        self.b0.copy_(b_target[:1].to(self.b0.dtype))
        steps = (b_target[:-1] - b_target[1:]).clamp_min(1e-4)   # > 0 by construction
        # invert softplus: softplus(delta) = step  =>  delta = log(exp(step) - 1)
        inv_softplus = torch.log(torch.expm1(steps))
        self.deltas.copy_(inv_softplus.to(self.deltas.dtype))
        return self.thresholds().detach()

    def forward(self, x: torch.Tensor) -> OrdinalOutput:
        h = self.pre(x)
        z = self.rank(h)                               # (B,1)
        cum_logits = z + self.thresholds().unsqueeze(0)  # (B,K-1)
        cum = torch.sigmoid(cum_logits)

        # P(Y=0)=1-P(Y>0); P(Y=k)=P(Y>k-1)-P(Y>k); P(Y=K-1)=P(Y>K-2)
        ones = torch.ones_like(cum[:, :1])
        zeros = torch.zeros_like(cum[:, :1])
        upper = torch.cat([ones, cum], dim=1)
        lower = torch.cat([cum, zeros], dim=1)
        probs = (upper - lower).clamp_min(1e-8)
        probs = probs / probs.sum(1, keepdim=True)

        grades = torch.arange(self.K, device=x.device, dtype=probs.dtype)
        return OrdinalOutput(cum_logits=cum_logits, cum_probs=cum,
                             class_probs=probs, logits_ce=self.ce(h),
                             expected_grade=(probs * grades).sum(1),
                             hier_logits=self.hier(h),
                             projection=F.normalize(self.proj(h), dim=-1))


def coral_decode(cum_probs: torch.Tensor) -> torch.Tensor:
    """Canonical CORAL prediction: how many thresholds the shared rank exceeds.

        yhat = sum_k  1[ P(Y>k) > 0.5 ]

    This is NOT the same as argmax over the derived categorical distribution,
    and the difference is not cosmetic.  With a single shared rank z and
    thresholds t_k, the interior classes are differences of sigmoids

        P(Y=k) = sigmoid(z + t_{k-1}) - sigmoid(z + t_k)

    so P(Y=k) is bounded above by tanh((t_{k-1} - t_k)/4), while the two
    endpoint classes P(Y=0) and P(Y=K-1) are unbounded and can reach 1.0.  When
    the learned threshold gaps are small the interior classes can never win an
    argmax at any z, and their recall is pinned at exactly zero no matter how
    the sampler or the loss is re-weighted.  Measured on this model the gaps are
    ~0.974, capping the interior classes at P = 0.239 and making Mild and Severe
    unreachable - which is precisely the observed per-grade recall
    [0.82, 0.000, 0.153, 0.000, 0.833].

    Decoding by rank instead makes every grade reachable and costs nothing: it
    reuses the same monotone cumulative probabilities the model already emits.
    """
    return (cum_probs > 0.5).sum(1).long()


def coral_decode_numpy(class_probs):
    """CORAL rank decoding from a derived categorical distribution.

    P(Y>k) = sum_{j>k} P(Y=j), so the cumulative probabilities - and therefore
    the rank prediction - can be recovered from `class_probs` alone.  That means
    an already-trained checkpoint can be re-decoded without retraining.
    """
    import numpy as _np
    probs = _np.asarray(class_probs, dtype=_np.float64)
    rc = _np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]      # rc[:, j] = P(Y>=j)
    cum = rc[:, 1:]                                        # P(Y>k), k = 0..K-2
    return (cum > 0.5).sum(1).astype(_np.int64)


def coral_targets(y: torch.Tensor, n_grades: int) -> torch.Tensor:
    """y -> binary rank targets [y>0, y>1, y>2, y>3]."""
    ks = torch.arange(n_grades - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > ks).float()


def dqk_loss(class_probs: torch.Tensor, y: torch.Tensor, n_grades: int,
            eps: float = 1e-6) -> torch.Tensor:
    """DQK-Loss - Differentiable Quadratic-weighted Kappa loss.

    DR-NOVA addition, attacking the "five-class severity grading is too low"
    weakness directly: every other loss term in this file (ordinal BCE,
    hierarchical, boundary, class-balanced focal) is a proxy for QWK, not
    QWK itself, so nothing before this term ever received gradient signal
    shaped like the metric the whole objective is actually judged on.

    Builds on the soft/weighted-kappa relaxations used in ordinal Kaggle
    competitions and formalised by de la Torre, Puig & Valls (2018,
    "Weighted kappa loss function for multi-class classification of ordinal
    data in deep learning") - the modification here is using this project's
    OWN `class_probs` (CORAL's derived categorical distribution, not a plain
    softmax) as the soft prediction, so the surrogate is differentiated with
    respect to the same rank-monotone parameterisation PI-CORAL already
    enforces, rather than an unconstrained softmax that could fight it.

    O_ij (soft observed) = sum_n 1[y_n=i] * p_n[j]      - a soft confusion matrix
    E_ij (expected)      = row_marginal_i * col_marginal_j / N
    w_ij                 = (i-j)^2 / (K-1)^2             - quadratic weights
    loss = sum_ij w_ij O_ij / (sum_ij w_ij E_ij + eps)   - minimised as QWK -> 1
    """
    K = n_grades
    y1h = F.one_hot(y, K).float()                          # (B, K)
    O = y1h.t() @ class_probs                               # (K, K) soft confusion
    row_marg = y1h.sum(0)                                   # (K,)
    col_marg = class_probs.sum(0)                            # (K,)
    N = class_probs.shape[0]
    E = torch.outer(row_marg, col_marg) / max(N, 1)
    idx = torch.arange(K, device=class_probs.device, dtype=class_probs.dtype)
    w = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2 / max((K - 1) ** 2, 1)
    return (w * O).sum() / ((w * E).sum() + eps)


def ordinal_loss(out: OrdinalOutput, y: torch.Tensor,
                 rank_weights: torch.Tensor | None = None) -> torch.Tensor:
    t = coral_targets(y, out.class_probs.shape[1])
    loss = F.binary_cross_entropy_with_logits(out.cum_logits, t, reduction="none")
    if rank_weights is not None:
        loss = loss * rank_weights.unsqueeze(0)
    return loss.mean()


def hierarchical_loss(out: OrdinalOutput, y: torch.Tensor,
                      thresholds: tuple = (1, 2, 3, 4),
                      pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Binary DR / referable DR / STDR / PDR, each as its own decision.

    Every threshold is positive-weighted by its own prevalence in the batch, so
    PDR - which is ~2% of EyePACS - is not drowned out by the any-DR head.
    """
    ks = torch.as_tensor(thresholds, device=y.device).unsqueeze(0)   # (1,H)
    t = (y.unsqueeze(1) >= ks).float()                               # (B,H)
    if pos_weight is None:
        pos = t.sum(0)
        neg = t.shape[0] - pos
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.5, 20.0)
    return F.binary_cross_entropy_with_logits(
        out.hier_logits, t, pos_weight=pos_weight.to(out.hier_logits.dtype))


def boundary_loss(out: OrdinalOutput, y: torch.Tensor,
                  gamma: float = 1.5) -> torch.Tensor:
    """L_boundary = |y - yhat|^gamma on the continuous expected grade.

    gamma > 1 makes the penalty super-linear in the size of the miss, which is
    what pushes probability mass across an adjacent-class boundary instead of
    letting it hedge between two neighbours.
    """
    err = (out.expected_grade - y.to(out.expected_grade.dtype)).abs()
    return (err.clamp_min(1e-6) ** gamma).mean()


def ordinal_contrastive_loss(projection: torch.Tensor, y: torch.Tensor,
                             temperature: float = 0.1,
                             distance_weighted: bool = True) -> torch.Tensor:
    """Supervised contrastive loss, weighted by ordinal grade distance.

    Standard SupCon treats every negative equally, which for an ordinal target
    is wrong: a Moderate pushed away from a Severe should be pushed less hard
    than a Moderate pushed away from a No-DR.  Scaling each negative by
    |y_i - y_j| encodes the severity scale directly into the embedding.
    """
    B = projection.shape[0]
    if B < 2:
        return projection.sum() * 0.0
    sim = projection @ projection.t() / max(temperature, 1e-6)
    eye = torch.eye(B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    y = y.view(-1, 1)
    pos = (y == y.t()) & ~eye                                        # (B,B)
    if pos.sum() == 0:
        return projection.sum() * 0.0

    if distance_weighted:
        dist = (y - y.t()).abs().to(sim.dtype)
        # negatives further away in grade get a larger weight in the denominator
        w = torch.where(pos, torch.ones_like(dist), 1.0 + dist)
        logits = sim + w.clamp_min(1e-6).log()
    else:
        logits = sim

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    keep = n_pos > 0
    # Select rather than multiply: the self-similarity masked to -inf above
    # leaves -inf on the diagonal of log_prob, and -inf * 0.0 is NaN, which
    # would poison every row of the sum (and with it the whole objective).
    masked_lp = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    mean_lp = masked_lp.sum(1)[keep] / n_pos[keep]
    return -mean_lp.mean()


def class_balanced_focal_loss(logits: torch.Tensor, y: torch.Tensor,
                              class_counts: torch.Tensor, beta: float = 0.999,
                              gamma: float = 2.0,
                              weight_power: float | None = None) -> torch.Tensor:
    """Cui et al. class-balanced weighting on top of a focal loss.

    `weight_power` moderates the weighting: with an aggressive n^-0.5 sampler
    already in place, stacking full effective-number weights on top double-counts
    the imbalance correction and destabilises the minority classes.  Passing e.g.
    -0.25 uses n^-0.25 weights instead of the full (1-beta^n)/(1-beta) scheme.
    """
    if weight_power is not None:
        w = class_counts.clamp_min(1.0) ** weight_power
    else:
        eff = (1.0 - torch.pow(beta, class_counts.clamp_min(1.0))) / (1.0 - beta)
        w = 1.0 / eff
    w = w / w.sum() * len(class_counts)
    w = w.to(logits.device, logits.dtype)

    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    logp_t = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
    focal = -((1 - p_t) ** gamma) * logp_t
    return (focal * w[y]).mean()
