"""A10 - Uncertainty & Calibration + the clinical decision gate.

  * Ensemble prediction   P(y|x) via MC-dropout passes
  * Uncertainty           predictive entropy H(x) = -sum p log p, split into
                          aleatoric (mean of per-pass entropies) and epistemic
                          (mutual information) components
  * Calibration           temperature / vector scaling fitted on the validation
                          split by minimising NLL, or isotonic regression
  * Decision gate         low uncertainty + good quality -> autonomous AI decision
                          high uncertainty              -> human review
                          low quality                   -> retake

Calibrators are fitted on **validation** logits only and then applied unchanged
to test, which is what keeps the reported ECE honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UncertaintyOutput:
    probs: torch.Tensor          # (B,K) ensemble mean
    entropy: torch.Tensor        # (B,) normalised predictive entropy [0,1]
    aleatoric: torch.Tensor      # (B,)
    epistemic: torch.Tensor      # (B,) mutual information
    std: torch.Tensor            # (B,) std of the expected grade across passes


def enable_mc_dropout(model: nn.Module) -> None:
    """Put ONLY dropout layers into train mode (BatchNorm stays in eval)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_predict(model: nn.Module, batch: dict, n_samples: int = 10
               ) -> UncertaintyOutput:
    model.eval()
    enable_mc_dropout(model)
    probs, grades = [], []
    for _ in range(max(1, n_samples)):
        out = model(batch)
        probs.append(out["ordinal"].class_probs)
        grades.append(out["ordinal"].expected_grade)
    P = torch.stack(probs)                                   # (S,B,K)
    mean = P.mean(0)
    K = mean.shape[1]
    logK = float(np.log(K))
    total = -(mean * mean.clamp_min(1e-9).log()).sum(1)
    per = -(P * P.clamp_min(1e-9).log()).sum(2).mean(0)       # aleatoric
    model.eval()
    return UncertaintyOutput(
        probs=mean, entropy=total / logK, aleatoric=per / logK,
        epistemic=((total - per) / logK).clamp_min(0.0),
        std=torch.stack(grades).std(0),
    )


class TemperatureScaler(nn.Module):
    """Single-parameter calibration: p = softmax(z / T)."""

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp().clamp(0.05, 20.0)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, iters: int = 300):
        logits, labels = logits.detach().cpu(), labels.detach().cpu()
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=iters)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self


class VectorScaler(nn.Module):
    """Per-class affine calibration: p = softmax(a * z + b)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.a + self.b

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, iters: int = 300):
        logits, labels = logits.detach().cpu(), labels.detach().cpu()
        opt = torch.optim.LBFGS([self.a, self.b], lr=0.05, max_iter=iters)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self


class IsotonicCalibrator:
    """Per-class isotonic regression, renormalised to a distribution."""

    def __init__(self):
        self.models: list = []

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        from sklearn.isotonic import IsotonicRegression
        self.models = []
        for k in range(probs.shape[1]):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(probs[:, k], (labels == k).astype(float))
            self.models.append(ir)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        out = np.stack([m.predict(probs[:, k]) for k, m in enumerate(self.models)], 1)
        return out / (out.sum(1, keepdims=True) + 1e-9)


def build_calibrator(kind: str, n_classes: int):
    if kind == "temperature":
        return TemperatureScaler()
    if kind == "vector":
        return VectorScaler(n_classes)
    if kind == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unknown calibrator '{kind}'")


DECISION_AUTONOMOUS = "AI_DECISION"
DECISION_REVIEW = "HUMAN_REVIEW"
DECISION_RETAKE = "RETAKE_IMAGE"


def decision_gate(entropy: float, quality_score: float, quality_decision: str,
                  entropy_hi: float, referable_prob: float) -> tuple[str, str]:
    """Three-way clinical routing. Returns (decision, reason)."""
    if quality_decision == "retake":
        return DECISION_RETAKE, (
            f"image quality Q={quality_score:.2f} below the retake threshold")
    if entropy >= entropy_hi:
        return DECISION_REVIEW, (
            f"predictive entropy {entropy:.2f} >= {entropy_hi:.2f}")
    if 0.35 <= referable_prob <= 0.65:
        return DECISION_REVIEW, (
            f"referable-DR probability {referable_prob:.2f} sits in the "
            "equivocal band")
    return DECISION_AUTONOMOUS, (
        f"entropy {entropy:.2f} and quality {quality_score:.2f} within "
        "autonomous-operation limits")
