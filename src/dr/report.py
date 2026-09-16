"""Final Clinical AI Report - the right-hand panel of the architecture figure.

Assembles DR grade, referable / sight-threatening flags, lesion summary,
progression risk, uncertainty and the XAI panel into a text report plus a
four-panel figure (original, attribution heat-map, lesion mask, counterfactual).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .data.eyepacs import GRADE_NAMES
from .data.lesion_priors import LESION_NAMES

LESION_LABELS = {
    "MA": "Microaneurysms", "HE": "Haemorrhages", "EX_H": "Hard exudates",
    "EX_S": "Soft exudates (cotton-wool spots)", "NV": "Neovascularisation",
    "ME": "Macular oedema (CFP proxy)",
}
RISK_BANDS = ((0.20, "Low"), (0.45, "Med"), (0.70, "High"), (1.01, "High"))


def risk_band(p: float) -> str:
    for hi, name in RISK_BANDS:
        if p < hi:
            return name
    return "High"


@dataclass
class ClinicalReport:
    image_id: str
    grade: int
    grade_probs: np.ndarray
    referable: float
    sight_threatening: float
    lesions: dict
    quality: dict
    uncertainty: dict
    decision: str
    decision_reason: str
    progression: dict = field(default_factory=dict)
    counterfactual_delta: float | None = None
    triage: dict | None = None      # screening decision + the operating point used

    def to_text(self) -> str:
        L = []
        w = 68
        L.append("=" * w)
        L.append(f"FINAL CLINICAL AI REPORT - {self.image_id}")
        L.append("=" * w)

        L.append("\nIMAGE QUALITY (A1)")
        q = self.quality
        L.append(f"  Quality score Q      : {q['score']:.3f}  -> {q['decision'].upper()}")
        L.append(f"  Gradability {q['gradability']:.2f} | Blur {q['blur']:.2f} | "
                 f"Illumination {q['illumination']:.2f} | Artefact {q['artifact']:.2f} "
                 f"| FOV {q['fov']:.2f}")
        L.append(f"  Camera/domain        : {q['domain']}")

        L.append("\nDR GRADE (A6, ordinal 0-4)")
        for k, name in enumerate(GRADE_NAMES):
            bar = "#" * int(round(self.grade_probs[k] * 40))
            mark = " <-- predicted" if k == self.grade else ""
            L.append(f"  {k} {name:9s} {self.grade_probs[k]:6.3f} {bar}{mark}")
        L.append(f"  Predicted grade      : {self.grade} ({GRADE_NAMES[self.grade]})")

        if self.triage:
            t = self.triage
            L.append("\nSCREENING TRIAGE (primary output)")
            L.append(f"  >> {t['decision']}")
            L.append(f"  task                 : {t['task']}")
            L.append(f"  score / threshold    : {t['score']:.3f} / {t['threshold']:.3f}")
            L.append(f"  reason               : {t['reason']}")
            L.append(f"  operating point NPV  : {t['npv']:.4f} on held-out test "
                     f"(clears {t['workload_reduction']*100:.1f}% of patients)")
            L.append("  NOTE: this threshold was fitted on EyePACS validation data. "
                     "Recalibrate on local data before use at a new site.")

        L.append("\nCLINICAL FLAGS")
        ref = "YES" if self.referable >= 0.5 else "NO"
        stg = "YES" if self.sight_threatening >= 0.5 else "NO"
        L.append(f"  Referable DR (>=grade 2)        : {ref:3s} (p={self.referable:.3f})")
        L.append(f"  Sight-threatening (>=grade 3)   : {stg:3s} "
                 f"(p={self.sight_threatening:.3f})")

        L.append("\nLESION SUMMARY (A3 mixture-of-experts)")
        L.append(f"  {'lesion':34s} {'present':>8s} {'p':>7s} {'gate a':>8s}")
        for k in LESION_NAMES:
            d = self.lesions[k]
            flag = "yes" if d["present"] else " - "
            L.append(f"  {LESION_LABELS[k]:34s} {flag:>8s} {d['prob']:7.3f} "
                     f"{d['alpha']:8.3f}")

        if self.progression:
            L.append("\nPROGRESSION RISK (A7)")
            L.append("  " + "  ".join(f"{h:>6s}" for h in self.progression))
            L.append("  " + "  ".join(f"{risk_band(v):>6s}" for v in
                                      self.progression.values()))

        L.append("\nCONFIDENCE / UNCERTAINTY (A10)")
        u = self.uncertainty
        L.append(f"  Predictive entropy   : {u['entropy']:.3f} "
                 f"(0 = certain, 1 = maximal)")
        L.append(f"  Aleatoric / epistemic: {u['aleatoric']:.3f} / {u['epistemic']:.3f}")
        L.append(f"  Grade std (MC)       : {u['std']:.3f}")

        if self.counterfactual_delta is not None:
            L.append("\nEXPLAINABILITY (A9)")
            L.append(f"  Counterfactual: erasing the detected lesions changes the "
                     f"referable-DR probability by {self.counterfactual_delta:+.3f}")

        L.append("\nDECISION GATE (A10)")
        L.append(f"  >> {self.decision}")
        L.append(f"  reason: {self.decision_reason}")
        L.append("\n" + "-" * w)
        L.append("Research prototype. Not a medical device; not for clinical use.")
        L.append("=" * w)
        return "\n".join(L)


def _colorise(sal: np.ndarray, size: int) -> np.ndarray:
    s = cv2.resize(sal.astype(np.float32), (size, size))
    s = np.clip(s, 0, 1)
    return cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)


def render_panel(image_bgr: np.ndarray, saliency: np.ndarray,
                 lesion_mask: np.ndarray, counterfactual: np.ndarray,
                 out_path: str) -> str:
    """Four-panel XAI figure: original | attribution | lesion evidence | counterfactual."""
    size = image_bgr.shape[0]
    heat = cv2.addWeighted(image_bgr, 0.55, _colorise(saliency, size), 0.45, 0)
    m = cv2.resize(lesion_mask.astype(np.float32), (size, size))
    overlay = image_bgr.copy()
    overlay[m > 0.4] = (0, 255, 255)
    lesion_vis = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

    panels = [image_bgr, heat, lesion_vis, counterfactual]
    titles = ["Original I*", "Attribution", "Lesion evidence", "Counterfactual"]
    labelled = []
    for p, t in zip(panels, titles):
        p = p.copy()
        cv2.rectangle(p, (0, 0), (size, 22), (0, 0, 0), -1)
        cv2.putText(p, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        labelled.append(p)
    cv2.imwrite(out_path, np.hstack(labelled))
    return out_path
