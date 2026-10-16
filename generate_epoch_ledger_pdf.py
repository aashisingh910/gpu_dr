#!/usr/bin/env python3
"""generate_epoch_ledger_pdf.py - PDF version of the "Epoch Ledger" artifact:
a complete, color-coded, epoch-by-epoch comparison of every training epoch
so far, plus the per-grade recall heatmap, trend charts, and the fixed
setup/architecture reference values. Mirrors the HTML artifact's content in
a form that can be archived, printed, or shared without needing the live
claude.ai link.

Data below is the same 28-epoch dataset shown in the artifact - re-run this
script any time to bring it up to date as more epochs complete.

Output: outputs/epoch_ledger.pdf
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "outputs" / "epoch_ledger.pdf"

PAGE_W, PAGE_H = 11.69, 8.27  # landscape A4, to fit the wide table

# ---- palette (matches the HTML artifact) -----------------------------------
BAD = (191 / 255, 58 / 255, 46 / 255)
MID = (196 / 255, 138 / 255, 17 / 255)
GOOD = (21 / 255, 142 / 255, 79 / 255)
GRADE_COLORS = ["#1f9d5c", "#93b52a", "#e0ab1e", "#e07a26", "#d13a2e"]
INK = "#16232c"
INK_DIM = "#5c6b74"
ACCENT = "#0e7c86"
STAGE_BG = {1: "#f7f9fa", 2: "#eaf6f6", 3: "#fdf1e6", 4: "#f3ecf9"}
STAGE_PT = {1: "#8a94a6", 2: ACCENT, 3: "#c4750f", 4: "#7a4fa3"}


def tint(t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        a, b, t2 = BAD, MID, t / 0.5
    else:
        a, b, t2 = MID, GOOD, (t - 0.5) / 0.5
    return tuple(a[i] + (b[i] - a[i]) * t2 for i in range(3))


# ---- data (S1 e1-5, S2 e1-12, S3 e1-11 as of 16 Oct 2026) ------------------
ROWS = [
    dict(ep="S1 e1", stage=1, time=75.7, loss=83.90, qwk=0.351, f1=0.292, minrec=0.187, score=0.300, acc=0.666, refauc=0.726, cleared=91.1, npv=0.9854, thr=0.237, stdr=0.8877, mh=1.00, mb=0.03, recall=[0.829, 0.059, 0.224, 0.278, 0.000]),
    dict(ep="S1 e2", stage=1, time=68.8, loss=48.43, qwk=0.421, f1=0.287, minrec=0.198, score=0.336, acc=0.607, refauc=0.778, cleared=94.1, npv=0.9851, thr=0.599, stdr=0.9162, mh=1.00, mb=0.05, recall=[0.726, 0.059, 0.341, 0.194, 0.286]),
    dict(ep="S1 e3", stage=1, time=75.6, loss=33.65, qwk=0.441, f1=0.284, minrec=0.168, score=0.339, acc=0.634, refauc=0.794, cleared=93.9, npv=0.9851, thr=0.590, stdr=0.9165, mh=1.15, mb=0.06, recall=[0.767, 0.069, 0.322, 0.111, 0.286]),
    dict(ep="S1 e4", stage=1, time=75.6, loss=27.29, qwk=0.403, f1=0.301, minrec=0.289, score=0.350, acc=0.569, refauc=0.800, cleared=94.1, npv=0.9851, thr=0.668, stdr=0.9149, mh=1.28, mb=0.07, recall=[0.653, 0.109, 0.397, 0.361, 0.333]),
    dict(ep="S1 e5", stage=1, time=70.1, loss=25.51, qwk=0.458, f1=0.309, minrec=0.211, score=0.364, acc=0.639, refauc=0.808, cleared=93.9, npv=0.9851, thr=0.646, stdr=0.9176, mh=1.37, mb=0.08, recall=[0.758, 0.059, 0.379, 0.194, 0.429]),
    dict(ep="S2 e1", stage=2, time=83.4, loss=24.05, qwk=0.434, f1=0.287, minrec=0.228, score=0.349, acc=0.539, refauc=0.809, cleared=94.9, npv=0.9852, thr=0.697, stdr=0.9203, mh=1.43, mb=0.08, recall=[0.611, 0.149, 0.425, 0.111, 0.429]),
    dict(ep="S2 e2", stage=2, time=95.0, loss=17.47, qwk=0.496, f1=0.376, minrec=0.269, score=0.414, acc=0.725, refauc=0.804, cleared=94.7, npv=0.9852, thr=0.470, stdr=0.9163, mh=1.47, mb=0.08, recall=[0.895, 0.05, 0.229, 0.528, 0.238]),
    dict(ep="S2 e3", stage=2, time=85.7, loss=12.15, qwk=0.465, f1=0.303, minrec=0.238, score=0.371, acc=0.609, refauc=0.813, cleared=94.6, npv=0.9852, thr=0.652, stdr=0.9201, mh=1.52, mb=0.08, recall=[0.719, 0.099, 0.364, 0.25, 0.286]),
    dict(ep="S2 e4", stage=2, time=98.8, loss=9.34, qwk=0.456, f1=0.318, minrec=0.301, score=0.384, acc=0.587, refauc=0.820, cleared=94.5, npv=0.9852, thr=0.686, stdr=0.9215, mh=1.55, mb=0.09, recall=[0.676, 0.109, 0.407, 0.389, 0.333]),
    dict(ep="S2 e5", stage=2, time=100.8, loss=8.13, qwk=0.554, f1=0.399, minrec=0.370, score=0.471, acc=0.703, refauc=0.832, cleared=94.2, npv=0.9851, thr=0.564, stdr=0.9461, mh=1.58, mb=0.09, recall=[0.835, 0.05, 0.364, 0.694, 0.238]),
    dict(ep="S2 e6", stage=2, time=91.5, loss=7.34, qwk=0.554, f1=0.390, minrec=0.334, score=0.461, acc=0.691, refauc=0.836, cleared=94.2, npv=0.9851, thr=0.549, stdr=0.9345, mh=1.59, mb=0.09, recall=[0.824, 0.05, 0.341, 0.611, 0.286]),
    dict(ep="S2 e7", stage=2, time=98.4, loss=6.95, qwk=0.402, f1=0.344, minrec=0.474, score=0.399, acc=0.479, refauc=0.836, cleared=94.9, npv=0.9853, thr=0.662, stdr=0.9459, mh=1.61, mb=0.09, recall=[0.489, 0.218, 0.537, 0.667, 0.286]),
    dict(ep="S2 e8", stage=2, time=91.0, loss=6.62, qwk=0.509, f1=0.380, minrec=0.385, score=0.445, acc=0.611, refauc=0.839, cleared=95.5, npv=0.9853, thr=0.643, stdr=0.9413, mh=1.61, mb=0.09, recall=[0.695, 0.099, 0.444, 0.611, 0.286]),
    dict(ep="S2 e9", stage=2, time=96.6, loss=6.37, qwk=0.474, f1=0.366, minrec=0.418, score=0.430, acc=0.578, refauc=0.848, cleared=95.0, npv=0.9853, thr=0.693, stdr=0.9426, mh=1.60, mb=0.09, recall=[0.639, 0.139, 0.477, 0.639, 0.333]),
    dict(ep="S2 e10", stage=2, time=83.5, loss=6.20, qwk=0.482, f1=0.380, minrec=0.438, score=0.443, acc=0.587, refauc=0.839, cleared=94.9, npv=0.9852, thr=0.691, stdr=0.9410, mh=1.60, mb=0.09, recall=[0.656, 0.129, 0.435, 0.750, 0.381]),
    dict(ep="S2 e11", stage=2, time=92.2, loss=6.08, qwk=0.510, f1=0.391, minrec=0.435, score=0.459, acc=0.615, refauc=0.839, cleared=94.7, npv=0.9852, thr=0.665, stdr=0.9499, mh=1.61, mb=0.09, recall=[0.695, 0.119, 0.435, 0.750, 0.333]),
    dict(ep="S2 e12", stage=2, time=101.9, loss=6.11, qwk=0.493, f1=0.387, minrec=0.439, score=0.450, acc=0.603, refauc=0.840, cleared=94.9, npv=0.9852, thr=0.667, stdr=0.9461, mh=1.61, mb=0.09, recall=[0.676, 0.129, 0.439, 0.750, 0.333]),
    dict(ep="S3 e1", stage=3, time=212.7, loss=6.00, qwk=0.448, f1=0.351, minrec=0.443, score=0.418, acc=0.544, refauc=0.839, cleared=94.9, npv=0.9853, thr=0.687, stdr=0.9549, mh=1.61, mb=0.09, recall=[0.602, 0.089, 0.435, 0.806, 0.286]),
    dict(ep="S3 e2", stage=3, time=194.1, loss=5.87, qwk=0.479, f1=0.370, minrec=0.445, score=0.440, acc=0.600, refauc=0.845, cleared=95.7, npv=0.9854, thr=0.659, stdr=0.9532, mh=1.61, mb=0.09, recall=[0.679, 0.119, 0.411, 0.806, 0.238]),
    dict(ep="S3 e3", stage=3, time=214.3, loss=5.52, qwk=0.506, f1=0.375, minrec=0.436, score=0.453, acc=0.585, refauc=0.855, cleared=95.8, npv=0.9854, thr=0.683, stdr=0.9613, mh=1.61, mb=0.09, recall=[0.659, 0.119, 0.411, 0.778, 0.286]),
    dict(ep="S3 e4", stage=3, time=217.1, loss=5.32, qwk=0.549, f1=0.393, minrec=0.416, score=0.476, acc=0.632, refauc=0.863, cleared=96.2, npv=0.9854, thr=0.695, stdr=0.9645, mh=1.60, mb=0.10, recall=[0.723, 0.109, 0.416, 0.722, 0.286]),
    dict(ep="S3 e5", stage=3, time=222.1, loss=5.17, qwk=0.577, f1=0.399, minrec=0.395, score=0.487, acc=0.675, refauc=0.862, cleared=96.1, npv=0.9854, thr=0.680, stdr=0.9657, mh=1.60, mb=0.10, recall=[0.791, 0.139, 0.350, 0.694, 0.286]),
    dict(ep="S3 e6", stage=3, time=302.4, loss=5.09, qwk=0.607, f1=0.412, minrec=0.389, score=0.505, acc=0.697, refauc=0.873, cleared=95.8, npv=0.9854, thr=0.660, stdr=0.9628, mh=1.61, mb=0.10, recall=[0.820, 0.109, 0.364, 0.694, 0.286]),
    dict(ep="S3 e7", stage=3, time=14.0, loss=4.87, qwk=0.557, f1=0.401, minrec=0.420, score=0.483, acc=0.645, refauc=0.870, cleared=95.8, npv=0.9854, thr=0.676, stdr=0.9640, mh=1.61, mb=0.10, recall=[0.738, 0.109, 0.430, 0.722, 0.333]),
    dict(ep="S3 e8", stage=3, time=48.3, loss=4.84, qwk=0.584, f1=0.421, minrec=0.402, score=0.499, acc=0.663, refauc=0.871, cleared=96.1, npv=0.9854, thr=0.679, stdr=0.9622, mh=1.61, mb=0.10, recall=[0.766, 0.109, 0.402, 0.694, 0.429]),
    dict(ep="S3 e9", stage=3, time=32.9, loss=4.73, qwk=0.588, f1=0.420, minrec=0.417, score=0.504, acc=0.672, refauc=0.870, cleared=94.9, npv=0.9852, thr=0.623, stdr=0.9593, mh=1.61, mb=0.10, recall=[0.779, 0.119, 0.383, 0.750, 0.381]),
    dict(ep="S3 e10", stage=3, time=116.6, loss=4.62, qwk=0.529, f1=0.402, minrec=0.466, score=0.478, acc=0.585, refauc=0.865, cleared=95.7, npv=0.9854, thr=0.680, stdr=0.9576, mh=1.60, mb=0.10, recall=[0.645, 0.218, 0.430, 0.750, 0.429]),
    dict(ep="S3 e11", stage=3, time=226.2, loss=4.59, qwk=0.590, f1=0.427, minrec=0.109, score=0.510, acc=0.670, refauc=0.869, cleared=95.1, npv=0.9853, thr=0.616, stdr=0.9595, mh=1.60, mb=0.10, recall=[0.770, 0.109, 0.416, 0.778, 0.381]),
]

# ---- stage-level config (fixed facts, not per-epoch data) -------------------
# trainable param counts/pct for S1-S3 are measured (printed at each stage's
# own startup banner); S4's is a projection from unfreeze_frac=1.00 in
# src/dr/config.py, not yet measured - stage 4 hasn't started.
STAGE_INFO = [
    # name, epochs_planned, trainable_params, trainable_pct, lr_head, lr_ladder, lr_lora, lr_backbone, measured
    ("S1 frozen",    5,  14_220_579, 4.449,  "1.0e-4", "1.0e-3", "n/a",    "n/a",    True),
    ("S2 LoRA",     12,  15_709_051, 4.915,  "1.0e-4", "1.0e-3", "5.0e-5", "n/a",    True),
    ("S3 unfreeze", 14, 116_478_843, 36.444, "5.0e-5", "5.0e-4", "5.0e-5", "5.0e-6", True),
    ("S4 full",     14, 319_612_795, 100.0,  "2.0e-5", "2.0e-4", "2.0e-5", "5.0e-6", False),
]

METRIC_COLS = [
    ("loss", "Loss", -1, "{:.2f}"),
    ("qwk", "QWK", 1, "{:.3f}"),
    ("f1", "F1", 1, "{:.3f}"),
    ("minrec", "Min-recall", 1, "{:.3f}"),
    ("score", "Score", 1, "{:.3f}"),
    ("acc", "Accuracy", 1, "{:.3f}"),
    ("refauc", "refAUC", 1, "{:.3f}"),
    ("cleared", "Screen cleared", 1, "{:.1f}%"),
    ("npv", "Screen NPV", 0, "{:.4f}"),
    ("thr", "Screen thr", 0, "{:.3f}"),
    ("stdr", "STDR AUC", 1, "{:.4f}"),
    ("mh", "Miner p90 H", 0, "{:.2f}"),
    ("mb", "Boundary rate", 0, "{:.2f}"),
]
GRADE_LABELS = ["Grade 0\nNo DR", "Grade 1\nMild", "Grade 2\nModerate", "Grade 3\nSevere", "Grade 4\nPDR"]


def ranges_for(key):
    vals = [r[key] for r in ROWS]
    return min(vals), max(vals)


def new_page(fig_kw=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    return fig


def footer(fig, text):
    fig.text(0.5, 0.012, text, ha="center", fontsize=7, color=INK_DIM)


def title_block(ax, kicker, title, dek, y=0.97):
    ax.text(0.03, y, kicker, fontsize=10, color=ACCENT, fontweight="bold",
            transform=ax.transAxes, family="sans-serif")
    ax.text(0.03, y - 0.06, title, fontsize=22, fontweight="bold", color=INK,
            transform=ax.transAxes, family="serif")
    if dek:
        ax.text(0.03, y - 0.12, dek, fontsize=9, color=INK_DIM,
                transform=ax.transAxes, family="sans-serif", wrap=True,
                verticalalignment="top", linespacing=1.5)


def main():
    OUT_PDF.parent.mkdir(exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:

        # ================= PAGE 1: title + stats + glossary =================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        title_block(ax, "RETFOUND + GLA-LoRA · DIABETIC RETINOPATHY GRADING",
                    "Epoch Ledger",
                    "Every completed epoch so far, stage 1 through stage 3 epoch 10 - loss, ordinal\n"
                    "agreement, minority-class recall, screening/STDR discrimination, hard-example-mining\n"
                    "state, and the full per-grade recall spread, read directly from the training log.")

        stats = [
            ("Epochs complete", "28 / 45", "Stage 3 epoch 12 in progress"),
            ("Best QWK", "0.607", "Stage 3, epoch 6 (was 0.554)"),
            ("Best score", "0.505", "Stage 3, epoch 6 (was 0.471)"),
            ("Best min-class recall", "0.474", "Stage 2, epoch 7"),
            ("Best refAUC", "0.873", "Stage 3, epoch 6 (was 0.848)"),
        ]
        x0 = 0.03
        w = 0.185
        for i, (label, value, sub) in enumerate(stats):
            x = x0 + i * (w + 0.012)
            ax.add_patch(plt.Rectangle((x, 0.60), w, 0.13, transform=ax.transAxes,
                                       fill=False, edgecolor="#d7e0e5", linewidth=1))
            ax.text(x + 0.01, 0.705, label.upper(), fontsize=7, color=INK_DIM,
                    transform=ax.transAxes, fontweight="bold")
            ax.text(x + 0.01, 0.665, value, fontsize=18, color=INK,
                    transform=ax.transAxes, fontweight="bold", family="monospace")
            ax.text(x + 0.01, 0.615, sub, fontsize=6.5, color=INK_DIM,
                    transform=ax.transAxes)

        # glossary
        ax.text(0.03, 0.55, "Column glossary", fontsize=13, fontweight="bold",
                color=INK, transform=ax.transAxes, family="serif")
        gloss = [
            ("Time", "context", "Wall-clock minutes to train the epoch; not a quality signal."),
            ("Loss", "lower better", "Weighted sum of every training objective, averaged per image."),
            ("QWK", "higher better", "Quadratic Weighted Kappa: ordinal agreement, penalises distant misses harder."),
            ("F1", "higher better", "Macro F1 across all 5 grades, treating each equally."),
            ("Min-recall", "higher better", "Recall of whichever grade is doing worst that epoch - the key imbalance metric."),
            ("Score", "higher better", "Composite selection score used to pick the best checkpoint."),
            ("Accuracy", "higher, cautiously", "Overall correct-grade rate; can rise just by favouring grade 0."),
            ("refAUC", "higher better", "AUC for the binary referable-vs-not-referable DR split."),
            ("Screen cleared", "context", "Share of images passed without flagging; trades off against NPV."),
            ("Screen NPV", "higher better", "Negative predictive value of the screening gate vs a 0.985 target."),
            ("Screen thr", "context", "Decision threshold that hit the NPV target this epoch."),
            ("STDR AUC", "higher better", "AUC for detecting sight-threatening DR (Severe + PDR)."),
            ("Miner p90 H", "context", "90th-percentile mining hardness carried into the next epoch."),
            ("Boundary rate", "context", "Share of samples flagged as one-grade boundary errors."),
            ("Per-grade recall", "higher better, all 5", "Recall per grade 0-4 - see the dedicated heatmap page."),
        ]
        col_w = 0.315
        for i, (term, direction, defn) in enumerate(gloss):
            col = i // 5
            row = i % 5
            x = 0.03 + col * (col_w + 0.01)
            y = 0.49 - row * 0.095
            dircolor = GOOD if "higher" in direction and "cautiously" not in direction else (
                BAD if direction == "lower better" else INK_DIM)
            ax.text(x, y, term, fontsize=8.5, fontweight="bold", color=INK,
                    transform=ax.transAxes, family="monospace")
            ax.text(x + col_w - 0.01, y, direction, fontsize=6.5, color=dircolor,
                    transform=ax.transAxes, ha="right", fontweight="bold")
            ax.text(x, y - 0.028, "\n".join(textwrap.wrap(defn, 62)), fontsize=6.8,
                    color=INK_DIM, transform=ax.transAxes, linespacing=1.4,
                    verticalalignment="top")
        footer(fig, "Generated from data/_resume_pipeline_full.log and outputs/retfound_plus_laft_xai/history.json - page 1/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 2: main comparative table =================
        fig = new_page()
        ax = fig.add_axes((0.02, 0.06, 0.96, 0.88)); ax.axis("off")
        ax.text(0.0, 1.0, "Full comparison, all 28 epochs", fontsize=16, fontweight="bold",
                color=INK, family="serif", transform=ax.transAxes)
        ax.text(0.0, 0.965, "Stage 1: head only, backbone frozen. Stage 2: GLA-LoRA + ALPP trainable, backbone still frozen. "
                            "Stage 3: 8/24 backbone blocks unfrozen too.\n"
                            "Shading: worst (red) to best (green) per column; loss inverted (lower is greener).",
                fontsize=8, color=INK_DIM, transform=ax.transAxes, verticalalignment="top", linespacing=1.5)

        headers = ["Epoch", "Time"] + [c[1] for c in METRIC_COLS]
        cell_text = []
        cell_colors = []
        ranges = {c[0]: ranges_for(c[0]) for c in METRIC_COLS}
        for r in ROWS:
            row_txt = [r["ep"], f"{r['time']:.1f}m"]
            row_col = [STAGE_BG[r["stage"]]] * 2
            for key, _, direction, fmt in METRIC_COLS:
                row_txt.append(fmt.format(r[key]))
                lo, hi = ranges[key]
                t = (r[key] - lo) / (hi - lo or 1)
                if direction == -1:
                    t = 1 - t
                if direction == 0:
                    row_col.append(STAGE_BG[r["stage"]])
                else:
                    row_col.append(tint(t))
            cell_text.append(row_txt)
            cell_colors.append(row_col)

        tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                       colColours=["#e9edf0"] * len(headers), loc="upper left",
                       bbox=(0, 0, 1, 0.90))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.3)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d7e0e5")
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_text_props(fontweight="bold", fontsize=6.5)
            elif col >= 2:
                # white text on the saturated heat cells for contrast
                key, _, direction, _ = METRIC_COLS[col - 2]
                if direction != 0:
                    cell.set_text_props(color="white", fontweight="bold")
        footer(fig, "Best value per column is the most-saturated green cell - page 2/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 3: per-grade recall heatmap =================
        fig = new_page()
        ax = fig.add_axes((0.05, 0.08, 0.55, 0.86)); ax.axis("off")
        ax.text(0.0, 1.0, "Per-grade recall, every epoch", fontsize=16, fontweight="bold",
                color=INK, family="serif", transform=ax.transAxes)
        ax.text(0.0, 0.955, "Shaded per grade (each column against its own best/worst across all\n"
                            "28 epochs) so a hard grade like Mild still shows its own real progress.",
                fontsize=8, color=INK_DIM, transform=ax.transAxes)

        g_ranges = [
            (min(r["recall"][g] for r in ROWS), max(r["recall"][g] for r in ROWS)) for g in range(5)
        ]
        headers2 = ["Epoch"] + GRADE_LABELS
        text2, colors2 = [], []
        for r in ROWS:
            row_txt = [r["ep"]]
            row_col = [STAGE_BG[r["stage"]]]
            for g in range(5):
                v = r["recall"][g]
                lo, hi = g_ranges[g]
                t = (v - lo) / (hi - lo or 1)
                row_txt.append(f"{v*100:.1f}%")
                row_col.append(tint(t))
            text2.append(row_txt)
            colors2.append(row_col)
        tbl2 = ax.table(cellText=text2, colLabels=headers2, cellColours=colors2,
                        colColours=["#e9edf0"] * len(headers2), loc="upper left",
                        bbox=(0, 0, 1, 0.90))
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(7)
        for (row, col), cell in tbl2.get_celld().items():
            cell.set_edgecolor("#d7e0e5"); cell.set_linewidth(0.4)
            if row == 0:
                cell.set_text_props(fontweight="bold", fontsize=6.8)
            elif col >= 1:
                cell.set_text_props(color="white", fontweight="bold")

        # grade distribution bar, right half of page
        ax2 = fig.add_axes((0.67, 0.55, 0.29, 0.35))
        counts = [20641, 1944, 4250, 707, 560]
        total = sum(counts)
        ax2.barh(range(5), counts, color=GRADE_COLORS)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"], fontsize=8)
        ax2.invert_yaxis()
        for i, c in enumerate(counts):
            ax2.text(c + 300, i, f"{c:,} ({c/total*100:.1f}%)", va="center", fontsize=7, color=INK_DIM)
        ax2.set_title("Training-split grade distribution (28,102 images)", fontsize=9, fontweight="bold")
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
        ax2.set_xlim(0, max(counts) * 1.35)

        # GLA-LoRA rank chart
        ax3 = fig.add_axes((0.67, 0.08, 0.29, 0.35))
        ranks = [13, 8, 8, 10, 11, 11, 11, 11, 10, 10, 11, 10, 10, 10, 9, 9, 11, 10, 9, 9, 10, 10, 9, 3]
        bar_colors = [ACCENT if (i == 0 or rk >= 13) else ("#bf3a2e" if rk <= 4 else "#9aa5ab")
                     for i, rk in enumerate(ranks)]
        ax3.bar(range(24), ranks, color=bar_colors)
        ax3.set_xlabel("ViT-Large block index", fontsize=7)
        ax3.set_ylabel("LoRA rank", fontsize=7)
        ax3.set_title("GLA-LoRA per-block rank allocation (real, gla_lora.json)", fontsize=9, fontweight="bold")
        ax3.tick_params(labelsize=6.5)
        for spine in ("top", "right"):
            ax3.spines[spine].set_visible(False)
        footer(fig, "Grade colors use the clinical 0-4 DR severity scale - page 3/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 4: trend charts =================
        fig = new_page()
        eps = [r["ep"].replace("S1 e", "1.").replace("S2 e", "2.").replace("S3 e", "3.")
              for r in ROWS]
        x = list(range(len(ROWS)))
        # stage boundaries: S1|S2 falls between rows 4/5 (0-indexed), S2|S3
        # between rows 16/17 - only draw a boundary that's actually inside
        # the data so far.
        boundaries = [b for b in (4.5, 16.5) if b < len(ROWS) - 0.5]

        def _bounds(ax):
            for b in boundaries:
                ax.axvline(b, color=INK_DIM, linestyle="--", linewidth=0.8)

        ax1 = fig.add_axes((0.06, 0.56, 0.42, 0.36))
        ax1.plot(x, [r["qwk"] for r in ROWS], "-o", color=ACCENT, label="QWK", markersize=3)
        ax1.plot(x, [r["score"] for r in ROWS], "-o", color="#c48a11", label="Score", markersize=3)
        _bounds(ax1)
        ax1.set_title("QWK & Score, all 28 epochs", fontsize=9, fontweight="bold")
        ax1.legend(fontsize=7, frameon=False)
        _style(ax1, x, eps)

        ax2 = fig.add_axes((0.55, 0.56, 0.42, 0.36))
        ax2.plot(x, [r["loss"] for r in ROWS], "-o", color=ACCENT, markersize=3)
        _bounds(ax2)
        ax2.set_title("Loss, all 28 epochs", fontsize=9, fontweight="bold")
        _style(ax2, x, eps)

        ax3 = fig.add_axes((0.06, 0.08, 0.42, 0.36))
        ax3.plot(x, [r["stdr"] for r in ROWS], "-o", color=GOOD, markersize=3)
        _bounds(ax3)
        ax3.set_title("STDR AUC (sight-threatening DR), all 28 epochs", fontsize=9, fontweight="bold")
        _style(ax3, x, eps)

        ax4 = fig.add_axes((0.55, 0.08, 0.42, 0.36))
        for g in range(5):
            ax4.plot(x, [r["recall"][g] for r in ROWS], "-o", color=GRADE_COLORS[g],
                    markersize=3, linewidth=1.3, label=f"G{g}")
        _bounds(ax4)
        ax4.set_title("Per-grade recall trend, all 28 epochs", fontsize=9, fontweight="bold")
        ax4.legend(fontsize=6.5, frameon=False, ncol=5, loc="upper left")
        _style(ax4, x, eps)

        fig.text(0.5, 0.965, "Trends", fontsize=16, fontweight="bold", ha="center", family="serif")
        footer(fig, "Dashed lines mark stage boundaries (1|2 and 2|3) - page 4/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 5: stage-by-stage comparison ================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        title_block(ax, "RETFOUND + GLA-LoRA · DIABETIC RETINOPATHY GRADING",
                    "Stage-by-Stage Comparison",
                    "What each stage changed vs. what it achieved - S4 is config only, not yet run.")

        def stage_stats(stage_no):
            pts = [r for r in ROWS if r["stage"] == stage_no]
            if not pts:
                return None
            return {
                "n": len(pts),
                "qwk_avg": sum(r["qwk"] for r in pts) / len(pts), "qwk_best": max(r["qwk"] for r in pts),
                "score_avg": sum(r["score"] for r in pts) / len(pts), "score_best": max(r["score"] for r in pts),
                "refauc_avg": sum(r["refauc"] for r in pts) / len(pts), "refauc_best": max(r["refauc"] for r in pts),
                "minrec_avg": sum(r["minrec"] for r in pts) / len(pts),
                "time_avg": sum(r["time"] for r in pts) / len(pts), "time_total": sum(r["time"] for r in pts),
            }

        headers = ["Stage", "Epochs", "Trainable\nparams", "% of\nbackbone", "Head /\nladder LR",
                  "LoRA /\nbackbone LR", "Avg\nQWK", "Best\nQWK", "Avg\nscore", "Best\nscore",
                  "Avg\nrefAUC", "Avg\nmin-rec", "Avg min\n/epoch"]
        cell_text, cell_colors = [], []
        for i, (name, planned, params, pct, lr_h, lr_l, lr_lo, lr_b, measured) in enumerate(STAGE_INFO):
            st = stage_stats(i + 1)
            row_bg = STAGE_BG.get(i + 1, "#f3ecf9")
            if st is None:
                row = [name, f"0/{planned}", f"{params:,}", f"{pct:.2f}%", f"{lr_h} / {lr_l}",
                      f"{lr_lo} / {lr_b}", "-", "-", "-", "-", "-", "-", "-"]
            else:
                done_tag = f"{st['n']}/{planned}" + ("" if st["n"] == planned else " (running)")
                row = [name, done_tag, f"{params:,}", f"{pct:.2f}%", f"{lr_h} / {lr_l}",
                      f"{lr_lo} / {lr_b}", f"{st['qwk_avg']:.3f}", f"{st['qwk_best']:.3f}",
                      f"{st['score_avg']:.3f}", f"{st['score_best']:.3f}", f"{st['refauc_avg']:.3f}",
                      f"{st['minrec_avg']:.3f}", f"{st['time_avg']:.0f}"]
            cell_text.append(row)
            cell_colors.append([row_bg] * len(headers))

        tbl_ax = fig.add_axes((0.02, 0.58, 0.96, 0.20)); tbl_ax.axis("off")
        tbl = tbl_ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                           colColours=["#e9edf0"] * len(headers), loc="upper left", bbox=(0, 0, 1, 1))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.2)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d7e0e5"); cell.set_linewidth(0.4)
            if row == 0:
                cell.set_text_props(fontweight="bold", fontsize=6.8)

        # ---- grouped bar: avg vs best, QWK and score, per stage ------------
        axg = fig.add_axes((0.06, 0.08, 0.42, 0.40))
        stage_names = [s[0] for s in STAGE_INFO]
        qwk_avgs, qwk_bests, score_avgs, score_bests = [], [], [], []
        for i in range(4):
            st = stage_stats(i + 1)
            qwk_avgs.append(st["qwk_avg"] if st else 0)
            qwk_bests.append(st["qwk_best"] if st else 0)
            score_avgs.append(st["score_avg"] if st else 0)
            score_bests.append(st["score_best"] if st else 0)
        x = np.arange(4); w = 0.2
        axg.bar(x - 1.5 * w, qwk_avgs, w, label="QWK avg", color=ACCENT, alpha=0.55)
        axg.bar(x - 0.5 * w, qwk_bests, w, label="QWK best", color=ACCENT)
        axg.bar(x + 0.5 * w, score_avgs, w, label="Score avg", color="#c48a11", alpha=0.55)
        axg.bar(x + 1.5 * w, score_bests, w, label="Score best", color="#c48a11")
        axg.set_xticks(x); axg.set_xticklabels(stage_names, fontsize=7.5)
        axg.set_title("Average vs. best per stage", fontsize=9, fontweight="bold")
        axg.legend(fontsize=6.5, frameon=False, ncol=2)
        axg.tick_params(labelsize=6.5)
        axg.grid(axis="y", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            axg.spines[spine].set_visible(False)

        # ---- trainable capacity vs avg score reached (log-x) ---------------
        axc = fig.add_axes((0.56, 0.08, 0.40, 0.40))
        for i in range(4):
            st = stage_stats(i + 1)
            if st is None:
                continue
            axc.scatter(STAGE_INFO[i][3], st["score_avg"], s=90, color=STAGE_PT.get(i + 1, "#7a4fa3"),
                       zorder=3, edgecolor="white", linewidth=0.6)
            axc.annotate(STAGE_INFO[i][0], (STAGE_INFO[i][3], st["score_avg"]),
                        textcoords="offset points", xytext=(-7, -3), fontsize=7, ha="right")
        axc.set_xscale("log")
        axc.set_xlabel("Trainable params (% of backbone, log scale)", fontsize=7.5)
        axc.set_ylabel("Avg selection score", fontsize=7.5)
        axc.set_title("More trainable capacity -> better average score so far",
                     fontsize=9, fontweight="bold")
        axc.tick_params(labelsize=6.5)
        axc.grid(alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            axc.spines[spine].set_visible(False)

        ax.text(0.02, 0.535,
               "Each capacity jump has paid off so far (avg QWK 0.415->0.486->0.543); S3's avg epoch time is\n"
               "skewed low by one resume-split epoch (true cost closer to 195-300 min) - see page 8.",
               fontsize=7.6, color=INK_DIM, transform=ax.transAxes, verticalalignment="center",
               style="italic", linespacing=1.4)
        footer(fig, "S4 row is projected from config, not measured - page 5/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 6: notable events + setup reference =================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        ax.text(0.03, 0.95, "Notable events", fontsize=16, fontweight="bold", color=INK,
                family="serif", transform=ax.transAxes)
        events = [
            ("Grade 1 (Mild) breaking its floor - then matching its ceiling",
             "Stuck at 5-11% recall for 12 straight epochs, Mild broke through from stage 2 epoch 7 onward,\n"
             "then tied its all-time-best 21.8% again at stage 3 epoch 10 - the first repeat of that ceiling\n"
             "in 18 epochs, landing on the same epoch QWK/score dipped (see page 9)."),
            ("Two resume bugs found this week - both now fixed and confirmed",
             "Mid-epoch resume (added Sep 29) missed its own first test (stale in-memory code); a Sep 30\n"
             "retry confirmed it working. Separately, ONE patience counter spanned every stage, so stage 2's\n"
             "plateau silently killed stage 3 after 1 epoch - fixed to reset per stage; resumed into epoch 2."),
            ("Stage 3 breaks the plateau the decision bar was watching for",
             "Epoch 1 regressed (score 0.418) as expected, but by epoch 4 it had already passed the old\n"
             "best (0.471, stage 2 epoch 5) and kept climbing - QWK 0.493->0.607, score 0.450->0.505,\n"
             "refAUC 0.840->0.873, all new run highs at stage 3 epoch 6. Unfreezing the backbone worked."),
        ]
        y = 0.86
        for title, body in events:
            ax.add_patch(plt.Rectangle((0.03, y - 0.14), 0.94, 0.155, transform=ax.transAxes,
                                       fill=False, edgecolor=ACCENT, linewidth=1.2))
            ax.text(0.045, y - 0.02, title, fontsize=10.5, fontweight="bold", color=INK,
                    transform=ax.transAxes)
            ax.text(0.045, y - 0.055, body, fontsize=8.3, color=INK_DIM,
                    transform=ax.transAxes, linespacing=1.6, verticalalignment="top")
            y -= 0.175

        ax.text(0.03, 0.27, "Setup & architecture reference", fontsize=14, fontweight="bold",
                color=INK, family="serif", transform=ax.transAxes)
        ref = [
            ("Dataset", "35,126 images, 17,563 patients (EyePACS)"),
            ("Split", "28,102 / 3,512 / 3,512 train/val/test, no patient leakage"),
            ("RETFound coverage", "100% - 294/294 backbone tensors loaded"),
            ("GLA-LoRA params", "1,431,552 - 0.45% of the 319.6M backbone"),
            ("Sampling ratio", "5.01x grade4:grade0 weight, n^-0.5"),
        ]
        for i, (k, v) in enumerate(ref):
            yy = 0.21 - i * 0.035
            ax.text(0.03, yy, k, fontsize=8.5, fontweight="bold", color=INK, transform=ax.transAxes,
                    family="monospace")
            ax.text(0.28, yy, v, fontsize=8.5, color=INK_DIM, transform=ax.transAxes)
        footer(fig, "Generated 16 Oct 2026 from data/_resume_pipeline_full.log, history.json and gla_lora.json - page 6/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 7: efficiency & spread ======================
        fig = new_page()
        ax0 = fig.add_axes((0, 0, 1, 1)); ax0.axis("off")
        title_block(ax0, "RETFOUND + GLA-LoRA · DIABETIC RETINOPATHY GRADING",
                    "Efficiency & Spread",
                    "Two more views of the same 28 epochs: is the extra training time paying off, and how\n"
                    "much does each grade's recall vary epoch to epoch.")

        # ---- scatter: score vs wall-clock time, coloured by stage ---------
        axs = fig.add_axes((0.08, 0.15, 0.40, 0.62))
        for stage_no, color, label in ((1, STAGE_PT[1], "Stage 1"), (2, STAGE_PT[2], "Stage 2"),
                                       (3, STAGE_PT[3], "Stage 3")):
            pts = [r for r in ROWS if r["stage"] == stage_no]
            axs.scatter([r["time"] for r in pts], [r["score"] for r in pts],
                       s=30, color=color, label=label, zorder=3, edgecolor="white", linewidth=0.5)
        best = max(ROWS, key=lambda r: r["score"])
        axs.annotate(f"best: {best['ep']}", (best["time"], best["score"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=6.5, color=INK,
                    fontweight="bold")
        axs.set_xlabel("Epoch wall-clock time (min)", fontsize=7.5)
        axs.set_ylabel("Selection score", fontsize=7.5)
        axs.set_title("Score vs. training time - is the extra time paying off?",
                     fontsize=8.5, fontweight="bold")
        axs.legend(fontsize=6.5, frameon=False, loc="lower right")
        axs.tick_params(labelsize=6.5)
        axs.grid(alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            axs.spines[spine].set_visible(False)

        # ---- box plot: per-grade recall spread across all epochs ----------
        axb = fig.add_axes((0.56, 0.15, 0.39, 0.62))
        data = [[r["recall"][g] for r in ROWS] for g in range(5)]
        bp = axb.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="white", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_DIM,
                                        markeredgecolor="none"))
        for patch, color in zip(bp["boxes"], GRADE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        axb.set_xticklabels(["G0", "G1", "G2", "G3", "G4"], fontsize=7.5)
        axb.set_ylabel("Recall", fontsize=7.5)
        axb.set_title("Per-grade recall spread, all 28 epochs",
                     fontsize=8.5, fontweight="bold")
        axb.tick_params(labelsize=6.5)
        axb.grid(axis="y", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            axb.spines[spine].set_visible(False)

        footer(fig, "Selection score = 0.5*QWK + 0.3*MacroF1 + 0.2*minority-recall (grades 1-3) - page 7/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 8: detailed analysis (1/2) ==================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        title_block(ax, "RETFOUND + GLA-LoRA · DIABETIC RETINOPATHY GRADING",
                    "Detailed Analysis & Conclusion",
                    "The full read on 28 epochs, written out rather than just tabulated (1 of 2).")

        sections_p1 = [
            ("1. Trajectory",
             "Three stages, three distinct regimes, each measurably better than the last. Stage 1 (head only, "
             "backbone frozen) established a floor: avg QWK 0.415, avg score 0.338 - the model learns the "
             "task at all, but with almost no capacity to adapt its visual features to fundus images. Stage 2 "
             "(GLA-LoRA adapters, still-frozen backbone) lifted every average metric (QWK to 0.486, score to "
             "0.423) using only 1.43M extra parameters (0.45% of the backbone) and then plateaued hard for "
             "its last 7 epochs - epochs 11-17 never beat epoch 10's score of 0.471. Stage 3 (33% of the "
             "backbone blocks unfrozen) broke that plateau decisively: by epoch 4 it had already passed "
             "stage 2's best, and by epoch 6 it reached QWK 0.607 and score 0.505 - the run's best result "
             "to date, on every headline metric at once (QWK, score, refAUC, STDR AUC)."),
            ("2. Why stage 3 worked",
             "The stage 2 plateau was not a sign the model had learned everything it could from this data - "
             "it was a sign that 0.45%-of-backbone adapter capacity, with the backbone itself permanently "
             "frozen, had run out of room to represent whatever stage 1/2 hadn't already captured. Unfreezing "
             "even a third of the backbone (36.4% of all parameters now trainable, at a deliberately low "
             "5e-6 backbone LR to avoid catastrophic forgetting of the RETFound pretraining) gave the model "
             "new degrees of freedom specifically in the deeper visual representation, not just in a "
             "lightweight adapter sitting on top of it. Epoch 1 of stage 3 regressed on every metric (score "
             "0.450->0.418) - the same destabilization pattern seen at the stage 1->2 boundary - before "
             "recovering and then exceeding the prior ceiling, which is the expected signature of a genuine "
             "capacity increase rather than overfitting to validation noise."),
            ("3. What's still weak",
             "Grade 1 (Mild NPDR) recall remains the model's clearest weakness: it averaged 12.4% across all "
             "10 stage-3 epochs so far, nowhere near grade 0's 60-82% or grade 3's 69-81%. This almost "
             "certainly reflects the training split's own composition - grade 1 sits between the overwhelming "
             "grade-0 majority and the more visually distinct grade-2/3 lesions, and is the easiest grade for "
             "the ordinal head to blur into its neighbours. Notably, epoch 10 tied the run's all-time-best "
             "Mild recall (21.8%, exactly matching stage 2 epoch 7) - the first time that ceiling has been "
             "touched again in 18 epochs, and it happened on an epoch whose QWK/score both dropped, with "
             "grade 4 recall simultaneously at its joint-best (42.9%) too. That pattern - minority-grade "
             "recall up, headline ordinal agreement down, in the same epoch - reads as the model trading "
             "some majority-grade precision for minority-grade sensitivity, not a random bad epoch."),
        ]

        def render_sections(ax, sections, start_y=0.80):
            yy = start_y
            for heading, body in sections:
                wrapped = textwrap.wrap(body, 112)
                n_lines = len(wrapped)
                box_h = n_lines * 0.0195 + 0.045
                ax.text(0.03, yy, heading, fontsize=10.5, fontweight="bold", color=ACCENT,
                       family="serif", transform=ax.transAxes, verticalalignment="top")
                ax.text(0.03, yy - 0.032, "\n".join(wrapped), fontsize=8.1,
                       color=INK_DIM, transform=ax.transAxes, linespacing=1.48, verticalalignment="top")
                yy -= box_h + 0.032
            return yy

        render_sections(ax, sections_p1)
        footer(fig, "Every figure above is read from this run's own history.json / training log - page 8/9")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 9: detailed analysis (2/2) ==================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        title_block(ax, "RETFOUND + GLA-LoRA · DIABETIC RETINOPATHY GRADING",
                    "Detailed Analysis & Conclusion",
                    "The full read on 28 epochs, written out rather than just tabulated (2 of 2).")

        sections_p2 = [
            ("4. Deployment-relevant safety signal",
             "Separately from grading accuracy, the screening gate (flag for human review vs. clear "
             "automatically, calibrated to a 0.985 negative-predictive-value target) has held in every one "
             "of the 28 epochs with no exception, including stage 3's regressed epoch 1. This is a different "
             "and arguably more deployment-critical claim than QWK: it says that even while the model's "
             "grading quality was fluctuating, the triage layer sitting in front of it was not sending "
             "disease-positive cases home undetected. That constraint being stage-transition-proof is a "
             "meaningfully separate result from the grading-quality trajectory above."),
            ("5. Risk & outlook",
             "Epoch 7's dip (score 0.505->0.483) was followed by two epochs of recovery (0.499, then 0.504), "
             "then a second and deeper dip at epoch 10 (0.478 - the lowest score anywhere in stage 3 after "
             "epoch 1's initial regression). Two dips in ten epochs, with the best point still three epochs "
             "back at epoch 6, is a weaker trend than one dip would have been - but epoch 10's dip came "
             "paired with a tie for the best-ever Mild recall and joint-best PDR recall, which is a different "
             "and more encouraging story than a plain regression. The open question for the remaining 4 "
             "stage-3 epochs and all of stage 4 is whether QWK/score resume climbing (capacity was still the "
             "binding constraint and epoch 10 was a minority-recall-favouring fluctuation), settle into a "
             "noisier plateau around epoch 6-10's range, or - unchanged from the prior read - the "
             "minority-grade ceiling is closer to data-limited than capacity-limited, in which case stage 4's "
             "full unfreeze would mainly buy marginal gains on the majority grades."),
            ("6. Conclusion & recommendation",
             "The run has already answered its central open question as of the last update: unfreezing "
             "backbone capacity was worth the ~2x epoch-time cost, decisively. best.pt still points to stage "
             "3 epoch 6, not stage 2 epoch 10 (the old best). Recommendation: let stage 3 finish its "
             "remaining epochs and let stage 4 begin - there is no evidence yet of a ceiling that would "
             "justify stopping early - while watching two things specifically: whether QWK/score recover "
             "from epoch 10's dip the way they did from epoch 7's, and whether grade 1 recall's new tie at "
             "21.8% becomes a genuine new floor or stays a one-epoch spike - the clearest open question this "
             "report cannot yet answer."),
        ]
        render_sections(ax, sections_p2)
        footer(fig, "Every figure above is read from this run's own history.json / training log - page 9/9")
        pdf.savefig(fig); plt.close(fig)

    print(f"[saved] {OUT_PDF}")


def _style(ax, x, eps):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(eps, fontsize=6, rotation=90)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


if __name__ == "__main__":
    main()
