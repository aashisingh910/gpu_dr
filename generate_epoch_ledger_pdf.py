#!/usr/bin/env python3
"""generate_epoch_ledger_pdf.py - PDF version of the "Epoch Ledger" artifact:
a complete, color-coded, epoch-by-epoch comparison of every training epoch
so far, plus the per-grade recall heatmap, trend charts, and the fixed
setup/architecture reference values. Mirrors the HTML artifact's content in
a form that can be archived, printed, or shared without needing the live
claude.ai link.

Data below is the same 16-epoch dataset shown in the artifact - re-run this
script any time to bring it up to date as more epochs complete.

Output: outputs/epoch_ledger.pdf
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def tint(t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        a, b, t2 = BAD, MID, t / 0.5
    else:
        a, b, t2 = MID, GOOD, (t - 0.5) / 0.5
    return tuple(a[i] + (b[i] - a[i]) * t2 for i in range(3))


# ---- data (S1 e1-5, S2 e1-12 as of 29 Sep 2026) ----------------------------
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
                transform=ax.transAxes, family="sans-serif", wrap=True)


def main():
    OUT_PDF.parent.mkdir(exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:

        # ================= PAGE 1: title + stats + glossary =================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        title_block(ax, "RETFOUND + GLA-LoRA · DIABETIC RETINOPATHY GRADING",
                    "Epoch Ledger",
                    "Every completed epoch so far, stage 1 through stage 2 (in progress) - loss, ordinal\n"
                    "agreement, minority-class recall, screening/STDR discrimination, hard-example-mining\n"
                    "state, and the full per-grade recall spread, read directly from the training log.")

        stats = [
            ("Epochs complete", "16 / 45", "Stage 2, epoch 12 of 12 in progress"),
            ("Best QWK", "0.554", "Stage 2, epochs 5 & 6"),
            ("Best score", "0.471", "Stage 2, epoch 5"),
            ("Best min-class recall", "0.474", "Stage 2, epoch 7"),
            ("Best refAUC", "0.848", "Stage 2, epoch 9 - climbing steadily"),
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
                    color=INK_DIM, transform=ax.transAxes, linespacing=1.4)
        footer(fig, "Generated from data/_resume_pipeline_full.log and outputs/retfound_plus_laft_xai/history.json - page 1/5")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 2: main comparative table =================
        fig = new_page()
        ax = fig.add_axes((0.02, 0.06, 0.96, 0.88)); ax.axis("off")
        ax.text(0.0, 1.0, "Full comparison, all 16 epochs", fontsize=16, fontweight="bold",
                color=INK, family="serif", transform=ax.transAxes)
        ax.text(0.0, 0.965, "Stage 1: head only, backbone frozen. Stage 2: GLA-LoRA + ALPP trainable, backbone still frozen. "
                            "Shading: worst (red) to best (green) per column; loss inverted (lower is greener).",
                fontsize=8, color=INK_DIM, transform=ax.transAxes)

        headers = ["Epoch", "Time"] + [c[1] for c in METRIC_COLS]
        cell_text = []
        cell_colors = []
        ranges = {c[0]: ranges_for(c[0]) for c in METRIC_COLS}
        for r in ROWS:
            row_txt = [r["ep"], f"{r['time']:.1f}m"]
            row_col = ["#f7f9fa" if r["stage"] == 1 else "#eaf6f6"] * 2
            for key, _, direction, fmt in METRIC_COLS:
                row_txt.append(fmt.format(r[key]))
                lo, hi = ranges[key]
                t = (r[key] - lo) / (hi - lo or 1)
                if direction == -1:
                    t = 1 - t
                if direction == 0:
                    row_col.append("#f7f9fa" if r["stage"] == 1 else "#eaf6f6")
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
        footer(fig, "Best value per column is the most-saturated green cell - page 2/5")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 3: per-grade recall heatmap =================
        fig = new_page()
        ax = fig.add_axes((0.05, 0.08, 0.55, 0.86)); ax.axis("off")
        ax.text(0.0, 1.0, "Per-grade recall, every epoch", fontsize=16, fontweight="bold",
                color=INK, family="serif", transform=ax.transAxes)
        ax.text(0.0, 0.955, "Shaded per grade (each column against its own best/worst across all\n"
                            "16 epochs) so a hard grade like Mild still shows its own real progress.",
                fontsize=8, color=INK_DIM, transform=ax.transAxes)

        g_ranges = [
            (min(r["recall"][g] for r in ROWS), max(r["recall"][g] for r in ROWS)) for g in range(5)
        ]
        headers2 = ["Epoch"] + GRADE_LABELS
        text2, colors2 = [], []
        for r in ROWS:
            row_txt = [r["ep"]]
            row_col = ["#f7f9fa" if r["stage"] == 1 else "#eaf6f6"]
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
        ax2 = fig.add_axes((0.63, 0.55, 0.33, 0.35))
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
        ax3 = fig.add_axes((0.63, 0.08, 0.33, 0.35))
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
        footer(fig, "Grade colors use the clinical 0-4 DR severity scale - page 3/5")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 4: trend charts =================
        fig = new_page()
        eps = [r["ep"].replace("S1 e", "1.").replace("S2 e", "2.") for r in ROWS]
        x = list(range(len(ROWS)))

        ax1 = fig.add_axes((0.06, 0.56, 0.42, 0.36))
        ax1.plot(x, [r["qwk"] for r in ROWS], "-o", color=ACCENT, label="QWK", markersize=3)
        ax1.plot(x, [r["score"] for r in ROWS], "-o", color="#c48a11", label="Score", markersize=3)
        ax1.axvline(4.5, color=INK_DIM, linestyle="--", linewidth=0.8)
        ax1.set_title("QWK & Score, all 16 epochs", fontsize=9, fontweight="bold")
        ax1.legend(fontsize=7, frameon=False)
        _style(ax1, x, eps)

        ax2 = fig.add_axes((0.55, 0.56, 0.42, 0.36))
        ax2.plot(x, [r["loss"] for r in ROWS], "-o", color=ACCENT, markersize=3)
        ax2.axvline(4.5, color=INK_DIM, linestyle="--", linewidth=0.8)
        ax2.set_title("Loss, all 16 epochs", fontsize=9, fontweight="bold")
        _style(ax2, x, eps)

        ax3 = fig.add_axes((0.06, 0.08, 0.42, 0.36))
        ax3.plot(x, [r["stdr"] for r in ROWS], "-o", color=GOOD, markersize=3)
        ax3.axvline(4.5, color=INK_DIM, linestyle="--", linewidth=0.8)
        ax3.set_title("STDR AUC (sight-threatening DR), all 16 epochs", fontsize=9, fontweight="bold")
        _style(ax3, x, eps)

        ax4 = fig.add_axes((0.55, 0.08, 0.42, 0.36))
        for g in range(5):
            ax4.plot(x, [r["recall"][g] for r in ROWS], "-o", color=GRADE_COLORS[g],
                    markersize=3, linewidth=1.3, label=f"G{g}")
        ax4.axvline(4.5, color=INK_DIM, linestyle="--", linewidth=0.8)
        ax4.set_title("Per-grade recall trend, all 16 epochs", fontsize=9, fontweight="bold")
        ax4.legend(fontsize=6.5, frameon=False, ncol=5, loc="upper left")
        _style(ax4, x, eps)

        fig.text(0.5, 0.965, "Trends", fontsize=16, fontweight="bold", ha="center", family="serif")
        footer(fig, "Dashed line marks the stage 1 -> stage 2 boundary - page 4/5")
        pdf.savefig(fig); plt.close(fig)

        # ================= PAGE 5: notable events + setup reference =================
        fig = new_page()
        ax = fig.add_axes((0, 0, 1, 1)); ax.axis("off")
        ax.text(0.03, 0.95, "Notable events", fontsize=16, fontweight="bold", color=INK,
                family="serif", transform=ax.transAxes)
        events = [
            ("Grade 1 (Mild) breaking its floor",
             "Stuck at 5-11% recall for 12 straight epochs, Mild reached 21.8% (e7), 13.9% (e9), 12.9% (e10),\n"
             "and 11.9% (e11) - four of the last five epochs above the old ceiling. Too early to call it settled,\n"
             "but the first sign this isn't just noise."),
            ("Epoch 10's five failed attempts",
             "Restarts kept landing before the ~90-100 minute epoch could finish and checkpoint, wiping it\n"
             "out each time across 4 days. Fixed by adding a checkpoint every 500 steps mid-epoch, plus\n"
             "(later) exact-step resume via a seeded sampler, instead of redoing the whole epoch."),
            ("Stage 1 -> Stage 2 lift",
             "Turning on GLA-LoRA and ALPP training lifted every average metric: QWK 0.415->0.483,\n"
             "score 0.338->0.412, min-recall 0.211->0.331 - at the cost of ~27% longer epochs."),
        ]
        y = 0.86
        for title, body in events:
            ax.add_patch(plt.Rectangle((0.03, y - 0.14), 0.94, 0.155, transform=ax.transAxes,
                                       fill=False, edgecolor=ACCENT, linewidth=1.2))
            ax.text(0.045, y - 0.02, title, fontsize=10.5, fontweight="bold", color=INK,
                    transform=ax.transAxes)
            ax.text(0.045, y - 0.055, body, fontsize=8.3, color=INK_DIM,
                    transform=ax.transAxes, linespacing=1.6)
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
        footer(fig, "Generated 29 Sep 2026 from data/_resume_pipeline_full.log, history.json and gla_lora.json - page 5/5")
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
