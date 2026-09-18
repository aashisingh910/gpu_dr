#!/usr/bin/env python3
"""generate_session_report.py - implementation / session documentation PDF.

Not part of the DR training pipeline itself - this documents the WORK done
to get the pipeline running on this machine (bugs found and fixed, the
checkpoint/resume architecture, the CSV export additions, GPU verification)
plus a live snapshot of pipeline progress at generation time, so there is a
durable record independent of this chat.

Re-run any time to refresh the "current status" section with live numbers:

    .venv\\Scripts\\python.exe generate_session_report.py

Output: outputs/session_report.pdf (and a plain-text outputs/session_report.txt
with the same content, for a quick read without a PDF viewer).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "outputs" / "session_report.pdf"
OUT_TXT = ROOT / "outputs" / "session_report.txt"

PAGE_W, PAGE_H = 8.27, 11.69   # A4, inches
MARGIN_IN = 0.75
LINE_IN = {6.8: 0.0155, 7.0: 0.016, 8.4: 0.0185, 8.8: 0.019,
          9: 0.0195, 10: 0.021, 12: 0.026, 16: 0.034}


class Doc:
    """Minimal paginated text-PDF builder (matplotlib PdfPages backend, same
    approach this project's own objectiveN_report.py scripts use) plus a
    parallel plain-text transcript, so the content is readable either way."""

    def __init__(self, pdf: PdfPages, footer: str):
        self.pdf = pdf
        self.footer = footer
        self.txt_lines: list[str] = []
        self.fig = None
        self.ax = None
        self.y = None
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self._flush_page()
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = 1 - MARGIN_IN / PAGE_H

    def _flush_page(self):
        self.ax.text(0.5, MARGIN_IN / PAGE_H * 0.4, self.footer,
                     ha="center", fontsize=7, color="#888888")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def _room(self, need_in: float) -> bool:
        return (self.y - need_in / PAGE_H) > (MARGIN_IN / PAGE_H)

    def _ensure(self, need_in: float):
        if not self._room(need_in):
            self._new_page()

    def title(self, text: str, sub: str = ""):
        self._ensure(0.9)
        self.ax.text(0.5, self.y, text, ha="center", fontsize=20,
                     fontweight="bold", va="top")
        self.y -= 0.05
        if sub:
            self.ax.text(0.5, self.y, sub, ha="center", fontsize=11,
                         color="#555555", va="top")
            self.y -= 0.05
        self.y -= 0.02
        self.txt_lines.append(text)
        if sub:
            self.txt_lines.append(sub)
        self.txt_lines.append("")

    def h1(self, text: str):
        self._ensure(0.09)
        self.y -= 0.012
        self.ax.text(MARGIN_IN / PAGE_W, self.y, text, fontsize=15,
                     fontweight="bold", va="top", color="#0b3d66")
        self.y -= 0.008
        self.ax.plot([MARGIN_IN / PAGE_W, 1 - MARGIN_IN / PAGE_W],
                     [self.y, self.y], color="#0b3d66", linewidth=0.8,
                     transform=self.ax.transAxes)
        self.y -= 0.028
        self.txt_lines.append("")
        self.txt_lines.append("=" * 78)
        self.txt_lines.append(text)
        self.txt_lines.append("=" * 78)

    def h2(self, text: str):
        self._ensure(0.045)
        self.ax.text(MARGIN_IN / PAGE_W, self.y, text, fontsize=11.5,
                     fontweight="bold", va="top", color="#1a1a1a")
        self.y -= 0.028
        self.txt_lines.append("")
        self.txt_lines.append("-- " + text)

    def para(self, text: str, size: int = 9, mono: bool = False,
             indent: float = 0.0, color: str = "black", width: int = 108,
             bold: bool = False):
        w = width if not mono else int(width * 0.92)
        for raw_line in text.split("\n"):
            wrapped = textwrap.wrap(raw_line, width=w) or [""]
            for ln in wrapped:
                lh = LINE_IN.get(size, 0.02)
                self._ensure(lh)
                self.ax.text(MARGIN_IN / PAGE_W + indent, self.y, ln,
                             fontsize=size, va="top", color=color,
                             fontweight=("bold" if bold else "normal"),
                             family="monospace" if mono else "sans-serif")
                self.y -= lh
                self.txt_lines.append(" " * int(indent * 60) + ln)
        self.y -= 0.006

    def bullet(self, text: str, size: int = 9):
        # a plain hyphen rather than a unicode bullet - the .txt companion
        # should read cleanly in any editor/terminal without depending on
        # UTF-8 auto-detection (Windows PowerShell 5.1's Get-Content in
        # particular garbles unicode without a BOM)
        self.para(f"-  {text}", size=size, indent=0.015, width=104)

    def kv(self, key: str, value: str, size: int = 9):
        self.para(f"{key}: {value}", size=size, mono=False)

    def spacer(self, h: float = 0.015):
        self.y -= h

    def close(self):
        self._flush_page()
        OUT_TXT.write_text("\n".join(self.txt_lines), encoding="utf-8")


# =============================================================================
# Live snapshot helpers - best-effort; missing data prints "not available"
# rather than crashing the report generator.
# =============================================================================
def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                             # noqa: BLE001
        return None


def _tail(path: Path, n: int = 6) -> list[str]:
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return lines[-n:]
    except Exception:                                             # noqa: BLE001
        return []


def _gpu_info() -> dict:
    info = {"torch": "not available", "cuda": False, "device": "n/a", "vram_gb": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            cap = torch.cuda.get_device_capability(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
    except Exception as e:                                        # noqa: BLE001
        info["error"] = str(e)
    return info


def _disk_free_gb(path: Path) -> float:
    import shutil
    return round(shutil.disk_usage(path).free / 1e9, 1)


def _preprocess_progress() -> dict:
    root = ROOT / "data" / "cache"
    prog = root / "preprocess_progress.csv"
    total_lines = None
    if prog.exists():
        try:
            with open(prog, "r", newline="") as f:
                total_lines = max(0, sum(1 for _ in f) - 1)
        except Exception:                                         # noqa: BLE001
            pass
    log_path = ROOT / "data" / "_preprocess_full.log"
    log_tail = _tail(log_path, 4)
    # NOTE: data/cache/meta.csv existing is NOT a reliable "done" signal on
    # this project - an earlier small-scale pilot run left one there before
    # this session's full-scale run even started, and the current resumable
    # 02_preprocess.py intentionally does not delete a stale meta.csv from an
    # unrelated prior run (only its own progress/failed-index files). The
    # live log's own "[preproc] completed" line, printed exactly once right
    # before THIS run writes meta.csv, is the actual signal.
    full_log = log_path.read_text(errors="ignore") if log_path.exists() else ""
    cache_complete = "[preproc] completed" in full_log
    return {"images_done": total_lines, "log_tail": log_tail, "cache_complete": cache_complete}


def _lesion_pretrain_status() -> dict:
    hist = _read_json(ROOT / "outputs" / "lesion_pretrain" / "history.json")
    return {"epochs_run": len(hist) if hist else 0,
           "best_dice": max((h.get("dice_mean", float("nan")) for h in hist), default=None)
           if hist else None,
           "encoder_exists": (ROOT / "outputs" / "lesion_pretrain" / "lesion_encoder.pt").exists()}


def _training_status() -> dict:
    ckpt = ROOT / "outputs" / "retfound_plus_laft_xai" / "checkpoint.pt"
    hist = _read_json(ROOT / "outputs" / "retfound_plus_laft_xai" / "history.json")
    return {"checkpoint_exists": ckpt.exists(),
           "epochs_completed": len(hist) if hist else 0,
           "last_epoch": hist[-1] if hist else None}


# =============================================================================
def build(pdf: PdfPages):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    D = Doc(pdf, footer=f"DR pipeline session report - generated {now}")

    D.title("Diabetic Retinopathy Pipeline",
            "Implementation & Session Documentation")
    D.para(f"Generated: {now}", size=9, color="#555555")
    D.para(f"Machine: {platform.node()} ({platform.system()} {platform.release()})",
          size=9, color="#555555")
    D.para(f"Project root: {ROOT}", size=9, color="#555555", mono=True)
    D.spacer(0.02)
    D.para("This document records the work done to take the rebuilt dr/data/ "
          "package from an untested checkout to a running, resumable, full-"
          "scale training pipeline on this machine - every real bug found "
          "along the way, the checkpoint/resume architecture added because "
          "this machine cannot stay powered on for the full multi-day run, "
          "the CSV export additions, and a live snapshot of where the "
          "pipeline stands as of the timestamp above. Re-run this script "
          "any time to refresh that snapshot.", size=9.5)

    # ------------------------------------------------------------------
    D.h1("1. Environment Setup")
    D.para("The project ships requirements.txt pinned to torch>=2.2 with no "
          "Python version constraint; this machine's only available "
          "interpreter is Python 3.14, which is new enough that the default "
          "PyTorch CUDA wheel index (cu121) has no build for it at all.")
    D.bullet("Created .venv on the system Python 3.14 interpreter.")
    D.bullet("Installed PyTorch 2.11.0 + torchvision from the cu128 wheel "
            "index (download.pytorch.org/whl/cu128) - the cu121/cu124 "
            "indices either lack a 3.14 build or lack Blackwell (RTX 50-"
            "series) kernel support.")
    D.bullet("Installed the remainder of requirements.txt (timm, opencv-"
            "python-headless, scikit-learn/-image, pandas, matplotlib, "
            "onnx/onnxruntime, kaggle) - all resolved cleanly for 3.14.")
    D.bullet("Placed the user-supplied Kaggle API credentials at "
            "%USERPROFILE%\\.kaggle\\kaggle.json so scripts/11_download_"
            "lesions.py could fetch the real IDRiD and DDR lesion-"
            "annotation corpora.")

    # ------------------------------------------------------------------
    D.h1("2. GPU Functionality")
    gpu = _gpu_info()
    D.para("Verified end-to-end before committing any real compute time: "
          "package installation alone does not prove the CUDA kernels "
          "actually match this GPU's architecture.")
    D.kv("PyTorch build", gpu.get("torch", "n/a"))
    D.kv("CUDA available", str(gpu.get("cuda")))
    D.kv("Device", gpu.get("device", "n/a"))
    D.kv("VRAM", f"{gpu.get('vram_gb')} GB" if gpu.get("vram_gb") else "n/a")
    D.kv("Compute capability", gpu.get("capability", "n/a"))
    D.spacer(0.01)
    D.para("Verification performed: allocated a tensor on the CUDA device "
          "and ran a real matrix multiply on it (not just an availability "
          "check) - confirmed the cu128 build's sm_120 kernels execute "
          "correctly on this Blackwell-architecture card, which is not "
          "guaranteed by an older CUDA build simply importing without error.",
          size=9)
    D.spacer(0.01)
    D.para("Constraint this GPU imposes on the rest of the run: the "
          "project's own full-schedule script (run_gpu_full.sh) assumes "
          ">=16GB VRAM at batch-size 32 for a ViT-Large backbone across a "
          "448px global view plus 6 local crops. This card has 8GB, so "
          "the training stage uses --batch-size 4 --grad-accum 8 "
          "--grad-checkpoint (same effective batch of 32, same total "
          "compute, more but smaller steps) as the starting point - see "
          "resume_pipeline.ps1.", size=9)
    D.spacer(0.01)
    D.para("Observed throughput on this GPU/CPU combination so far:", size=9)
    D.bullet("Preprocessing (CPU-bound, 24 worker processes): ~1.7-2.0 "
            "images/sec sustained on real 1024px EyePACS images.")
    D.bullet("Lesion-expert pretraining (GPU-bound, small model): ~8-9 "
            "images/sec at 448px on the real IDRiD/DDR crop cache.")
    D.bullet("Full staged training throughput has not been measured yet - "
            "preprocessing must finish first, since training reads from "
            "its cache.")

    # ------------------------------------------------------------------
    D.h1("3. Real Bugs Found and Fixed")
    D.para("This code had clearly never been run end-to-end against real, "
          "full-scale data before - every one of these surfaced only once "
          "actual EyePACS/IDRiD/DDR data and a real training step were "
          "exercised, not from the project's own synthetic self-test.",
          size=9)

    bugs = [
        ("src/dr/data/eyepacs.py",
         "build_manifest() only searched a few hardcoded subfolders for "
         "images and never found this machine's actual layout "
         "(resized_train/resized_train/); it silently found 0 of the "
         "35,126 images.",
         "Added the actual nested layout (and its _cropped variant) to "
         "the search list."),
        ("src/dr/data/lesion_datasets.py",
         "Folder discovery picked the FIRST matching image/mask folder "
         "found anywhere in the tree. For IDRiD (three sections, each "
         "with its own 'Original Images' folder) this paired images from "
         "the wrong section against the mask folder; for DDR (train/test/"
         "valid splits) it paired one split's images against another "
         "split's masks. Both cases produced 0 (image, mask) pairs.",
         "Rewrote discovery to merge every matching folder by filename "
         "stem instead of picking one arbitrarily. Now correctly finds "
         "all 81 IDRiD and 757 DDR annotated images."),
        ("scripts/08_selftest.py",
         "The crop-centre assertion expected normalised 0-1 coordinates, "
         "but generate_lesion_crops() returns pixel coordinates in the "
         "cache's own pixel grid (confirmed against how 02_preprocess.py "
         "and CachedEyePACS actually consume the values) - a self-test "
         "bug, not a production bug.",
         "Fixed the assertion to check against the cache_size bound "
         "instead of [0,1]."),
        ("src/dr/modules/a4_backbone_lora.py",
         "When require_retfound=False and no checkpoint could be "
         "resolved at all, load_retfound_weights() returned a partial "
         "report dict; the status-string builder unconditionally read "
         "fields (matched_tensors, etc.) that only exist on a successful "
         "load, crashing with KeyError.",
         "Branch on whether the load actually succeeded before building "
         "the status string; fall back to a clear 'not loaded' message "
         "otherwise."),
        ("scripts/10_lesion_pretrain.py",
         "Called train_ds.lesion_prevalence() as a method; it is a "
         "@property already returning a dict, so calling it raised "
         "TypeError. The print statement right behind it then zipped a "
         "6-name tuple against the resulting 4-key dict, which would "
         "have crashed again on the next attempt. Separately, the "
         "training loop read batch keys 'lesion_masks'/'presence' that "
         "the dataset never produces (it yields 'masks'/'valid', with no "
         "separate presence label at all).",
         "Removed the erroneous call, fixed the print to iterate the "
         "dict directly, and derived presence from the masks "
         "(per-channel: any positive pixel in the crop)."),
        ("src/dr/modules/external_lesion_batches.py and "
         "scripts/05_predict.py",
         "Both independently treated generate_lesion_crops()'s pixel-"
         "space centres as 0-1 fractions and multiplied by image height "
         "again, pushing every centre out of range. Since both clip "
         "bounds are equal in that regime, every crop for a given image "
         "collapsed onto the same fixed corner regardless of where the "
         "annotated lesion actually was - this fed both the real-mask "
         "auxiliary training path and the Objective 4 XAI evidence path.",
         "Removed the erroneous re-multiplication in both files."),
        ("scripts/03_train.py",
         "The training loop read a batch key called 'index' that "
         "CachedEyePACS never produces (it's called 'cache_row') - would "
         "have crashed on the first training step.",
         "Changed the read to 'cache_row'."),
        ("scripts/03_train.py",
         "HardExampleState.empty(len(train_ds.meta)) sized the hard-"
         "example-mining arrays to the TRAIN-SPLIT count (~28,102), but "
         "they are indexed by cache_row, a global row index assigned "
         "before the train/val/test split (so it ranges over the whole "
         "~35,126-row cache). Any sample whose cache_row fell outside "
         "the narrower train-count range would index out of bounds.",
         "Sized the state to len(train_ds.meta_full) instead, matching "
         "how CachedEyePACS already sizes its own internal hardness "
         "buffer for the same reason."),
    ]
    for i, (loc, problem, fix) in enumerate(bugs, 1):
        D.h2(f"3.{i}  {loc}")
        D.para("Problem: " + problem, size=8.8, indent=0.01)
        D.para("Fix: " + fix, size=8.8, indent=0.01, color="#0b3d66")

    # ------------------------------------------------------------------
    D.h1("4. Checkpoint / Resume Architecture")
    D.para("This machine cannot stay powered on for the multi-hour "
          "preprocessing pass or the multi-day training run, so neither "
          "script could originally survive an interruption - a shutdown "
          "mid-run meant restarting from image 1 or epoch 1. Both now "
          "checkpoint and auto-resume.", size=9.5)

    D.h2("4.1  scripts/02_preprocess.py")
    D.bullet("Writes one row to preprocess_progress.csv per successfully "
            "processed image (flushed + fsynced every 200 images), and a "
            "separate preprocess_failed.json for permanently-unreadable/"
            "errored images so they are excluded from the dataset "
            "without being retried forever on every resume.")
    D.bullet("On startup: if the cache files already exist at the exact "
            "shape this invocation implies, opens them in r+ mode "
            "(instead of overwriting) and skips every already-completed "
            "or already-failed index.")
    D.bullet("meta.csv (the file training actually reads) is written to "
            "a .tmp path and atomically renamed into place, so a crash "
            "mid-write can never leave a corrupt/partial meta.csv.")
    D.bullet("Verified live: stopped the real run after ~400 images, "
            "restarted the exact same command, and confirmed it printed "
            "'400/35126 already cached, continuing with the remainder' "
            "instead of reprocessing from scratch.")

    D.h2("4.2  scripts/03_train.py")
    D.bullet("Saves a full checkpoint.pt after EVERY completed epoch "
            "(not only new-best epochs): model weights, optimizer state, "
            "LR-scheduler state, EMA shadow weights, the hard-example-"
            "mining state, training history, best/best_epoch/stale, and "
            "the torch/CUDA/numpy RNG states - written to a .tmp file and "
            "atomically renamed, so a shutdown mid-save cannot corrupt "
            "the one checkpoint a resume depends on.")
    D.bullet("On restart: reconstructs the LoRA adapter architecture "
            "deterministically from the saved per-block ranks (bypassing "
            "the RNG-sensitive GLA-LoRA calibration pass entirely, since "
            "the LoRA layers are real injected modules that "
            "model.load_state_dict() needs to already exist), restores "
            "every piece of state above, skips any fully-completed "
            "stage, and resumes the partially-completed stage at the "
            "next epoch with its original optimizer/scheduler state.")
    D.bullet("Worst-case loss from an unplanned shutdown: the single "
            "epoch in progress, never more.")

    D.h2("4.3  scripts/10_lesion_pretrain.py")
    D.para("Given the same per-epoch checkpoint treatment for "
          "consistency, though its crop-cache reuse already made a full "
          "re-run cheap.", size=9)

    D.h2("4.4  resume_pipeline.ps1")
    D.para("A single launcher at the project root chaining self-test -> "
          "preprocessing -> lesion pretraining -> staged training -> the "
          "full evaluation/reporting pipeline. Safe to re-run after any "
          "restart - each stage auto-detects and continues from its own "
          "checkpoint. Usage:", size=9)
    D.para("powershell -ExecutionPolicy Bypass -File resume_pipeline.ps1",
          mono=True, size=9, indent=0.01)
    D.para("Caveat: once training has written its first checkpoint, the "
          "arguments inside this script (particularly --batch-size/"
          "--grad-accum) should not change - a resumed run reuses the "
          "saved LR-schedule state, which a change would throw off. The "
          "final evaluation/report stages are not individually "
          "checkpointed (only the two multi-hour stages are); if one of "
          "those is interrupted, only that one step needs re-running.",
          size=8.8, color="#555555")

    # ------------------------------------------------------------------
    D.h1("5. CSV Export Additions")
    D.para("A shared helper, src/dr/csv_export.py "
          "(save_csv_alongside), writes a best-effort tabular CSV next "
          "to whichever JSON a script already produces, wherever the "
          "data has one obvious shape: a list of flat dicts becomes one "
          "row per entry; a dict of dicts becomes one named row per key; "
          "a flat dict becomes a single summary row. Verified against "
          "each of these shapes before wiring it into every script below.",
          size=9)
    csv_scripts = [
        ("03_train.py", "history.csv (per-epoch curve), gla_lora.csv (per-block "
                        "LoRA ranks/importance)"),
        ("10_lesion_pretrain.py", "history.csv, sources.csv"),
        ("07_compare_models.py", "comparison.csv (QWK-sorted leaderboard), "
                                 "results.csv (every field), significance.csv"),
        ("04_evaluate.py", "evaluation.csv (long format: group/metric/value), "
                           "evaluation_confusion_matrix.csv"),
        ("06_deploy.py", "distilled_metrics.csv, distilled_variants.csv"),
        ("09_screening.py", "screening.csv, screening_deployable.csv"),
        ("12_lesion_visibility.py", "lesion_visibility.csv"),
        ("13_linear_probe.py", "linear_probe_<tag>.csv"),
        ("14/15/18/20/22_*.py", "one CSV per objective evidence-pack JSON"),
        ("16_ordinal_calibration.py", "ordinal_calibration_results.csv"),
        ("19_objective3_validation.py", "objective3_validation_arms.csv"),
        ("21_xai_evaluation.py", "xai_evaluation_per_image.csv, "
                                 "xai_evaluation_metrics.csv"),
        ("23_objectives_summary.py", "objectives_consolidated.csv"),
    ]
    for script, files in csv_scripts:
        D.bullet(f"{script}: {files}", size=8.8)

    # ------------------------------------------------------------------
    D.h1("6. Current Pipeline Status (live at generation time)")
    pp = _preprocess_progress()
    lp = _lesion_pretrain_status()
    tr = _training_status()
    free_gb = _disk_free_gb(ROOT)

    D.h2("6.1  Preprocessing")
    if pp["cache_complete"]:
        D.para("COMPLETE - data/cache/meta.csv has been written.", size=9.5,
              color="#0b6b2d")
    elif pp["images_done"] is not None:
        D.kv("Images cached so far", f"{pp['images_done']:,} / 35,126")
    else:
        D.para("Not started, or progress file not found.", size=9)
    if pp["log_tail"]:
        D.para("Last log lines:", size=8.5, color="#555555")
        D.para("\n".join(pp["log_tail"]), mono=True, size=7.6, indent=0.01)

    D.h2("6.2  Lesion-Expert Pretraining")
    if lp["encoder_exists"]:
        bd = f"{lp['best_dice']:.4f}" if lp["best_dice"] is not None else "n/a"
        D.para(f"COMPLETE - {lp['epochs_run']} epochs run, best validation "
              f"Dice {bd}. Checkpoint: outputs/lesion_pretrain/"
              f"lesion_encoder.pt", size=9.5, color="#0b6b2d")
    else:
        D.para("Not yet complete.", size=9)

    D.h2("6.3  Staged Training")
    if tr["checkpoint_exists"]:
        D.para(f"IN PROGRESS / RESUMABLE - {tr['epochs_completed']} epoch(s) "
              f"checkpointed so far.", size=9.5)
        if tr["last_epoch"]:
            le = tr["last_epoch"]
            D.para(f"Last completed epoch: {le.get('epoch')}  stage={le.get('stage')}  "
                  f"QWK={le.get('val_quadratic_weighted_kappa', float('nan')):.4f}  "
                  f"score={le.get('selection_score', float('nan')):.4f}",
                  mono=True, size=8.5, indent=0.01)
    else:
        D.para("Not started - waiting on preprocessing to finish.", size=9)

    D.h2("6.4  Disk / Environment")
    D.kv("Free disk space", f"{free_gb:,.1f} GB")
    D.kv("GPU", f"{gpu.get('device','n/a')} ({gpu.get('vram_gb')} GB VRAM)")

    # ------------------------------------------------------------------
    D.h1("7. Next Steps")
    D.bullet("Let preprocessing finish (full 35,126-image, 1024px cache).")
    D.bullet("Start scripts/03_train.py with --lesion-encoder pointed at "
            "the completed lesion_encoder.pt (already wired into "
            "resume_pipeline.ps1), starting at --batch-size 4 --grad-"
            "accum 8 --grad-checkpoint for the 8GB card.")
    D.bullet("If training OOMs before its first checkpoint, lower "
            "--batch-size further and raise --grad-accum to compensate - "
            "safe to do at that point since nothing has been "
            "checkpointed yet.")
    D.bullet("Once training completes all four stages, run the full "
            "evaluation/reporting sequence (already chained in "
            "resume_pipeline.ps1) and read outputs/objectives_"
            "consolidated.json / .csv first.")
    D.bullet("Re-run this script (generate_session_report.py) at any "
            "point to refresh this document with current progress.")

    # ------------------------------------------------------------------
    _append_experiment_values_registry(D)

    D.close()


def _append_experiment_values_registry(D: "Doc") -> None:
    """Appendix: the full experiment-values registry (data/experiment_values.json,
    produced by generate_experiment_values.py) - every real, verified value this
    codebase and this run produce, plus explicit PENDING/NOT-APPLICABLE markers
    for anything else. Generic renderer so this section always reflects
    whatever the registry currently contains, with no manual upkeep here."""
    reg_path = ROOT / "data" / "experiment_values.json"
    reg = _read_json(reg_path)
    if not reg:
        return
    D.h1("Appendix: Experiment Values Registry")
    D.para(f"Source: {reg_path.relative_to(ROOT)}, generated {reg.get('generated', '?')}. "
          "Every value below is read from a real config/log/output file at "
          "generation time; anything not yet computable is marked PENDING, and "
          "anything this codebase does not implement is marked NOT APPLICABLE, "
          "rather than estimated or invented.", size=9)
    for key, value in reg.items():
        if key == "generated":
            continue
        _render_registry_value(D, key.replace("_", " "), value, depth=0)


def _render_registry_value(D: "Doc", label: str, value, depth: int) -> None:
    indent = 0.015 * depth
    if isinstance(value, dict):
        if depth == 0:
            D.h2(label)
        else:
            D.para(label + ":", size=8.8, bold=True, indent=indent)
        for k, v in value.items():
            _render_registry_value(D, k.replace("_", " "), v, depth + 1)
    elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        D.para(label + ":", size=8.8, bold=True, indent=indent)
        cols = list(value[0].keys())
        D.para(" | ".join(cols), mono=True, size=7.0, indent=indent + 0.015)
        for row in value:
            D.para(" | ".join(str(row.get(c, "")) for c in cols),
                  mono=True, size=6.8, indent=indent + 0.015)
    elif isinstance(value, list):
        D.para(f"{label}: {value}", size=8.4, indent=indent)
    else:
        D.para(f"{label}: {value}", size=8.4, indent=indent)


def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        build(pdf)
    print(f"[saved] {OUT_PDF}")
    print(f"[saved] {OUT_TXT}")


if __name__ == "__main__":
    main()
