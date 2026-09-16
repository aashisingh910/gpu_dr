#!/usr/bin/env bash
# run_gpu_full.sh - the complete pipeline at full scale, for a real GPU box.
#
# Differences from every other run_*.sh in this repo: no --limit anywhere,
# native 1024px cache, the full 5/12/14/14 stage schedule, a GPU-sized batch,
# and every evaluation/comparison script this repo has - not a subset.
#
# This is NOT a quick smoke test. On a single modern GPU (>=16GB VRAM),
# expect preprocessing to take a few hours (35,126 images at 1024px) and the
# full staged training schedule to take on the rough order of the ~40
# GPU-hours this project's own config.py defaults assume - budget accordingly.
# Run scripts/08_selftest.py (Step 0 below) before committing to the rest.
#
# Usage:
#   ./run_gpu_full.sh              # everything, in order
#   ./run_gpu_full.sh from-train   # skip data download/preprocess, start at training
#   ./run_gpu_full.sh eval-only    # skip straight to evaluation + all reports
set -euo pipefail

PYTHON="${PYTHON:-python3}"
STAGE="${1:-all}"

section () { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

run_from_start=true
run_from_train=false
if [[ "$STAGE" == "from-train" ]]; then run_from_start=false; fi
if [[ "$STAGE" == "eval-only" ]]; then run_from_start=false; run_from_train=false; fi

# ---------------------------------------------------------------------------
section "Step 0 - self-test (always run this, it costs seconds)"
$PYTHON scripts/08_selftest.py

if [[ "$STAGE" != "eval-only" && "$STAGE" != "from-train" ]]; then
  section "Step 1a - download real EyePACS at native 1024px resolution"
  $PYTHON scripts/01_download_data.py --dataset eyepacs-hi

  section "Step 1b - download real IDRiD + DDR lesion annotations"
  $PYTHON scripts/11_download_lesions.py

  section "Step 2 - preprocess: A1 + A2 + lesion/anatomy priors, FULL corpus, NO --limit"
  # No --limit: every one of the 35,126 images. This needs ~110GB free disk
  # (the script itself will refuse to start and tell you the shortfall if
  # there isn't room - lower --image-size rather than silently subsampling
  # if you hit that wall).
  $PYTHON scripts/02_preprocess.py --root data/raw/eyepacs-hi \
      --image-size 1024 --n-crops 6 --crop-size 320 \
      --workers "$(nproc)"

  section "Step 2b - pretrain the A3 lesion experts on real IDRiD/DDR masks"
  $PYTHON scripts/10_lesion_pretrain.py --image-size 448
fi

if [[ "$STAGE" != "eval-only" ]]; then
  section "Step 3 - staged training, FULL schedule, real RETFound, GPU batch size"
  # --stage-epochs 5,12,14,14 and --samples-per-epoch 6000 are this project's
  # own documented full schedule (config.py's defaults) - not shortened here.
  # batch-size/grad-accum below assume >=16GB VRAM; halve batch-size and
  # double grad-accum if you hit OOM, rather than shrinking the schedule.
  $PYTHON scripts/03_train.py \
      --stage-epochs 5,12,14,14 --samples-per-epoch 6000 --max-val 1500 \
      --batch-size 32 --grad-accum 1 \
      --global-size 448 --crop-input 224 --n-crops 6 \
      --lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt \
      --amp --ema-decay 0.995 \
      --xai-weight 0.05 \
      --aux-lesion-masks --aux-lesion-root data/raw/lesion
fi

# ---------------------------------------------------------------------------
section "Step 4 - full evaluation: all 38 parameters, 7 groups"
$PYTHON scripts/04_evaluate.py --external-cache data/cache_external --tta

section "Step 5 - screening triage (validation-fitted, applied to test + external)"
$PYTHON scripts/09_screening.py --external-cache data/cache_external

section "Step 6 - Objective 1: severity-grading evidence"
$PYTHON scripts/15_objective1_report.py

section "Step 7 - Objective 2: lesion visibility (CLAHE) + Objective 2 report"
$PYTHON scripts/12_lesion_visibility.py --algo clahe
$PYTHON scripts/14_objective2_report.py

section "Step 8 - Objective 3: efficient adaptation (linear probe + validation + complete)"
$PYTHON scripts/13_linear_probe.py --backbone vit_large_patch16_224
$PYTHON scripts/16_ordinal_calibration.py
$PYTHON scripts/17_calibration_report.py
$PYTHON scripts/18_objective3_report.py
$PYTHON scripts/19_objective3_validation.py
$PYTHON scripts/20_objective3_complete.py

section "Step 9 - Objective 4: explainability, scored against REAL IDRiD/DDR masks"
$PYTHON scripts/21_xai_evaluation.py --ref-masks idrid_ddr --counterfactual \
    --lesion-root data/raw/lesion
$PYTHON scripts/22_objective4_report.py

section "Step 10 - consolidated objectives summary + methods reference"
$PYTHON scripts/23_objectives_summary.py
$PYTHON scripts/24_methods_reference.py

section "Step 11 - full model comparison: all baselines + all proposed ablations"
# No --max-train/--max-test caps and no --only filter: every baseline and
# every proposed/ablation entry in the harness, at the full epoch count.
$PYTHON scripts/07_compare_models.py --epochs 3

section "Step 12 - deployment optimization (distillation + FP16/INT8/ONNX)"
$PYTHON scripts/06_deploy.py --epochs 5

section "DONE"
echo "All outputs are under outputs/ - see each script's own printed [saved] path."
echo "Read outputs/*/evaluation.json and outputs/objectives_consolidated.json first."
