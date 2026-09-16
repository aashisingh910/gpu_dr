#!/usr/bin/env bash
# End-to-end RETFound Plus-LAFT-XAI pipeline on real EyePACS data.
#   ./run_all.sh            full run
#   ./run_all.sh quick      small subset, for a fast smoke test
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
MODE="${1:-full}"

# Resolution policy. The cache holds a 1024 px retinal field; the global branch
# sees a resize of it for context and the local branch sees N windows cut at
# native cache resolution. Lowering CACHE below ~768 defeats the point of the
# whole change - a microaneurysm is a handful of pixels and does not survive it.
CACHE_PX=1024
GLOBAL_PX=224        # context only - the crops carry the lesion detail
CROP_PX=320          # window size in cache pixels
CROP_IN=224          # window size fed to the backbone
N_CROPS=6
# Sampling density per lesion: a 320px window of a 1024px field fed at 224 gives
# 224/320 = 0.70 output px per cache px, against 224/1024 = 0.22 for the global
# view - 3.2x linear, 10x areal. That ratio, not the nominal "1024", is what
# decides whether a microaneurysm survives.

# Defaults are tuned for an 8 GB Apple M3 (MPS). RETFound is a ViT-Large, and
# each sample now costs one global forward plus N_CROPS local ones, so the
# batch has to be small; --grad-accum recovers the effective batch size.
if [ "$MODE" = "quick" ]; then
  LIMIT="--limit 2000"; STAGES="1,1,1,1"; SPE=400;  CMP_TRAIN=1000; CMP_EPOCHS=1
else
  # The published schedule is 5/12/14/14 at 6000 samples/epoch. On an 8 GB M3
  # driving a ViT-Large over one global plus six local views that is ~40 GPU-hours,
  # so the default here is the same four stages at a budget that finishes. Raise
  # STAGES and SPE on a CUDA box - nothing else needs to change.
  LIMIT="--limit 12000"; STAGES="2,4,4,4"; SPE=2000; CMP_TRAIN=2000; CMP_EPOCHS=1
fi
BATCH=2
ACCUM=8

echo "=============================================================="
echo " STEP 0  self-test: every block A1-A10 on synthetic tensors"
echo "=============================================================="
$PY scripts/08_selftest.py

echo "=============================================================="
echo " STEP 1a download real EyePACS 2015 (1024 px mirror) from Kaggle"
echo "=============================================================="
$PY scripts/01_download_data.py --dataset eyepacs-hi

echo "=============================================================="
echo " STEP 1b download real lesion annotations (IDRiD + DDR)"
echo "=============================================================="
$PY scripts/11_download_lesions.py || \
  echo "  (continuing without real lesion masks; the A3 experts will fall back"
echo "   to the morphological priors)"
echo "=============================================================="
echo " STEP 2  A1 Q* gate + A2 field extraction + priors + lesion crops"
echo "=============================================================="
$PY -u scripts/02_preprocess.py --root data/raw/eyepacs-hi $LIMIT \
    --image-size "$CACHE_PX" --n-crops "$N_CROPS" --crop-size "$CROP_PX"

echo "=============================================================="
echo " STEP 2b pretrain the A3 lesion experts on REAL annotations"
echo "=============================================================="
LESION_ENC=outputs/lesion_pretrain/lesion_encoder.pt
$PY -u scripts/10_lesion_pretrain.py --image-size "$GLOBAL_PX" || true

echo "=============================================================="
echo " STEP 3  staged training (frozen -> GLA-LoRA -> unfreeze -> full)"
echo "=============================================================="
ENC=""
[ -f "$LESION_ENC" ] && ENC="--lesion-encoder $LESION_ENC"
$PY -u scripts/03_train.py --stage-epochs "$STAGES" --samples-per-epoch "$SPE" \
    --max-val 1500 --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --global-size "$GLOBAL_PX" --crop-input "$CROP_IN" --n-crops "$N_CROPS" $ENC

echo "=============================================================="
echo " STEP 4  evaluate: 38 parameters across 7 metric groups"
echo "=============================================================="
EXT=""
[ -d data/cache_external ] && EXT="--external-cache data/cache_external"
$PY -u scripts/04_evaluate.py $EXT

echo "=============================================================="
echo " STEP 5  single-image clinical report (A1..A10 end to end)"
echo "=============================================================="
IMG=$(find data/raw/eyepacs-hi -name "*.jpeg" -o -name "*.png" | head -1)
$PY -u scripts/05_predict.py --image "$IMG"

echo "=============================================================="
echo " STEP 6  deployment: distillation, FP16/INT8, ONNX"
echo "=============================================================="
$PY -u scripts/06_deploy.py --epochs 2

echo "=============================================================="
echo " STEP 7  model comparison + component ablations"
echo "=============================================================="
$PY -u scripts/07_compare_models.py --epochs "$CMP_EPOCHS" --max-train "$CMP_TRAIN" \
    --image-size "$GLOBAL_PX"

echo "=============================================================="
echo " STEP 9  screening triage: val-fitted referral operating points"
echo "=============================================================="
$PY -u scripts/09_screening.py $EXT

echo
echo "All steps complete. Artefacts in outputs/"
