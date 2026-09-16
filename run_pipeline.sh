#!/usr/bin/env bash
# Sequential end-to-end run.
#
# Deliberately sequential: on a laptop SSD with 8 GB of unified memory, running
# the download, the 1024 px preprocessing pass and a ViT-Large training loop at
# the same time makes all three I/O-starved and none of them finish. Each stage
# writes its own log under outputs/logs/ and is skipped if its output already
# exists, so the script is safe to re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
mkdir -p outputs/logs

CACHE_PX=1024
GLOBAL_PX=224
CROP_PX=320
CROP_IN=224
N_CROPS=6
LIMIT=${LIMIT:-6000}   # 1024px cache is ~3.2 MB/image; 6000 keeps it near 19 GB
STAGES="${STAGES:-2,4,4,4}"
SPE="${SPE:-2000}"
TRAIN_BUDGET_MIN="${TRAIN_BUDGET_MIN:-330}"

step () { echo; echo "=============================================================="
          echo " $1"; echo "=============================================================="; date; }

step "1  extract the 1024 px EyePACS mirror"
if [ ! -d data/raw/eyepacs-hi ] || [ -z "$(find data/raw/eyepacs-hi -name '*.jpeg' -print -quit 2>/dev/null)" ]; then
  $PY -u scripts/01_download_data.py --dataset eyepacs-hi 2>&1 | tee outputs/logs/01_download.log | tail -5
else
  echo "already extracted"
fi
find data/raw/eyepacs-hi -name "*.jpeg" | head -1 | xargs -I{} $PY -c \
  "import cv2,sys;print('sample image size:', cv2.imread('{}').shape)"

step "2  Q* gate + 1024 px field extraction + priors + lesion crops"
# ~2 h at 1024 px: skip if a cache with matching geometry is already present
if $PY -c "
import sys,numpy as np,pandas as pd,pathlib
d=pathlib.Path('data/cache')
sys.exit(0 if (d/'crops.npy').exists() and (d/'meta.csv').exists()
         and np.load(d/'images.npy', mmap_mode='r').shape[1]==$CACHE_PX
         and np.load(d/'crops.npy', mmap_mode='r').shape[1]==$N_CROPS
         and len(pd.read_csv(d/'meta.csv'))==np.load(d/'images.npy',mmap_mode='r').shape[0]
         else 1)" 2>/dev/null; then
  echo "cache already built at ${CACHE_PX}px with ${N_CROPS} crops - skipping"
else
$PY -u scripts/02_preprocess.py --root data/raw/eyepacs-hi --limit "$LIMIT" \
    --image-size "$CACHE_PX" --n-crops "$N_CROPS" --crop-size "$CROP_PX" \
    --workers 5 > outputs/logs/02_preprocess.log 2>&1
fi
tail -30 outputs/logs/02_preprocess.log

step "2b pretrain the A3 lesion experts on REAL IDRiD + DDR masks"
$PY -u scripts/10_lesion_pretrain.py --image-size "$CROP_IN" --batch-size "${PRETRAIN_BS:-6}" \
    --epochs "${PRETRAIN_EPOCHS:-10}" --crops-per-image 8 > outputs/logs/10_lesion_pretrain.log 2>&1
tail -15 outputs/logs/10_lesion_pretrain.log

step "3  staged training"
ENC=""
[ -f outputs/lesion_pretrain/lesion_encoder.pt ] && \
  ENC="--lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt"
$PY -u scripts/03_train.py --stage-epochs "$STAGES" --samples-per-epoch "$SPE" \
    --max-val 1200 --batch-size 2 --grad-accum 8 \
    --global-size "$GLOBAL_PX" --crop-input "$CROP_IN" --n-crops "$N_CROPS" \
    --time-budget-min "$TRAIN_BUDGET_MIN" $ENC \
    > outputs/logs/03_train.log 2>&1
tail -25 outputs/logs/03_train.log

step "4  full evaluation"
$PY -u scripts/04_evaluate.py --xai-samples 120 > outputs/logs/04_evaluate.log 2>&1
tail -40 outputs/logs/04_evaluate.log

step "9  screening triage"
$PY -u scripts/09_screening.py > outputs/logs/09_screening.log 2>&1
tail -20 outputs/logs/09_screening.log

step "DONE"
