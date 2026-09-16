#!/usr/bin/env bash
# Train RETFound Plus-LAFT-XAI inside a 50% hardware envelope, selecting the
# checkpoint on referral safety rather than on grading accuracy.
#
#   ./run_50pct.sh                  train, evaluate, screening report
#   TRAIN_BUDGET_MIN=120 ./run_50pct.sh    shorter run
#
# What "50%" means here, measured on this machine (8 GB M3, 8 cores):
#
#   memory   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 -> a hard 2.67 GiB ceiling on
#            accelerator allocations.  This is a real cap, not a hint: the
#            process raises rather than swapping, which is what keeps the rest
#            of the machine usable.
#   cores    4 of 8 (2 of 4 performance cores' worth of intra-op threads),
#            plus nice 10 so foreground work always wins the scheduler.
#   disk     reads the existing 1024 px cache; writes nothing large.
#
# The 2.67 GiB ceiling is what picks the model.  Measured peak, batch 2, one
# global view plus six local crops:
#
#   RETFound ViT-L, frozen + GLA-LoRA   2.56 GB   1.75 img/s   fits
#   ViT-Base,  LoRA + 33% unfreeze      OOM at 2.63 GB
#   ViT-Small, full unfreeze            OOM at 2.62 GB
#   RETFound ViT-L + 33% unfreeze       OOM at 3.05 GB
#
# So the largest and best-initialised backbone is also the only one that fits,
# provided the backbone stays frozen and only the LoRA adapters and heads move.
# --no-unfreeze enforces that for every stage while keeping the staged LR decay.
set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
mkdir -p outputs/logs

# ---- 50% of the machine ---------------------------------------------------
CORES=$(( $(sysctl -n hw.ncpu) / 2 ))
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export VECLIB_MAXIMUM_THREADS=$CORES
# 50% of the machine's 8 GB unified memory = 4.0 GB.  The MPS watermark is a
# fraction of Metal's *recommended working set* (5.73 GB here), not of installed
# RAM, so the ratio that means "half the machine" is 4.0/5.73 = 0.70 - not 0.50.
# At 0.50 the ceiling is 2.67 GiB, and an fp32 ViT-Large needs 2.22 GB of driver
# memory for 1.27 GB of live weights (the allocator fragments the 294-tensor
# transfer), which leaves too little for activations at any batch or crop count.
MEM_FRACTION="${MEM_FRACTION:-0.5}"
RATIO=$($PY -c "import torch;print(f'{$MEM_FRACTION*$(sysctl -n hw.memsize)/torch.mps.recommended_max_memory():.3f}')")
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=$RATIO
export PYTORCH_MPS_LOW_WATERMARK_RATIO=$($PY -c "print(f'{$RATIO*0.8:.3f}')")
export TOKENIZERS_PARALLELISM=false
NICE="nice -n 10"

# ---- run configuration ----------------------------------------------------
TAG="${TAG:-screening_npv985}"
# RETFound is published only as a ViT-Large.  A smaller backbone therefore has
# to fall back to ImageNet, which is an explicit ablation, not a silent default.
BACKBONE="${BACKBONE:-vit_large_patch16_224}"
RETFOUND="${RETFOUND:-data/checkpoints/RETFound_MAE/pytorch_model.bin}"
NPV_TARGET="${NPV_TARGET:-0.985}"
STAGES="${STAGES:-2,6,5,4}"          # all head+LoRA; the stages only decay LR
SPE="${SPE:-2500}"                   # sampled images per epoch
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"                  # effective batch 16
GLOBAL_PX=224
CROP_PX="${CROP_PX:-320}"
CROP_IN=224
N_CROPS="${N_CROPS:-6}"
MAX_VAL="${MAX_VAL:-}"                # blank = the full 920-image val split
TRAIN_BUDGET_MIN="${TRAIN_BUDGET_MIN:-600}"
OUT="outputs/$TAG"

step () { echo; echo "=============================================================="
          echo " $1"; echo "=============================================================="; date; }

step "0  hardware envelope"
echo "  cores            $CORES / $(sysctl -n hw.ncpu)  (nice 10)"
echo "  memory ceiling   $($PY -c "import torch;print(f'{torch.mps.recommended_max_memory()*$RATIO/1e9:.2f} GB')") of $(( $(sysctl -n hw.memsize) / 1073741824 )) GB installed (ratio $RATIO)"
echo "  free disk        $(df -h . | awk 'NR==2{print $4}')"
echo "  selection        cleared % at NPV >= $NPV_TARGET (sight-threatening DR)"
echo "  time budget      $TRAIN_BUDGET_MIN min"

step "1  staged training (frozen -> GLA-LoRA, backbone never unfrozen)"
ENC=""
[ -f outputs/lesion_pretrain/lesion_encoder.pt ] && \
  ENC="--lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt"
$NICE $PY -u scripts/03_train.py \
    --tag "$TAG" \
    --backbone "$BACKBONE" --retfound-ckpt "$RETFOUND" ${ALLOW_NO_RETFOUND:+--allow-no-retfound --pretrained} \
    --no-unfreeze --threads "$CORES" \
    --select screening --npv-target "$NPV_TARGET" \
    --stage-epochs "$STAGES" --samples-per-epoch "$SPE" \
    --batch-size "$BATCH" --grad-accum "$ACCUM" \
    --calib-batch 2 --calib-device cpu ${GRAD_CKPT:+--grad-checkpoint} \
    --global-size "$GLOBAL_PX" --crop-input "$CROP_IN" --n-crops "$N_CROPS" \
    ${MAX_VAL:+--max-val $MAX_VAL} \
    --time-budget-min "$TRAIN_BUDGET_MIN" $ENC \
    2>&1 | tee "outputs/logs/${TAG}_train.log" \
    | grep -E "^\[|screening:|saved new best|Error|error:|Traceback"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo; echo "!! training failed - last 30 lines of outputs/logs/${TAG}_train.log:"
  tail -30 "outputs/logs/${TAG}_train.log"
  exit 1
fi

step "2  full evaluation on the held-out test split"
$NICE $PY -u scripts/04_evaluate.py --ckpt "$OUT/best.pt" \
    > "outputs/logs/${TAG}_eval.log" 2>&1
tail -40 "outputs/logs/${TAG}_eval.log"

step "3  screening triage - thresholds fitted on val, frozen, applied to test"
EXT=""
[ -d data/cache_external ] && EXT="--external-cache data/cache_external"
$NICE $PY -u scripts/09_screening.py --ckpt "$OUT/best.pt" \
    --batch-size 2 --npv-target "$NPV_TARGET" $EXT \
    2>&1 | tee "outputs/logs/${TAG}_screening.log" | tail -60

step "DONE"
echo "artefacts in $OUT"
