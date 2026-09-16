#!/usr/bin/env bash
# Objective 3, the adaptation arm.
#
# The linear probe in step 13 measures a FROZEN representation. Objective 3 is
# not a claim about frozen representations - it says a large model can be
# "adapted efficiently, enabling lightweight optimization without full
# retraining". The instrument that matches that claim is a matched
# parameter-efficient fine-tune, not a probe, and the distinction is not
# cosmetic: MAE backbones are documented to underperform supervised ones under
# linear probing while matching or beating them once any adaptation is allowed
# (He et al. 2022, sec. 4.1 and fig. 9). RETFound is an MAE.
#
# So this runs the same GLA-LoRA adaptation twice, changing exactly one thing:
#
#   arm A   RETFound MAE ViT-L/16 initialisation
#   arm B   ImageNet ViT-L/16 initialisation (--allow-no-retfound --pretrained)
#
# Everything else is pinned identical - architecture, rank allocation policy,
# stage schedule, learning rates, sampler, loss weights, data subset, batch
# size, crop geometry, validation split and selection criterion. Both arms keep
# the backbone frozen throughout (--no-unfreeze), so the comparison is of
# initialisations under an identical, lightweight adaptation budget.
#
#   ./run_objective3.sh
#
set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
mkdir -p outputs/logs

BACKBONE="vit_large_patch16_224"
RETFOUND="data/checkpoints/RETFound_MAE/pytorch_model.bin"
STAGES="${STAGES:-1,2}"              # 1 frozen epoch, then 2 GLA-LoRA epochs.
                                     # Sized so the LoRA stage - the thing under
                                     # test - actually trains inside the budget:
                                     # measured throughput is ~0.5 img/s, so
                                     # 500 samples/epoch is ~17 min/epoch.
SPE="${SPE:-500}"                    # sampled images per epoch
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"                  # effective batch 16
NCROPS="${NCROPS:-2}"                # matches the probe's crop budget
MAXVAL="${MAXVAL:-400}"
BUDGET="${BUDGET:-50}"               # minutes per arm

ENC=""
[ -f outputs/lesion_pretrain/lesion_encoder.pt ] && \
  ENC="--lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt"

train_arm () {                       # $1 tag   $2... init flags
  local tag="$1"; shift
  echo; echo "=============================================================="
  echo " train  $tag"; echo "=============================================================="; date
  $PY -u scripts/03_train.py \
      --tag "$tag" --backbone "$BACKBONE" "$@" \
      --no-unfreeze --select ordinal \
      --stage-epochs "$STAGES" --samples-per-epoch "$SPE" \
      --batch-size "$BATCH" --grad-accum "$ACCUM" \
      --global-size 224 --crop-input 224 --n-crops "$NCROPS" \
      --max-val "$MAXVAL" --time-budget-min "$BUDGET" $ENC \
      > "outputs/logs/${tag}_train.log" 2>&1
  local rc=$?
  echo "  exit $rc"
  grep -E "^\[A4\] (backbone|RETFound|allocated)|^\[stage|^\[s[0-9]|saved new best" \
       "outputs/logs/${tag}_train.log" | tail -20
  return $rc
}

eval_arm () {
  local tag="$1"
  echo; echo "--- evaluate $tag ---"; date
  $PY -u scripts/04_evaluate.py --ckpt "outputs/$tag/best.pt" \
      --xai-samples 40 --corruption-samples 120 --mc-samples 4 \
      > "outputs/logs/${tag}_eval.log" 2>&1
  echo "  exit $?"
  grep -E "accuracy|quadratic_weighted|auc_sight|auc_referable|f1_macro" \
       "outputs/logs/${tag}_eval.log" | head -8
}

train_arm obj3_retfound --retfound-ckpt "$RETFOUND"
train_arm obj3_imagenet --retfound-ckpt none --allow-no-retfound --pretrained

eval_arm obj3_retfound
eval_arm obj3_imagenet

echo; echo "OBJECTIVE 3 ADAPTATION ARM COMPLETE"; date
