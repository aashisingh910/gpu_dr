#!/usr/bin/env bash
# Objective 4 - lesion-grounded explainability, actually exercised.
#
# The finding that motivates this run: lambda_XAI has never been non-zero in
# any recorded run of this repository.  XaiCfg.weight_schedule is
# (0.0, 0.0, 0.02, 0.05), indexed by stage, and no run has ever reached stage 3
# - the flagship stopped in stage 2, the RETFound objectives run died in stage
# 2, vits_objectives ended at s2_lora.  Every history.json carries
# "lambda_xai": 0.0 and every training log prints "xai 0.000@0.000".
#
# So the attribution-consistency term
#
#     L_XAI = 1 - Dice( attribution , lesion mask )
#
# has never been applied.  Measured attribution is at chance (AUROC 0.50,
# concentration ratio 1.0), which is the expected result for a model that was
# never asked to align anything.  Objective 4 has not failed; it has not been
# tested.
#
# This runs the test as a controlled A/B, changing exactly one thing:
#
#     arm A   lambda_XAI = 0     (control - the current behaviour)
#     arm B   lambda_XAI = 0.05  (treatment - the loss switched on)
#
# Everything else is pinned identical: backbone, stage schedule, learning
# rates, sampler, all other loss weights, data subset, batch size, crop
# geometry, validation split and selection criterion.  Both arms then get the
# same threshold-free attribution measurement (scripts/21_xai_evaluation.py),
# so the before/after is on identical metrics over identical images.
#
# ViT-S is used deliberately: it is small enough that both arms fit in a short
# budget, and the question - does switching the loss on move the attribution -
# does not depend on backbone scale.
#
#   ./run_objective4.sh
#
set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
mkdir -p outputs/logs

BACKBONE="${BACKBONE:-vit_small_patch16_224}"
STAGES="${STAGES:-1,2}"
SPE="${SPE:-500}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
NCROPS="${NCROPS:-2}"
MAXVAL="${MAXVAL:-400}"
BUDGET="${BUDGET:-40}"
XAI_ON="${XAI_ON:-0.05}"
NXAI="${NXAI:-120}"          # images in the attribution measurement

ENC=""
[ -f outputs/lesion_pretrain/lesion_encoder.pt ] && \
  ENC="--lesion-encoder outputs/lesion_pretrain/lesion_encoder.pt"

train_arm () {               # $1 tag   $2 lambda_XAI
  local tag="$1" lam="$2"
  echo; echo "=============================================================="
  echo " train  $tag   (lambda_XAI = $lam)"
  echo "=============================================================="; date
  $PY -u scripts/03_train.py \
      --tag "$tag" --backbone "$BACKBONE" \
      --retfound-ckpt none --allow-no-retfound --pretrained \
      --no-unfreeze --select ordinal --xai-weight "$lam" \
      --stage-epochs "$STAGES" --samples-per-epoch "$SPE" \
      --batch-size "$BATCH" --grad-accum "$ACCUM" \
      --global-size 224 --crop-input 224 --n-crops "$NCROPS" \
      --max-val "$MAXVAL" --time-budget-min "$BUDGET" $ENC \
      > "outputs/logs/${tag}_train.log" 2>&1
  echo "  exit $?"
  grep -E "lambda_XAI schedule|^\[s[0-9]|saved new best" \
       "outputs/logs/${tag}_train.log" | tail -12
}

measure_arm () {             # $1 tag
  local tag="$1"
  echo; echo "--- attribution measurement: $tag ---"; date
  $PY -u scripts/21_xai_evaluation.py \
      --ckpt "outputs/$tag/best.pt" --n "$NXAI" --device cpu --counterfactual \
      --out "outputs/xai_${tag}.json" \
      > "outputs/logs/xai_${tag}.log" 2>&1
  echo "  exit $?"
  tail -16 "outputs/logs/xai_${tag}.log"
}

train_arm obj4_xai_off "0"
train_arm obj4_xai_on  "$XAI_ON"

measure_arm obj4_xai_off
measure_arm obj4_xai_on

echo; echo "OBJECTIVE 4 A/B COMPLETE"; date
