# run_objective3.ps1 - Windows port of run_objective3.sh (Objective 3 adaptation arm).
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
# Only change from the original .sh: the venv python path (.venv/bin/python,
# which doesn't exist on Windows, -> .venv/Scripts/python.exe) and bash
# syntax -> PowerShell. Every flag, value and the two-arm structure are
# unchanged.
#
#   powershell -ExecutionPolicy Bypass -File run_objective3.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$PY = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path outputs\logs | Out-Null

$BACKBONE = "vit_large_patch16_224"
$RETFOUND = "data\checkpoints\RETFound_MAE\pytorch_model.bin"
$STAGES = if ($env:STAGES) { $env:STAGES } else { "1,2" }   # 1 frozen epoch, then 2 GLA-LoRA epochs
$SPE     = if ($env:SPE)     { $env:SPE }     else { "500" } # sampled images per epoch
$BATCH   = if ($env:BATCH)   { $env:BATCH }   else { "2" }
$ACCUM   = if ($env:ACCUM)   { $env:ACCUM }   else { "8" }   # effective batch 16
$NCROPS  = if ($env:NCROPS)  { $env:NCROPS }  else { "2" }   # matches the probe's crop budget
$MAXVAL  = if ($env:MAXVAL)  { $env:MAXVAL }  else { "400" }
$BUDGET  = if ($env:BUDGET)  { $env:BUDGET }  else { "50" }  # minutes per arm

$ENC = @()
if (Test-Path "outputs\lesion_pretrain\lesion_encoder.pt") {
    $ENC = @("--lesion-encoder", "outputs\lesion_pretrain\lesion_encoder.pt")
}

function Train-Arm {
    param([string]$Tag, [string[]]$InitFlags)
    Write-Host ""
    Write-Host "=============================================================="
    Write-Host " train  $Tag"
    Write-Host "=============================================================="
    Get-Date
    $logFile = "outputs\logs\${Tag}_train.log"
    $argsList = @("-u", "scripts/03_train.py", "--tag", $Tag, "--backbone", $BACKBONE) + $InitFlags + @(
        "--no-unfreeze", "--select", "ordinal",
        "--stage-epochs", $STAGES, "--samples-per-epoch", $SPE,
        "--batch-size", $BATCH, "--grad-accum", $ACCUM,
        "--global-size", "224", "--crop-input", "224", "--n-crops", $NCROPS,
        "--max-val", $MAXVAL, "--time-budget-min", $BUDGET
    ) + $ENC
    & $PY @argsList *> $logFile
    $rc = $LASTEXITCODE
    Write-Host "  exit $rc"
    Select-String -Path $logFile -Pattern '^\[A4\] (backbone|RETFound|allocated)|^\[stage|^\[s[0-9]|saved new best' |
        Select-Object -Last 20 | ForEach-Object { $_.Line }
    return $rc
}

function Eval-Arm {
    param([string]$Tag)
    Write-Host ""
    Write-Host "--- evaluate $Tag ---"
    Get-Date
    $logFile = "outputs\logs\${Tag}_eval.log"
    & $PY -u scripts/04_evaluate.py --ckpt "outputs/$Tag/best.pt" `
        --xai-samples 40 --corruption-samples 120 --mc-samples 4 *> $logFile
    Write-Host "  exit $LASTEXITCODE"
    Select-String -Path $logFile -Pattern 'accuracy|quadratic_weighted|auc_sight|auc_referable|f1_macro' |
        Select-Object -First 8 | ForEach-Object { $_.Line }
}

Train-Arm -Tag "obj3_retfound" -InitFlags @("--retfound-ckpt", $RETFOUND)
Train-Arm -Tag "obj3_imagenet" -InitFlags @("--retfound-ckpt", "none", "--allow-no-retfound", "--pretrained")

Eval-Arm -Tag "obj3_retfound"
Eval-Arm -Tag "obj3_imagenet"

Write-Host ""
Write-Host "OBJECTIVE 3 ADAPTATION ARM COMPLETE"
Get-Date
