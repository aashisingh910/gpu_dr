# resume_pipeline.ps1
#
# SAFE TO RE-RUN AT ANY TIME - after a reboot, a crash, closing the terminal,
# or just stopping it yourself. Each stage below checkpoints its own progress
# to disk and auto-detects that checkpoint on startup, so re-running this
# exact script always continues from the last completed step/epoch instead
# of starting over. You never need to track progress yourself; just run:
#
#   powershell -ExecutionPolicy Bypass -File resume_pipeline.ps1
#
# IMPORTANT: once a stage (especially "Staged training") has produced its
# first checkpoint, do not edit the arguments below - a resumed run reuses
# the optimizer/learning-rate-schedule state saved under the ORIGINAL
# arguments (particularly --batch-size/--grad-accum), so changing them
# mid-run throws off the schedule rather than cleanly resuming. If a stage
# runs out of GPU memory before its first checkpoint, it's safe to lower
# --batch-size (and raise --grad-accum to compensate) and rerun - nothing
# has been checkpointed yet at that point.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Duplicate-process guard - added 2026-10-30 once both the startup and
# logon triggers fire the same Task Scheduler action, which can now race
# each other (IgnoreNew on the TASK only stops a second copy of the same
# scheduled task; it does nothing for a second, independently-launched
# python.exe). Matches on 03_train.py AND this exact project path, not
# just "python.exe", so an unrelated Python process elsewhere on the
# machine is never mistaken for a second training run. Only checks and
# exits early - never touches an existing process.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match [regex]::Escape($PSScriptRoot) -and $_.CommandLine -match '03_train\.py' }
if ($existing) {
    Write-Host "[guard] a 03_train.py process for this project is already running " `
        "(PID $($existing.ProcessId -join ', ')) - exiting without starting a second copy." `
        -ForegroundColor Yellow
    exit 0
}

$env:PYTHONPATH = "src"
# Reduces CUDA allocator fragmentation - stage 4 (99.7% of the backbone
# trainable) runs this 8GB card right at its memory ceiling and has been
# crashing intermittently around step 200-300 of its first epoch. This
# doesn't touch --batch-size/--grad-accum or anything checkpointed, so it's
# safe to add on any resume, automatic or manual.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = ".\.venv\Scripts\python.exe"

function Step {
    # NOTE: the second parameter must NOT be named $Args - that collides with
    # PowerShell's automatic $args variable and silently breaks the splat
    # below, which invokes python with NO arguments at all: it drops into an
    # interactive REPL instead of running the intended script, and Python
    # 3.14's REPL then spins forever trying (and failing) to read a real
    # console from what is actually a non-interactive/background shell. This
    # bit exactly that way once already - do not reintroduce it.
    param([string]$Name, [string[]]$PyArgs)
    Write-Host ""
    Write-Host "==== $Name ====" -ForegroundColor Cyan
    # belt-and-braces: refuse to invoke python with zero arguments under any
    # circumstance, since that drops into an interactive REPL rather than
    # failing loudly - exactly the failure this function exists to prevent.
    if (-not $PyArgs -or $PyArgs.Count -eq 0) {
        Write-Host "[$Name] internal error: no arguments to pass to python - refusing to run it interactively." -ForegroundColor Red
        exit 1
    }
    & $py @PyArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[$Name] exited with code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "If this was an interruption (shutdown/Ctrl-C/OOM), just re-run" -ForegroundColor Yellow
        Write-Host "this script again - completed work is checkpointed and will" -ForegroundColor Yellow
        Write-Host "not be redone." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
}

Step "Self-test" @("-u", "scripts/08_selftest.py")

Step "Preprocessing (full 35,126 images @ 1024px)" @(
    "-u", "scripts/02_preprocess.py",
    "--root", "data/raw/eyepacs-hi",
    "--image-size", "1024", "--n-crops", "6", "--crop-size", "320",
    "--workers", "28"
)

Step "Lesion-expert pretraining (real IDRiD/DDR masks)" @(
    "-u", "scripts/10_lesion_pretrain.py",
    "--image-size", "448"
)

# batch-size 2 / grad-accum 16 (effective batch 32, unchanged) + --grad-
# checkpoint. Stages 1-3 ran at batch-size 4/grad-accum 8 and are fully
# complete - this only takes effect for stage 4, which has no checkpoint
# of its own batch-size/grad-accum choice to desync (resume_pipeline.ps1
# skips any stage already past, si < resume_stage, before ever touching
# this value). Lowered from 4/8 on 2026-10-30: stage 4 (99.7% of the
# backbone trainable, vs. 36% in stage 3) runs this 8GB RTX 5060 right at
# its memory ceiling - it hit a hard crash, a GPU-driver TDR, and a full
# hang, all within ~2 days, always around step 200-300 of its first
# epoch (the same deterministic sample order hitting the same memory-
# hungry batch every retry). This halves peak activation memory per
# micro-batch while keeping the effective batch and gradient quality
# identical.
Step "Staged training (full 5/12/14/14 schedule)" @(
    "-u", "scripts/03_train.py",
    "--stage-epochs", "5,12,14,14", "--patience", "45",
    "--samples-per-epoch", "6000", "--max-val", "1500",
    "--batch-size", "2", "--grad-accum", "16", "--grad-checkpoint",
    "--global-size", "448", "--crop-input", "224", "--n-crops", "6",
    # cfg.train.num_workers defaults to 0 (single-threaded, in-process data
    # loading) - confirmed via nvidia-smi that this starves the GPU (0%, 0%,
    # 99%, 0%, 0% utilization pattern: one quick compute burst per batch,
    # then idle while the next batch is prepared on a single CPU core).
    # 16 parallel loader workers keeps several batches prefetched so the GPU
    # stays fed instead of waiting on CPU-bound memmap reads/resizing.
    "--workers", "16",
    # the default config's RETFound checkpoint is a local path that doesn't
    # exist here, and its HF fallback (YukunZhou/RETFound_mae_natureCFP) is
    # gated (no access). bitfount/RETFound_MAE is a public, ungated mirror
    # of the same real RETFound MAE weights - confirmed public via the HF
    # API and the one this project's own earlier pilot run used successfully.
    "--retfound-ckpt", "bitfount/RETFound_MAE:pytorch_model.bin",
    "--lesion-encoder", "outputs/lesion_pretrain/lesion_encoder.pt",
    "--amp", "--ema-decay", "0.995", "--xai-weight", "0.05",
    "--aux-lesion-masks", "--aux-lesion-root", "data/raw/lesion"
)

Step "Full evaluation" @(
    "-u", "scripts/04_evaluate.py",
    "--external-cache", "data/cache_external", "--tta"
)

Step "Screening triage" @(
    "-u", "scripts/09_screening.py", "--external-cache", "data/cache_external"
)

Step "Objective 1 report" @("-u", "scripts/15_objective1_report.py")
Step "Lesion visibility (CLAHE)" @("-u", "scripts/12_lesion_visibility.py", "--algo", "clahe")
Step "Objective 2 report" @("-u", "scripts/14_objective2_report.py")
Step "Linear probe" @("-u", "scripts/13_linear_probe.py", "--backbone", "vit_large_patch16_224")
Step "Ordinal calibration" @("-u", "scripts/16_ordinal_calibration.py")
Step "Calibration report" @("-u", "scripts/17_calibration_report.py")
Step "Objective 3 report" @("-u", "scripts/18_objective3_report.py")
Step "Objective 3 validation" @("-u", "scripts/19_objective3_validation.py")
Step "Objective 3 complete" @("-u", "scripts/20_objective3_complete.py")
Step "XAI evaluation (real IDRiD/DDR masks)" @(
    "-u", "scripts/21_xai_evaluation.py", "--ref-masks", "idrid_ddr",
    "--counterfactual", "--lesion-root", "data/raw/lesion"
)
Step "Objective 4 report" @("-u", "scripts/22_objective4_report.py")
Step "Consolidated objectives summary" @("-u", "scripts/23_objectives_summary.py")
Step "Methods reference" @("-u", "scripts/24_methods_reference.py")
Step "Full model comparison" @("-u", "scripts/07_compare_models.py", "--epochs", "3")
Step "Deployment optimization" @("-u", "scripts/06_deploy.py", "--epochs", "5")

Write-Host ""
Write-Host "==== PIPELINE COMPLETE ====" -ForegroundColor Green
Write-Host "Results: outputs/objectives_consolidated.json and outputs/*/evaluation.json"
