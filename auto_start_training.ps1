# auto_start_training.ps1 - Task Scheduler entry point. Fires at user logon,
# waits briefly for GPU drivers/network to settle after boot, then launches
# the same resume_pipeline.ps1 used for manual runs, appending to the same
# log. Safe to run even if training already finished or isn't needed:
# resume_pipeline.ps1 itself detects current state (self-test, then checks
# preprocessing/lesion-pretrain/training progress) and does the right thing.
Set-Location $PSScriptRoot
Start-Sleep -Seconds 30
# Running under Task Scheduler with no attached console changes PowerShell's
# default output encoding vs. an interactive session, which corrupts *>>
# text redirection (nulls/spacing inserted between characters). Out-File
# with an explicit encoding sidesteps that regardless of console context.
powershell -ExecutionPolicy Bypass -File resume_pipeline.ps1 2>&1 | Out-File -FilePath data\_resume_pipeline_full.log -Append -Encoding utf8
