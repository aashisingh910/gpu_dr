# _watch_for_stage4.ps1 - fires run_objective3.ps1 automatically once the main
# training run reaches stage 4, WITHOUT ever touching the main training
# process itself. This is a separate, independent script: it only reads
# data/training_monitor.csv (written by the unrelated monitor_loop.ps1) and
# nvidia-smi, and only launches a brand-new process when its condition is
# met - it never signals, restarts, or depends on the main python.exe
# training process in any way.
#
# Two gates, both required before launching:
#   1. The main run's logged stage has reached 4 (s4_full) - checked every
#      poll from training_monitor.csv, which the separate monitor_loop.ps1
#      task keeps refreshing every 5 min regardless of this script.
#   2. The GPU has enough free memory for objective 3's smaller arms
#      (batch 2, 2 crops, 224px vs the main run's batch 4, 6 crops, 448px)
#      to actually start without risking an OOM - checked via nvidia-smi
#      once gate 1 is met, re-checked every poll until there's room. This
#      never forces anything off the GPU; it only waits.
#
# One-shot: once it launches run_objective3.ps1, it logs that and exits -
# it does not re-trigger.
#
#   powershell -ExecutionPolicy Bypass -File _watch_for_stage4.ps1

Set-Location $PSScriptRoot
$log = "data\_objective3_watcher.log"
$monitorCsv = "data\training_monitor.csv"
$minFreeMiB = 2500   # safety margin for objective 3's smaller-batch arms

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')  $msg"
    Add-Content -Path $log -Value $line
}

Log "watcher started - waiting for stage 4 (current gate: stage>=4 AND >=${minFreeMiB}MiB free GPU memory)"

while ($true) {
    Start-Sleep -Seconds 120
    try {
        $lastLine = Get-Content $monitorCsv -Tail 1
        if (-not $lastLine) { continue }
        $fields = $lastLine -split ','
        $stage = [int]$fields[1]
    } catch {
        continue
    }

    if ($stage -lt 4) {
        continue
    }

    Log "stage $stage detected (>=4) - checking GPU headroom before launching"

    $freeMiB = $null
    try {
        $smi = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
        $freeMiB = [int]($smi.Trim())
    } catch {
        Log "nvidia-smi query failed, will re-check next poll"
        continue
    }

    if ($freeMiB -lt $minFreeMiB) {
        Log "stage 4 reached but only ${freeMiB}MiB free (need ${minFreeMiB}MiB) - waiting, will re-check in 2 min"
        continue
    }

    Log "stage 4 confirmed, ${freeMiB}MiB free - launching run_objective3.ps1 now"
    Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','run_objective3.ps1' `
        -RedirectStandardOutput 'data\_objective3_stdout.log' `
        -RedirectStandardError 'data\_objective3_stderr.log' `
        -WindowStyle Hidden
    Log "launched - watcher exiting (one-shot)"
    break
}
