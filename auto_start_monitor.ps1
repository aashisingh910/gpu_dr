# auto_start_monitor.ps1 - Task Scheduler entry point. Fires at user logon,
# waits briefly, then launches the same monitor_loop.ps1 used for manual
# runs, so data/training_monitor.csv keeps getting a fresh row every 5
# minutes without anyone having to start it by hand.
Set-Location $PSScriptRoot
Start-Sleep -Seconds 45
powershell -ExecutionPolicy Bypass -File monitor_loop.ps1
