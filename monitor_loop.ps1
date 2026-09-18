# monitor_loop.ps1 - runs monitor_snapshot.py every 5 minutes, forever, as its
# own independent background process. This does NOT depend on the assistant
# being "awake" or on the conversation being idle - unlike a scheduled
# wakeup (which a new user message can supersede before it fires), this is a
# real OS process that keeps ticking regardless of what else is happening.
#
# Stop it with: Get-Process powershell | Where-Object { $_.CommandLine -match
# 'monitor_loop' } | Stop-Process   (or just let it run - it's cheap)

Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"

while ($true) {
    & $py monitor_snapshot.py
    Start-Sleep -Seconds 300
}
