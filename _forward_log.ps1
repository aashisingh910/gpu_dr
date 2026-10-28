Set-Location "C:\Users\HP\Downloads\dr-project-python-fixed\dr-project-python-master"
$src = 'data\_manual_resume_stdout.log'
$dst = 'data\_resume_pipeline_full.log'

# Polls and appends in short open/read/close bursts instead of holding $dst
# open with -Wait - a persistent handle there previously blocked the
# separate monitor_loop.ps1 task (running every 5 min) from reading it.
$forwarded = (Get-Content $src).Count

while ($true) {
    Start-Sleep -Seconds 45
    $lines = Get-Content $src
    if ($lines.Count -gt $forwarded) {
        $new = $lines | Select-Object -Skip $forwarded
        Add-Content -Path $dst -Value $new
        $forwarded = $lines.Count
    }
}
