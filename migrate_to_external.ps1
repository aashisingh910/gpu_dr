# migrate_to_external.ps1 - copies exactly the four folders needed to resume
# training on another machine (checkpoint + data cache + lesion data), onto
# an external drive or any other destination path.
#
# Does NOT touch the live training process - this only reads from the
# source folders. checkpoint.pt is safe to copy even while training is
# running: scripts/03_train.py writes it to a .tmp file and atomically
# renames it into place, so a reader (robocopy) only ever sees either the
# old complete file or the new complete file, never a half-written one.
#
# Uses robocopy (not Copy-Item) because:
#   - /Z makes it restartable - if it's interrupted partway through 108GB,
#     rerunning this script resumes instead of starting the copy over.
#   - /MT:8 copies several files in parallel, which matters a lot here
#     since data\cache holds a very large number of small-to-medium files.
#   - /NFL /NDL /NP keep the console from being flooded with a line per
#     file across ~100k+ images.
#
#   powershell -ExecutionPolicy Bypass -File migrate_to_external.ps1 -Destination "E:\dr_training_transfer"

param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$folders = @(
    "data\cache",
    "data\raw\lesion",
    "outputs\retfound_plus_laft_xai",
    "outputs\lesion_pretrain"
)

if (-not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}

Write-Host "Checking destination free space..." -ForegroundColor Cyan
$destDrive = (Resolve-Path $Destination).Path.Substring(0, 2)
$vol = Get-Volume -DriveLetter $destDrive.TrimEnd(":") -ErrorAction SilentlyContinue
if ($vol) {
    $freeGB = [math]::Round($vol.SizeRemaining / 1GB, 1)
    Write-Host "  $destDrive has $freeGB GB free"
    if ($freeGB -lt 140) {
        Write-Host "  WARNING: these four folders total roughly 135-140 GB - this drive may not have enough room." -ForegroundColor Yellow
    }
}

$overallStart = Get-Date

foreach ($rel in $folders) {
    $src = Join-Path $PSScriptRoot $rel
    $dst = Join-Path $Destination $rel

    if (-not (Test-Path $src)) {
        Write-Host "[skip] $rel - not found at $src" -ForegroundColor Yellow
        continue
    }

    New-Item -ItemType Directory -Path $dst -Force | Out-Null

    Write-Host ""
    Write-Host "==== Copying $rel ====" -ForegroundColor Cyan
    $start = Get-Date

    # /E all subdirs incl. empty, /Z restartable, /MT:8 parallel,
    # /R:3 /W:5 retry 3x with 5s wait on a locked/in-use file,
    # /NFL /NDL /NP /NJH /NJS keep output to one line per folder.
    robocopy $src $dst /E /Z /MT:8 /R:3 /W:5 /NFL /NDL /NP /NJH /NJS

    $elapsed = (Get-Date) - $start
    Write-Host ("  done in {0:N1} min" -f $elapsed.TotalMinutes) -ForegroundColor Green
}

$overallElapsed = (Get-Date) - $overallStart
Write-Host ""
Write-Host "==== MIGRATION COPY COMPLETE ====" -ForegroundColor Green
Write-Host ("Total time: {0:N1} min" -f $overallElapsed.TotalMinutes)
Write-Host "Verify sizes match before disconnecting the drive:"
foreach ($rel in $folders) {
    $src = Join-Path $PSScriptRoot $rel
    $dst = Join-Path $Destination $rel
    if ((Test-Path $src) -and (Test-Path $dst)) {
        $srcSize = (Get-ChildItem $src -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        $dstSize = (Get-ChildItem $dst -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        $match = if ($srcSize -eq $dstSize) { "OK" } else { "MISMATCH - rerun this script to resume/fix" }
        Write-Host ("  {0}: source {1:N1} GB / copied {2:N1} GB - {3}" -f $rel, ($srcSize/1GB), ($dstSize/1GB), $match)
    }
}
