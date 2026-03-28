$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Activate environment
& "$PSScriptRoot\activate_env.ps1"

$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"
$isaacLabBat = "F:\eai\isaaclab\IsaacLab\isaaclab.bat"

# Find the latest checkpoint
$logRoot = "logs\rsl_rl\badminton_intercept_direct"
if (-not (Test-Path $logRoot)) {
    Write-Host "[ERROR] Log directory not found: $logRoot"
    Write-Host "[INFO] Please check if training has been completed and logs exist"
    exit 1
}

# Get the latest run directory
$latestRun = Get-ChildItem -Path $logRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1

if (-not $latestRun) {
    Write-Host "[ERROR] No training runs found in $logRoot"
    exit 1
}

Write-Host "[INFO] Using run: $($latestRun.Name)"

# Find the latest checkpoint in the run
$checkpointPattern = Join-Path $latestRun.FullName "model_*.pt"
$latestCheckpoint = Get-ChildItem -Path $latestRun.FullName -Filter "model_*.pt" | Sort-Object Name -Descending | Select-Object -First 1

if (-not $latestCheckpoint) {
    Write-Host "[ERROR] No checkpoint found in $($latestRun.FullName)"
    Write-Host "[INFO] Looking for files matching: model_*.pt"
    exit 1
}

Write-Host "[INFO] Using checkpoint: $($latestCheckpoint.Name)"
Write-Host "[INFO] Starting visualization with 1 environment at 0.2x speed..."
Write-Host "[INFO] Press Ctrl+C in the terminal to stop"

& $isaacLabBat -p "$projectRoot\scripts\play_trained_policy.py" --checkpoint "$($latestCheckpoint.FullName)" --num_envs 1 --speed 0.2
