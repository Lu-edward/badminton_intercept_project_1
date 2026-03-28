param(
    [string]$CheckpointPath = "",
    [int]$NumEnvs = 1,
    [double]$Speed = 0.2,
    [string]$LogRoot = "logs\\rsl_rl\\badminton_intercept_direct"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

& "$PSScriptRoot\\activate_env.ps1"

$env:PYTHONPATH = "$projectRoot\\src;$env:PYTHONPATH"
$isaacLabBat = "F:\\eai\\isaaclab\\IsaacLab\\isaaclab.bat"

if (-not (Test-Path $isaacLabBat)) {
    Write-Host "[ERROR] isaaclab.bat not found: $isaacLabBat"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
    $resolvedLogRoot = Join-Path $projectRoot $LogRoot
    if (-not (Test-Path $resolvedLogRoot)) {
        Write-Host "[ERROR] Log directory not found: $resolvedLogRoot"
        exit 1
    }

    $latestCheckpoint = Get-ChildItem -Path $resolvedLogRoot -Recurse -File -Filter "*.pt" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestCheckpoint) {
        Write-Host "[ERROR] No .pt checkpoint found under: $resolvedLogRoot"
        exit 1
    }

    $CheckpointPath = $latestCheckpoint.FullName
}

if (-not (Test-Path $CheckpointPath)) {
    Write-Host "[ERROR] Checkpoint not found: $CheckpointPath"
    exit 1
}

Write-Host "[INFO] Using checkpoint: $CheckpointPath"
Write-Host "[INFO] Num envs: $NumEnvs"
Write-Host "[INFO] Playback speed: ${Speed}x"
Write-Host "[INFO] Press Ctrl+C to stop."

& $isaacLabBat -p "$projectRoot\\scripts\\play_trained_policy.py" `
    --checkpoint "$CheckpointPath" `
    --num_envs $NumEnvs `
    --speed $Speed
