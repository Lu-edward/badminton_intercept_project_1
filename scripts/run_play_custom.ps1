param(
    [Parameter(Mandatory=$true, HelpMessage="Path to the checkpoint file (.pt)")]
    [string]$CheckpointPath,
    
    [Parameter(Mandatory=$false, HelpMessage="Number of environments to visualize")]
    [int]$NumEnvs = 1,
    
    [Parameter(Mandatory=$false, HelpMessage="Playback speed multiplier (0.2 = 5x slower)")]
    [double]$Speed = 0.2
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Activate environment
& "$PSScriptRoot\activate_env.ps1"

$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"
$isaacLabBat = "F:\eai\isaaclab\IsaacLab\isaaclab.bat"

# Check if checkpoint exists
if (-not (Test-Path $CheckpointPath)) {
    Write-Host "[ERROR] Checkpoint not found: $CheckpointPath"
    exit 1
}

Write-Host "[INFO] Using checkpoint: $CheckpointPath"
Write-Host "[INFO] Number of environments: $NumEnvs"
Write-Host "[INFO] Playback speed: ${Speed}x"
Write-Host "[INFO] Starting visualization..."
Write-Host "[INFO] Press Ctrl+C in the terminal to stop"

& $isaacLabBat -p "$projectRoot\scripts\play_trained_policy.py" --checkpoint "$CheckpointPath" --num_envs $NumEnvs --speed $Speed
