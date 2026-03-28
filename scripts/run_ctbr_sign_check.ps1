param(
    [int]$NumEnvs = 1,
    [double]$RateCmd = 0.35,
    [double]$ThrustCmd = 9.81,
    [int]$StepsPerPhase = 180,
    [int]$WarmupSteps = 40,
    [ValidateSet("rigid_object", "articulation")]
    [string]$DroneKind = "articulation",
    [double]$FmGainScale = 1.0,
    [double]$MaxAngacc = 1.0,
    [double]$AxisSignX = -1.0,
    [double]$AxisSignY = -1.0,
    [double]$AxisSignZ = 1.0,
    [double]$Speed = 0.0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

& "$PSScriptRoot\activate_env.ps1"

$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"
$isaacLabBat = "F:\eai\isaaclab\IsaacLab\isaaclab.bat"

if (-not (Test-Path $isaacLabBat)) {
    Write-Host "[ERROR] isaaclab.bat not found: $isaacLabBat"
    exit 1
}

Write-Host "[INFO] Running CTBR sign check"
Write-Host "[INFO] Num envs: $NumEnvs"
Write-Host "[INFO] Drone kind: $DroneKind"
Write-Host "[INFO] Rate cmd: $RateCmd rad/s | Thrust cmd: $ThrustCmd"
Write-Host "[INFO] Steps/phase: $StepsPerPhase | Warmup: $WarmupSteps"
Write-Host "[INFO] FM gain scale: $FmGainScale | Max angacc: $MaxAngacc"
Write-Host "[INFO] Axis sign: ($AxisSignX, $AxisSignY, $AxisSignZ)"

$cmd = @(
    "-p", "$projectRoot\scripts\check_ctbr_sign.py",
    "--num_envs", "$NumEnvs",
    "--rate_cmd", "$RateCmd",
    "--thrust_cmd", "$ThrustCmd",
    "--steps_per_phase", "$StepsPerPhase",
    "--warmup_steps", "$WarmupSteps",
    "--drone_kind", "$DroneKind",
    "--fm_gain_scale", "$FmGainScale",
    "--max_angacc", "$MaxAngacc",
    "--axis_sign_x", "$AxisSignX",
    "--axis_sign_y", "$AxisSignY",
    "--axis_sign_z", "$AxisSignZ",
    "--speed", "$Speed",
    "--print_every", "60"
)

if ($Headless) {
    $cmd += "--headless"
}

& $isaacLabBat @cmd
