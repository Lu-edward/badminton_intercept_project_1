param(
    [int]$NumEnvs = 1,
    [int]$Steps = 1200,
    [double]$Speed = 0.5,
    [double]$FreqHz = 0.15,
    [double]$XCenter = 1.5,
    [double]$XAmp = 0.4,
    [double]$YAmp = 0.3,
    [double]$ZRef = 1.6,
    [double]$ZAmp = 0.05,
    [ValidateSet("rigid_object", "articulation")]
    [string]$DroneKind = "articulation",
    [ValidateSet("reference", "fixed_ctbr")]
    [string]$Mode = "reference",
    [double]$FixedWx = 0.0,
    [double]$FixedWy = 0.0,
    [double]$FixedWz = 0.0,
    [double]$FixedThrust = 9.81,
    [double]$FmGainScale = 1.0,
    [double]$MaxAngacc = 1.4,
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

Write-Host "[INFO] Running deterministic CTBR reference visualization"
Write-Host "[INFO] Num envs: $NumEnvs"
Write-Host "[INFO] Steps: $Steps"
Write-Host "[INFO] Speed: ${Speed}x"
Write-Host "[INFO] Freq: ${FreqHz} Hz"
Write-Host "[INFO] Reference center/amplitude: x=${XCenter}, x_amp=${XAmp}, y_amp=${YAmp}, z=${ZRef}"
Write-Host "[INFO] Drone kind: $DroneKind"
Write-Host "[INFO] Mode: $Mode"
if ($Mode -eq "fixed_ctbr") {
    Write-Host "[INFO] Fixed CTBR: wx=${FixedWx}, wy=${FixedWy}, wz=${FixedWz}, thrust=${FixedThrust}"
}
Write-Host "[INFO] FM gain scale: $FmGainScale, max_angacc: $MaxAngacc"
Write-Host "[INFO] Press Ctrl+C to stop."

$cmd = @(
    "-p", "$projectRoot\scripts\visualize_ctbr_reference.py",
    "--num_envs", "$NumEnvs",
    "--steps", "$Steps",
    "--speed", "$Speed",
    "--freq_hz", "$FreqHz",
    "--x_center", "$XCenter",
    "--x_amp", "$XAmp",
    "--y_amp", "$YAmp",
    "--z_ref", "$ZRef",
    "--z_amp", "$ZAmp",
    "--drone_kind", "$DroneKind",
    "--mode", "$Mode",
    "--fixed_wx", "$FixedWx",
    "--fixed_wy", "$FixedWy",
    "--fixed_wz", "$FixedWz",
    "--fixed_thrust", "$FixedThrust",
    "--fm_gain_scale", "$FmGainScale",
    "--max_angacc", "$MaxAngacc",
    "--print_every", "50"
)

if ($Headless) {
    $cmd += "--headless"
}

& $isaacLabBat @cmd
