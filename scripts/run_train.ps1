$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Activate environment
& "$PSScriptRoot\activate_env.ps1"

$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"
$isaacLabBat = "F:\eai\isaaclab\IsaacLab\isaaclab.bat"

# More stable default for 12GB laptop GPUs.
& $isaacLabBat -p "$projectRoot\scripts\train_rsl_official.py" --headless --num_envs 1024 --max_iterations 10000
