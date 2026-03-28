$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Activate environment
& "$PSScriptRoot\activate_env.ps1"

python -m badminton_intercept.train.eval_policy --project-root $projectRoot
