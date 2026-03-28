$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Activate environment
& "$PSScriptRoot\activate_env.ps1"

python -m badminton_intercept.train.train_ppo --project-root $projectRoot --smoke-test
