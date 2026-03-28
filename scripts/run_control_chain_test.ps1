$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

& "$PSScriptRoot\activate_env.ps1"

$env:PYTHONPATH = "src"
python "scripts/test_control_chain.py" --num_envs 1 --steps 240 --warmup_steps 40 --headless
