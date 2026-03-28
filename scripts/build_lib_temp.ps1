# Build trajectory library script
$ErrorActionPreference = "Stop"

$projectRoot = "F:\eai\isaaclab\badminton_intercept_project_1"
Set-Location $projectRoot

# Activate conda environment
$envPath = "F:\eai\isaaclab\env_isaaclab"
$activateScript = "$envPath\Scripts\Activate.ps1"
if ($env:VIRTUAL_ENV -ne $envPath) {
    & $activateScript
}

# Set PYTHONPATH
$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"

# Build trajectory library with 500000 samples per stage
python scripts/build_shuttle_trajectory_library.py --stage-counts 500000 500000 500000 --output-dir "assets/trajectory_library"
