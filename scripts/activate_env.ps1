# Common environment activation script
# This script handles environment switching for all run scripts

$envPath = "F:\eai\isaaclab\env_isaaclab"
$activateScript = "$envPath\Scripts\Activate.ps1"

# Check if we're in a conda environment and deactivate it
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "[INFO] Deactivating current conda environment: $env:CONDA_DEFAULT_ENV"
    conda deactivate
}

# Check if env_isaaclab is already activated
if ($env:VIRTUAL_ENV -eq $envPath) {
    Write-Host "[INFO] env_isaaclab is already activated"
} else {
    # Activate env_isaaclab virtual environment
    Write-Host "[INFO] Activating env_isaaclab virtual environment"
    & $activateScript
}

# Warp kernel cache: force a writable project-local path to avoid permission issues on default cache.
$projectRoot = Split-Path -Parent $PSScriptRoot
$warpCache = Join-Path $projectRoot ".warp_cache"
if (-not (Test-Path $warpCache)) {
    New-Item -ItemType Directory -Path $warpCache | Out-Null
}
$env:WARP_CACHE_PATH = $warpCache
Write-Host "[INFO] WARP_CACHE_PATH=$env:WARP_CACHE_PATH"
