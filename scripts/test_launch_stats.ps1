param(
    [int]$NumEnvs = 512,
    [int]$TestEpisodes = 100
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

Write-Host "======================================"
Write-Host "   Launch 统计测试 (直接环境测试)"
Write-Host "======================================"
Write-Host "[INFO] Num envs: $NumEnvs"
Write-Host "[INFO] Test episodes: $TestEpisodes"
Write-Host ""

$scriptContent = @"
import os
import sys
import torch
import tempfile
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
args, hydra_args = parser.parse_known_args()

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.train.register_task import TASK_ID, register_task

register_task()

env_cfg = InterceptEnvCfg()
env_cfg.scene.num_envs = $NumEnvs
env_cfg.seed = 42
env_cfg.randomization_enabled = False
env_cfg.uav_init_lin_vel_range = (0.0, 0.0)

task_id = 'Isaac-Badminton-Intercept-Direct-v0'
env = gym.make(task_id, cfg=env_cfg)

num_envs = env.unwrapped.num_envs
device = env.unwrapped.device

print(f'[INFO] Num envs: {num_envs}, Device: {device}')

print('[INFO] Starting environment reset test for launch statistics...')

launch_stats_history = []

for episode in range($TestEpisodes):
    obs, _ = env.reset()

    for _ in range(5):
        actions = torch.zeros(num_envs, 4, device=device)
        results = env.step(actions)
        if len(results) == 5:
            obs, rewards, terminated, truncated, infos = results
            dones = terminated | truncated
        else:
            obs, rewards, dones, infos = results

        if isinstance(infos, dict) and 'launch' in infos:
            launch_stats_history.append(infos['launch'])
            print(f'[Episode {episode + 1}] launch: {infos["launch"]}')
            break

env.close()

print('')
print('======================================')
print('   Launch 统计结果')
print('======================================')

if launch_stats_history:
    import numpy as np

    valid_rates = [s['launch_valid_rate'] for s in launch_stats_history]
    resample_rounds = [s['launch_resample_rounds_mean'] for s in launch_stats_history]
    fallback_counts = [s['launch_fallback_count'] for s in launch_stats_history]
    no_cross_counts = [s['launch_no_cross_count'] for s in launch_stats_history]
    oob_counts = [s['launch_oob_count'] for s in launch_stats_history]

    print(f'launch_valid_rate:       mean={np.mean(valid_rates):.4f}, min={np.min(valid_rates):.4f}, max={np.max(valid_rates):.4f}')
    print(f'launch_resample_rounds:  mean={np.mean(resample_rounds):.4f}, min={np.min(resample_rounds):.4f}, max={np.max(resample_rounds):.4f}')
    print(f'launch_fallback_count:   total={np.sum(fallback_counts)}, mean={np.mean(fallback_counts):.4f}')
    print(f'launch_no_cross_count:   total={np.sum(no_cross_counts)}, mean={np.mean(no_cross_counts):.4f}')
    print(f'launch_oob_count:        total={np.sum(oob_counts)}, mean={np.mean(oob_counts):.4f}')
else:
    print('[WARNING] No launch statistics found in extras')

print('')
print('[INFO] Test completed')

simulation_app.close()
"@

$tempScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'
Set-Content -Path $tempScript -Value $scriptContent -Encoding UTF8

try {
    & $isaacLabBat -p $tempScript --headless
} finally {
    if (Test-Path $tempScript) {
        Remove-Item $tempScript -Force
    }
}
