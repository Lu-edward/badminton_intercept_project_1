# Badminton Intercept Project

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的无人机羽毛球拦截强化学习项目。无人机(X152b 四旋翼)携带球拍,在物理仿真环境中学习拦截高速飞行的羽毛球。

## 项目亮点

- **端到端 RL 控制链**: Policy → CTBR 解码 → 姿态 PD → 电机混控,直接输出 4 电机推力
- **羽毛球空气动力学建模**: 阻力、轨迹预测与落点评估
- **三阶段课程学习**: 从慢速轨迹逐步晋升到快速平射
- **大规模并行训练**: 基于 Isaac Lab + RSL-RL,默认 4096 个并行环境
- **离线轨迹库**: 预计算发球轨迹,加速训练采样
- **wandb 训练监控**: 完整的 episode 指标记录

## 项目结构

```
src/badminton_intercept/
├── envs/                  # 环境定义
│   ├── intercept_env.py       # 主环境类 (DirectRLEnv)
│   ├── intercept_env_cfg.py   # 环境配置数据类
│   ├── scene_builder.py       # 场景实体构建 (无人机/球/网)
│   └── launch_sampler.py      # 发球采样 / 轨迹库管理
├── mdp/                   # RL 核心组件
│   ├── observations.py        # Actor/Critic 观测构建
│   ├── rewards.py             # 奖励函数
│   ├── terminations.py        # 终止条件
│   └── curriculum.py          # 3 阶段课程学习管理器
├── control/               # 无人机控制链
│   ├── ctbr_decoder.py        # Policy → CTBR 解码
│   ├── attitude_pd.py         # 姿态 PD 控制
│   └── rotor_mixer.py         # 电机混控
├── physics/               # 物理建模
│   └── shuttle_aero.py        # 羽毛球空气动力学与轨迹预测
├── randomization/         # 领域随机化
└── train/                 # 训练管道
    ├── train_ppo.py           # PPO 训练循环
    ├── serve_hover_runner.py  # Serve-Hover 任务训练器
    └── callbacks.py           # 训练回调
```

## 环境依赖

- Python >= 3.10
- NVIDIA Isaac Sim / Isaac Lab(请参考 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- PyTorch (CUDA)
- RSL-RL、wandb 等

安装本项目包(在 Isaac Lab 环境中):

```bash
pip install -e .
```

## 使用方法

### 1. 构建发球轨迹库(可选,训练可直接在线采样)

```bash
python scripts/build_shuttle_trajectory_library.py
```

生成的轨迹库存放于 `assets/trajectory_library/serve_stage_{1,2,3}.pt`。

### 2. 训练

```bash
python scripts/train_rsl_official.py \
    --task Isaac-Badminton-Intercept-Direct-v0 \
    --num_envs 4096 \
    --max_iterations 10000 \
    --wandb --wandb_project badminton_intercept
```

常用参数:

| 参数 | 说明 |
|------|------|
| `--num_envs` | 并行环境数量(默认 4096) |
| `--resume --load_run <run_dir>` | 从指定 run 断点续训 |
| `--init_from_checkpoint <ckpt>` | 加载权重微调(不加载优化器状态) |
| `--curriculum_stage_id {1,2,3}` | 指定起始课程阶段 |
| `--freeze_curriculum_stage` | 固定在当前阶段,不自动晋升 |
| `--enable_serve_hover` | 启用 Serve-Hover 两阶段训练 |
| `--enable_post_hit_tracking` | 启用击球后跟踪模式(任务 2) |
| `--headless` | 无渲染运行(训练推荐) |

Windows 下也可直接使用 `scripts/run_train.ps1` 一键启动。

### 3. 可视化 / 回放策略

```bash
python scripts/play_trained_policy.py --checkpoint <path/to/model.pt>
```

支持慢速回放(`--speed`)、多 episode 录视频(`--num_episodes`),输出保存在 `videos/`。

### 4. 其他工具脚本

| 脚本 | 用途 |
|------|------|
| `scripts/visualize_post_hit.py` | 可视化击球后的球拍/羽毛球轨迹 |
| `scripts/eval_checkpoint_on_stage1_trajectories.py` | 在 Stage 1 轨迹上评估 checkpoint |
| `scripts/verify_trajectory.py` / `verify_library.py` | 校验轨迹与轨迹库 |
| `scripts/pretrain_serve_hover_bc.py` | Serve-Hover 行为克隆预训练 |

## 关键设计

### 控制链

```
Policy (4D) → CTBR Decode → Rate PID → Rotor Mixer → 4 电机推力
```

### 课程学习 (Curriculum)

| 阶段 | 速度范围 | 描述 |
|------|----------|------|
| Stage 1 | vx 2–4 m/s | 慢速轨迹 |
| Stage 2 | vx 4–10 m/s | 中速全场 |
| Stage 3 | vx 8–14 m/s | 快速平射 |

晋升条件:连续 10 次迭代成功率 > 85%。

## 配置文件

| 文件 | 用途 |
|------|------|
| `configs/env/task.yaml` | 任务配置 |
| `configs/env/drone.yaml` | 无人机物理参数 |
| `configs/env/shuttlecock.yaml` | 羽毛球物理参数 |
| `configs/env/racket.yaml` | 球拍碰撞参数 |
| `configs/env/randomization.yaml` | 领域随机化范围 |
| `configs/train/ppo.yaml` | PPO 超参数 |
| `configs/train/network.yaml` | Actor-Critic 网络结构 |

## 测试

```bash
python -m pytest tests/
```

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
