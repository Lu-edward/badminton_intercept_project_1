# Badminton Intercept Project

基于 IsaacLab 的无人机羽毛球拦截强化学习训练系统。无人机携带球拍，在物理仿真环境中学习拦截羽毛球的轨迹。

## 项目结构

```
src/badminton_intercept/
├── envs/                  # 环境定义
│   ├── intercept_env.py       # 主环境类 (DirectRLEnv)
│   ├── intercept_env_cfg.py   # 环境配置数据类
│   ├── scene_builder.py        # 场景实体构建 (无人机/球/网)
│   └── launch_sampler.py      # 发球采样 / 轨迹库管理
├── mdp/                   # RL 核心组件
│   ├── observations.py        # Actor/Critic 观测构建
│   ├── rewards.py             # 奖励函数
│   ├── terminations.py        # 终止条件
│   └── curriculum.py          # 3阶段课程学习管理器
├── control/               # 无人机控制链
│   ├── ctbr_decoder.py        # Policy → CTBR 解码
│   ├── attitude_pd.py          # 姿态 PD 控制
│   └── rotor_mixer.py         # 电机混控
├── physics/               # 物理建模
│   └── shuttle_aero.py        # 羽毛球空气动力学与轨迹预测
├── randomization/         # 领域随机化
└── train/                 # 训练管道
    ├── train_ppo.py            # PPO 训练循环
    ├── serve_hover_runner.py   # Serve-Hover 任务训练器
    └── callbacks.py           # 训练回调
```

## 关键架构

### 控制链
```
Policy (4D) → CTBR Decode → Rate PID → Rotor Mixer → 4电机推力
```

### 课程学习 (Curriculum)
| 阶段 | 速度范围 | 描述 |
|------|----------|------|
| Stage 1 | vx 2-4 m/s | 慢速轨迹 |
| Stage 2 | vx 4-10 m/s | 中速全场 |
| Stage 3 | vx 8-14 m/s | 快速平射 |

晋升条件：连续 10 次迭代成功率 > 85%

### 轨迹库
预计算的发球轨迹存储于 `assets/trajectory_library/`:
- `serve_stage_1.pt` / `serve_stage_2.pt` / `serve_stage_3.pt`

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

## 快速开始

```powershell
# 训练
./scripts/run_train.ps1

# 评估预训练策略
python scripts/play_serve_hover_policy.py

# 可视化后点击球轨迹
python scripts/visualize_post_hit.py
```

## 核心模块说明

### 环境 (envs/)
- **InterceptEnv**: 主环境类，负责场景构建、物理注入、观测/奖励/终止计算
- **SceneBuilder**: 加载 `air.usd` 无人机模型，构建羽毛球、球网等实体
- **LaunchSampler**: 支持在线采样和轨迹库两种发球模式

### MDP (mdp/)
- **Observations**: 非对称 Actor/Critic 观测构建
- **Rewards**: 击球奖励 + 速度奖励 + 跟踪奖励 + 平滑惩罚
- **Terminations**: 成功拦截/出界/撞网等终止条件

### 控制 (control/)
- **CTBR Decoder**: 将策略输出映射为机体角速率目标
- **Attitude PD**: 姿态环 PID 控制
- **Rotor Mixer**: 将力和力矩分配到 4 个旋翼

### 物理 (physics/)
- **ShuttleAero**: 羽毛球阻力建模、轨迹预测、落点评估
