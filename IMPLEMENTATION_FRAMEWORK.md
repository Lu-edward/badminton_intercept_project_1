# IsaacLab 无人机-人羽毛球对打：代码框架设计（骨架版）

> 目标：先完成**可扩展的工程骨架**，不填充完整算法细节；后续按模块逐步实现与联调。

## 1. 项目定位与边界

- 仿真平台：IsaacLab
- 任务形态：单无人机对“人类来球”进行空间拦截（以程序化发球代理人类动作）
- 资产输入：`air.usd`（已包含球拍结构，位于 `transfer` 资产目录）
- 当前阶段：只搭建代码框架、配置结构、接口约定

## 2. 建议目录结构（单文件夹项目）

```text
badminton_intercept_project/
├─ README.md
├─ IMPLEMENTATION_FRAMEWORK.md            # 当前文档
├─ pyproject.toml                         # 可选：本地包化
├─ configs/
│  ├─ env/
│  │  ├─ task.yaml                        # 环境总配置（场地、终止条件、随机化开关）
│  │  ├─ drone.yaml                       # 无人机物理参数 + 控制限幅
│  │  ├─ shuttlecock.yaml                 # 羽毛球代理参数 + 空气动力学参数
│  │  ├─ racket.yaml                      # 甜区/边缘碰撞参数
│  │  └─ randomization.yaml               # 域随机化范围
│  └─ train/
│     ├─ ppo.yaml                         # PPO/GAE 超参
│     └─ network.yaml                     # Actor-Critic 网络结构
├─ assets/
│  └─ refs.md                             # 资产路径登记（含 air.usd 实际路径）
├─ src/
│  └─ badminton_intercept/
│     ├─ __init__.py
│     ├─ envs/
│     │  ├─ intercept_env_cfg.py          # IsaacLab EnvCfg 定义
│     │  ├─ intercept_env.py              # 主环境类：reset/step/obs/reward/done
│     │  └─ scene_builder.py              # 场景搭建（场地、网、无人机、球）
│     ├─ physics/
│     │  ├─ sysid_params.py               # 质量、惯量、推力系数等系统辨识参数
│     │  ├─ shuttle_aero.py               # 羽毛球空气动力学外力注入接口
│     │  ├─ collision_model.py            # 甜区/边缘弹性模型接口
│     │  └─ surrogate_shapes.py           # 非凸到代理几何体映射
│     ├─ control/
│     │  ├─ ctbr_decoder.py               # 策略动作 -> CTBR 解码
│     │  ├─ attitude_pd.py                # 姿态角速度 PD 控制
│     │  └─ rotor_mixer.py                # 扭矩/总推力 -> 四旋翼分配
│     ├─ mdp/
│     │  ├─ observations.py               # Actor/Critic 非对称观测构建
│     │  ├─ rewards.py                    # 稀疏+密集+平滑奖励
│     │  ├─ terminations.py               # 成功/失败/越界/超时判定
│     │  └─ curriculum.py                 # 可选：课程学习难度调度
│     ├─ randomization/
│     │  ├─ reset_manager.py              # 并行子环境 reset 与状态写回
│     │  ├─ launch_sampler.py             # 发球初速度/仰角采样
│     │  └─ domain_rand.py                # 质量/恢复系数/阻力系数随机化
│     ├─ train/
│     │  ├─ train_ppo.py                  # 训练入口
│     │  ├─ eval_policy.py                # 评估入口
│     │  └─ callbacks.py                  # 日志、checkpoint、可视化钩子
│     └─ utils/
│        ├─ tensors.py                    # 张量工具（批量索引、掩码写回）
│        ├─ metrics.py                    # 命中率、甜区率、能耗指标
│        └─ logging.py                    # 统一日志接口
└─ scripts/
   ├─ run_train.ps1
   ├─ run_eval.ps1
   └─ smoke_test.ps1
```

## 3. 六大理论模块到代码模块映射

### 3.1 系统参数化与物理标定层

- `physics/sysid_params.py`
  - 职责：维护无人机真实质量、惯量矩阵、电机推力系数等标定参数
  - 输出：供控制器与动力学注入模块调用的只读参数对象
- `configs/env/drone.yaml`、`configs/env/shuttlecock.yaml`
  - 职责：把标定值参数化，支持不同机型/球型切换

### 3.2 场景抽象与碰撞代理建模

- `envs/scene_builder.py`
  - 职责：场地、球网、无人机、羽毛球代理体实例化
- `physics/surrogate_shapes.py`
  - 职责：羽毛球非凸网格 -> 刚性球代理
- `physics/collision_model.py`
  - 职责：球拍甜区/边缘碰撞恢复系数切换接口

### 3.3 并行初始化与域随机化

- `randomization/reset_manager.py`
  - 职责：筛选终止子环境，批量 reset 并写回状态
- `randomization/launch_sampler.py`
  - 职责：左半场发球状态（速度、仰角、方向）采样
- `randomization/domain_rand.py`
  - 职责：每回合对质量、弹性、空气动力学参数采样并覆盖

### 3.4 动力学步进与空气动力注入

- `physics/shuttle_aero.py`
  - 职责：按速度平方计算阻力矢量，逐步注入外力
- `control/ctbr_decoder.py` + `control/attitude_pd.py` + `control/rotor_mixer.py`
  - 职责：策略动作 -> CTBR -> 角速度跟踪 -> 四旋翼命令

### 3.5 MDP 张量化

- `mdp/observations.py`
  - 职责：Actor 局部噪声观测、Critic 特权观测（非对称）
- `mdp/rewards.py`
  - 职责：命中奖励、甜区奖励、追踪奖励、动作平滑惩罚
- `mdp/terminations.py`
  - 职责：接球成功/失败、越界、时间终止等条件

### 3.6 策略优化管线

- `train/train_ppo.py`
  - 职责：rollout 收集、GAE 计算、PPO clipped 更新
- `configs/train/ppo.yaml`
  - 职责：学习率、clip ratio、entropy coef、epoch/batch 配置

## 4. 环境类建议接口（先定义签名）

文件：`src/badminton_intercept/envs/intercept_env.py`

```python
class InterceptEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None):
        ...

    def _setup_scene(self):
        """构建场景与实体句柄（无人机、羽毛球代理体、网等）"""
        ...

    def _pre_physics_step(self, actions):
        """动作解码 + 控制命令准备（CTBR -> 电机指令）"""
        ...

    def _apply_action(self):
        """把电机推力/力矩写入仿真"""
        ...

    def _apply_external_forces(self):
        """羽毛球空气动力学外力注入"""
        ...

    def _get_observations(self):
        """返回 actor_obs / critic_obs"""
        ...

    def _get_rewards(self):
        """并行奖励计算"""
        ...

    def _get_dones(self):
        """终止与截断条件"""
        ...

    def _reset_idx(self, env_ids):
        """并行子环境重置 + 域随机化"""
        ...
```

## 5. 核心数据流（Step 循环）

1. 读策略输出动作 `a_t`
2. `ctbr_decoder` 解码动作并经 PD + mixer 得到旋翼命令
3. 注入空气动力学外力（羽毛球代理体）
4. 物理引擎步进
5. 读取状态并组装 `actor_obs` / `critic_obs`
6. 并行计算 reward 与 done
7. 对 done 子环境调用 `_reset_idx`

## 6. 配置优先级（建议）

- 一级：`configs/env/*.yaml`（物理与任务定义）
- 二级：`configs/train/*.yaml`（训练策略定义）
- 三级：命令行覆盖（用于 sweep 与 ablation）

示例：

```bash
python -m badminton_intercept.train.train_ppo \
  env.task=intercept \
  env.randomization.enable=true \
  train.ppo.lr=3e-4
```

## 7. 第一阶段只做“可运行骨架”的最小里程碑

- M1：场景能加载（`air.usd` + 球 + 网），可单步仿真
- M2：动作链路打通（随机动作 -> 无人机稳定响应）
- M3：reset + 发球随机化可并行运行
- M4：奖励/终止返回正确张量形状
- M5：PPO 能启动并持续采样，不报 shape/NaN 错误

## 8. 与你的资产对接说明（air.usd）

- 在 `assets/refs.md` 固定记录资产绝对路径与版本
- `scene_builder.py` 里只暴露一个入口：`spawn_drone_with_racket()`
- 避免在多个模块重复写资产路径，统一走配置注入

## 9. 风险与预留扩展

- 风险1：羽毛球阻力模型与引擎步长耦合导致数值不稳定
  - 预留：`shuttle_aero.py` 中支持子步长系数与力裁剪
- 风险2：甜区接触判定抖动
  - 预留：`collision_model.py` 中加入接触滤波窗口
- 风险3：Actor 观测噪声过强导致训练初期崩溃
  - 预留：`curriculum.py` 逐步提升噪声强度

## 10. 下一步建议（你确认后我就开始搭代码空文件）

- 按上述目录一次性创建全部骨架文件（含类/函数签名与 TODO）
- 先打通 `InterceptEnv` + `scene_builder` + `reset_manager` 三件套
- 再补 `observations/rewards/terminations` 的最小占位实现，确保训练入口可启动

---

如果你认可这个框架，我下一步可以直接在该文件夹里生成完整空骨架代码（可 import、可运行到初始化阶段），然后你逐块审阅。
