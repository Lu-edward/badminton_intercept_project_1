import gymnasium as gym

TASK_ID = "Isaac-Badminton-Intercept-Direct-v0"


def register_task() -> None:
    try:
        gym.spec(TASK_ID)
        return
    except Exception:
        pass

    gym.register(
        id=TASK_ID,
        entry_point="badminton_intercept.envs.intercept_env:InterceptEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "badminton_intercept.envs.intercept_env_cfg:InterceptEnvCfg",
            "rsl_rl_cfg_entry_point": "badminton_intercept.train.rsl_rl_ppo_cfg:BadmintonInterceptPPORunnerCfg",
        },
    )
