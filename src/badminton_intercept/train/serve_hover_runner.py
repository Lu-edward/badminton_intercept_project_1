from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torch.distributions import Normal

def _resolve_rsl_symbol(package_name: str, symbol_name: str, fallback_modules: tuple[str, ...]):
    """Resolve an rsl_rl symbol across version-dependent module layouts."""
    tried_modules: list[str] = []

    for module_name in fallback_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            tried_modules.append(module_name)
            continue
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
        tried_modules.append(module_name)

    package = importlib.import_module(package_name)
    if hasattr(package, symbol_name):
        return getattr(package, symbol_name)

    package_path = getattr(package, "__path__", None)
    if package_path is not None:
        prefix = package.__name__ + "."
        for module_info in pkgutil.walk_packages(package_path, prefix):
            module_name = module_info.name
            try:
                module = importlib.import_module(module_name)
            except Exception:
                tried_modules.append(module_name)
                continue
            if hasattr(module, symbol_name):
                return getattr(module, symbol_name)
            tried_modules.append(module_name)

    raise ImportError(
        f"Could not resolve '{symbol_name}' from '{package_name}'. Tried modules: {tried_modules}"
    )


PPO = _resolve_rsl_symbol(
    "rsl_rl.algorithms",
    "PPO",
    (
        "rsl_rl.algorithms",
        "rsl_rl.algorithms.ppo",
    ),
)

OnPolicyRunner = _resolve_rsl_symbol(
    "rsl_rl.runners",
    "OnPolicyRunner",
    (
        "rsl_rl.runners",
        "rsl_rl.runners.on_policy_runner",
    ),
)

class _NoOpLogger:
    """Fallback for rsl_rl versions that no longer expose a Logger class."""

    def __init__(self, *args, **kwargs) -> None:
        self.log_dir = kwargs.get("log_dir", None)

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None

        return _noop


try:
    Logger = _resolve_rsl_symbol(
        "rsl_rl",
        "Logger",
        (
            "rsl_rl.utils",
            "rsl_rl.utils.logger",
            "rsl_rl.runners",
            "rsl_rl.runners.logger",
        ),
    )
except ImportError:
    Logger = _NoOpLogger


class EmpiricalNormalization(nn.Module):
    """Lightweight running mean/variance normalizer compatible with older checkpoints."""

    def __init__(self, shape: int, eps: float = 1.0e-4) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_var", torch.ones(shape))
        self.register_buffer("_std", torch.ones(shape))
        self.register_buffer("count", torch.tensor(self.eps))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self._mean) / self._std.clamp_min(1.0e-6)

    def update(self, value: torch.Tensor) -> None:
        if value.numel() == 0:
            return
        value = value.detach()
        batch_mean = value.mean(dim=0)
        batch_var = value.var(dim=0, unbiased=False)
        batch_count = float(value.shape[0])
        delta = batch_mean - self._mean
        total_count = self.count + batch_count
        new_mean = self._mean + delta * batch_count / total_count
        m_a = self._var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        new_var = m2 / total_count
        self._mean.copy_(new_mean)
        self._var.copy_(new_var.clamp_min(1.0e-6))
        self._std.copy_(torch.sqrt(self._var).clamp_min(1.0e-6))
        self.count.fill_(float(total_count))


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int] | list[int], activation: str) -> nn.Sequential:
    dims = [input_dim, *list(hidden_dims), output_dim]
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


def _filter_kwargs_for_callable(fn, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Filter kwargs to match a callable signature while tolerating version differences."""
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(kwargs), []

    allowed = set(signature.parameters.keys())
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    dropped = [key for key in kwargs.keys() if key not in allowed]
    return filtered, dropped


class CompatActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        if kwargs:
            print("CompatActorCritic ignored unexpected kwargs: " + str(sorted(kwargs.keys())))
        if state_dependent_std:
            raise ValueError("CompatActorCritic does not support state_dependent_std.")

        self.obs_groups = obs_groups
        num_actor_obs = sum(int(obs[group].shape[-1]) for group in obs_groups["policy"])
        num_critic_obs = sum(int(obs[group].shape[-1]) for group in obs_groups["critic"])
        self.actor = _build_mlp(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = _build_mlp(num_critic_obs, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if self.actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if self.critic_obs_normalization else nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}")
        self.distribution: Normal | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self.actor(actor_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


def _squeeze_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze tensor from [1, dim] to [dim] if needed for compatibility."""
    if tensor.dim() == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def build_combined_model_state_from_split_checkpoint(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    model_state: dict[str, torch.Tensor] = {}

    for key, value in checkpoint["actor_state_dict"].items():
        if key == "obs_normalizer._mean":
            model_state["actor_obs_normalizer._mean"] = _squeeze_if_needed(value)
        elif key == "obs_normalizer._var":
            model_state["actor_obs_normalizer._var"] = _squeeze_if_needed(value)
        elif key == "obs_normalizer._std":
            model_state["actor_obs_normalizer._std"] = _squeeze_if_needed(value)
        elif key == "obs_normalizer.count":
            model_state["actor_obs_normalizer.count"] = value
        elif key == "distribution.log_std_param":
            model_state["log_std"] = value
        elif key.startswith("mlp."):
            model_state["actor." + key[4:]] = value
        else:
            model_state[key] = value

    for key, value in checkpoint["critic_state_dict"].items():
        if key == "obs_normalizer._mean":
            model_state["critic_obs_normalizer._mean"] = _squeeze_if_needed(value)
        elif key == "obs_normalizer._var":
            model_state["critic_obs_normalizer._var"] = _squeeze_if_needed(value)
        elif key == "obs_normalizer._std":
            model_state["critic_obs_normalizer._std"] = _squeeze_if_needed(value)
        elif key == "obs_normalizer.count":
            model_state["critic_obs_normalizer.count"] = value
        elif key.startswith("mlp."):
            model_state["critic." + key[4:]] = value
        else:
            model_state[key] = value

    return model_state


def extract_model_state_dict(checkpoint_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    if "actor_state_dict" in checkpoint_payload and "critic_state_dict" in checkpoint_payload:
        return build_combined_model_state_from_split_checkpoint(checkpoint_payload)
    return checkpoint_payload


def split_actor_critic_state_dict(actor_critic: CompatActorCritic) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    actor_state: dict[str, torch.Tensor] = {}
    critic_state: dict[str, torch.Tensor] = {}
    for key, value in actor_critic.state_dict().items():
        if key.startswith("actor."):
            actor_state["mlp." + key[len("actor."):]] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state["obs_normalizer." + key[len("actor_obs_normalizer."):]] = value
        elif key == "log_std":
            actor_state["distribution.log_std_param"] = value
        elif key == "std":
            actor_state["std"] = value
        elif key.startswith("critic."):
            critic_state["mlp." + key[len("critic."):]] = value
        elif key.startswith("critic_obs_normalizer."):
            critic_state["obs_normalizer." + key[len("critic_obs_normalizer."):]] = value
    return actor_state, critic_state


class ServeHoverDualActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        num_actions: int,
        prehit_checkpoint_path: str | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        if kwargs:
            print("ServeHoverDualActorCritic ignored unexpected kwargs: " + str(sorted(kwargs.keys())))

        common_kwargs = dict(
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
        )
        self.task2_actor_critic = CompatActorCritic(
            obs,
            {"policy": ["task2_policy"], "critic": ["task2_critic_obs"]},
            num_actions,
            **common_kwargs,
        )
        self.hover_actor_critic = CompatActorCritic(
            obs,
            {"policy": ["policy"], "critic": ["critic"]},
            num_actions,
            **common_kwargs,
        )
        self.num_actions = num_actions
        self.distribution: Normal | None = None

        if prehit_checkpoint_path:
            checkpoint_payload = torch.load(prehit_checkpoint_path, weights_only=False, map_location="cpu")
            model_state = extract_model_state_dict(checkpoint_payload)
            self.task2_actor_critic.load_state_dict(model_state, strict=True)

        for parameter in self.task2_actor_critic.parameters():
            parameter.requires_grad = False

    @property
    def task2_actor(self):
        return self.task2_actor_critic.actor

    @property
    def task2_critic(self):
        return self.task2_actor_critic.critic

    @property
    def hover_actor(self):
        return self.hover_actor_critic.actor

    @property
    def hover_critic(self):
        return self.hover_actor_critic.critic

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        # rsl_rl logs policy.action_std directly. In serve-hover mode we care about
        # the trainable hover branch std, while task2 is frozen and filtered out.
        hover_distribution = getattr(self.hover_actor_critic, "distribution", None)
        if hover_distribution is not None:
            return hover_distribution.stddev
        if hasattr(self.hover_actor_critic, "log_std"):
            return torch.exp(self.hover_actor_critic.log_std)
        if hasattr(self.hover_actor_critic, "std"):
            return self.hover_actor_critic.std
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # ========== New rsl_rl MLPModel interface compatibility ==========

    @property
    def output_mean(self) -> torch.Tensor:
        """Alias for action_mean for rsl_rl compatibility."""
        return self.action_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Alias for action_std for rsl_rl compatibility."""
        return self.action_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Alias for entropy for rsl_rl compatibility."""
        return self.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return distribution parameters (mean, std) for rsl_rl compatibility."""
        return (self.distribution.mean, self.distribution.stddev)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs).sum(dim=-1)

    def get_hidden_state(self):
        """Return hidden state for recurrent models (None for MLP)."""
        return None

    def forward(
        self,
        obs: TensorDict,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Default forward keeps actor behavior for backward compatibility."""
        return self.forward_actor(obs, masks=masks, hidden_state=hidden_state, stochastic_output=stochastic_output)

    def forward_actor(
        self,
        obs: TensorDict,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        distribution = self._build_combined_distribution(obs)
        if stochastic_output:
            return distribution.sample()
        return distribution.mean

    def forward_critic(
        self,
        obs: TensorDict,
        masks=None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        return self.evaluate(obs)

    # ========== End new rsl_rl interface ==========

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None

    @staticmethod
    def _resolve_serve_hover_mask(obs: TensorDict) -> torch.Tensor:
        mask_key = "serve_hover_mask" if "serve_hover_mask" in obs.keys() else "post_hit_mask"
        mask = obs[mask_key]
        if mask.dim() > 1:
            mask = mask.squeeze(-1)
        return mask >= 0.5

    def _branch_distribution(self, actor_critic: CompatActorCritic, actor_obs: torch.Tensor) -> Normal:
        actor_critic._update_distribution(actor_obs)
        return actor_critic.distribution

    def _build_combined_distribution(self, obs: TensorDict) -> Normal:
        post_hit_mask = self._resolve_serve_hover_mask(obs).unsqueeze(-1)

        task2_actor_obs = self.task2_actor_critic.actor_obs_normalizer(self.task2_actor_critic.get_actor_obs(obs))
        hover_actor_obs = self.hover_actor_critic.actor_obs_normalizer(self.hover_actor_critic.get_actor_obs(obs))

        task2_distribution = self._branch_distribution(self.task2_actor_critic, task2_actor_obs)
        hover_distribution = self._branch_distribution(self.hover_actor_critic, hover_actor_obs)

        mean = torch.where(post_hit_mask, hover_distribution.mean, task2_distribution.mean)
        std = torch.where(post_hit_mask, hover_distribution.stddev, task2_distribution.stddev)
        self.distribution = Normal(mean, std)
        return self.distribution

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        distribution = self._build_combined_distribution(obs)
        return distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        post_hit_mask = self._resolve_serve_hover_mask(obs).unsqueeze(-1)
        task2_actions = self.task2_actor_critic.act_inference(obs)
        hover_actions = self.hover_actor_critic.act_inference(obs)
        return torch.where(post_hit_mask, hover_actions, task2_actions)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        post_hit_mask = self._resolve_serve_hover_mask(obs).unsqueeze(-1)
        task2_values = self.task2_actor_critic.evaluate(obs)
        hover_values = self.hover_actor_critic.evaluate(obs)
        return torch.where(post_hit_mask, hover_values, task2_values)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        post_hit_mask = self._resolve_serve_hover_mask(obs)
        if not bool(torch.any(post_hit_mask)):
            return
        masked_obs = TensorDict(
            {key: value[post_hit_mask] for key, value in obs.items()},
            batch_size=[int(post_hit_mask.sum().item())],
            device=obs.device,
        )
        self.hover_actor_critic.update_normalization(masked_obs)

    @staticmethod
    def _unique_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
        unique_parameters: list[torch.nn.Parameter] = []
        seen_ids: set[int] = set()
        for param in parameters:
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen_ids:
                continue
            seen_ids.add(param_id)
            unique_parameters.append(param)
        return unique_parameters

    def hover_parameters(self) -> list[torch.nn.Parameter]:
        return self._unique_parameters(list(self.hover_actor_critic.parameters()))

    def hover_actor_parameters(self) -> list[torch.nn.Parameter]:
        actor_parameters = list(self.hover_actor.parameters())
        if hasattr(self.hover_actor_critic, "log_std"):
            actor_parameters.append(self.hover_actor_critic.log_std)
        if hasattr(self.hover_actor_critic, "std"):
            actor_parameters.append(self.hover_actor_critic.std)
        return self._unique_parameters(actor_parameters)

    def hover_critic_parameters(self) -> list[torch.nn.Parameter]:
        return self._unique_parameters(list(self.hover_critic.parameters()))

    def set_action_std(self, target_std: float) -> None:
        if target_std <= 0.0:
            raise ValueError("target_std must be > 0.")
        with torch.no_grad():
            if hasattr(self.hover_actor_critic, "log_std"):
                self.hover_actor_critic.log_std.fill_(torch.log(torch.tensor(target_std, device=self.hover_actor_critic.log_std.device)))
            elif hasattr(self.hover_actor_critic, "std"):
                self.hover_actor_critic.std.fill_(target_std)
            else:
                raise RuntimeError("Hover actor does not expose a supported action std parameter.")

    def get_action_std_mean(self) -> float:
        if hasattr(self.hover_actor_critic, "log_std"):
            return float(torch.exp(self.hover_actor_critic.log_std).mean().item())
        if hasattr(self.hover_actor_critic, "std"):
            return float(self.hover_actor_critic.std.mean().item())
        raise RuntimeError("Hover actor does not expose a supported action std parameter.")


class ServeHoverActorAdapter(nn.Module):
    is_recurrent: bool = False

    def __init__(self, policy: ServeHoverDualActorCritic) -> None:
        super().__init__()
        self.policy = policy

    @property
    def output_mean(self) -> torch.Tensor:
        return self.policy.output_mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.policy.output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.policy.output_entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.output_distribution_params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.policy.get_output_log_prob(outputs)

    def get_hidden_state(self):
        return self.policy.get_hidden_state()

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.policy.reset(dones)

    def update_normalization(self, obs: TensorDict) -> None:
        self.policy.update_normalization(obs)

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        return self.policy.forward_actor(obs, masks=masks, hidden_state=hidden_state, stochastic_output=stochastic_output)

    def parameters(self, recurse: bool = True):
        return iter(self.policy.hover_actor_parameters())


class ServeHoverCriticAdapter(nn.Module):
    is_recurrent: bool = False

    def __init__(self, policy: ServeHoverDualActorCritic) -> None:
        super().__init__()
        self.policy = policy

    def get_hidden_state(self):
        return self.policy.get_hidden_state()

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.policy.reset(dones)

    def update_normalization(self, obs: TensorDict) -> None:
        self.policy.update_normalization(obs)

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        return self.policy.forward_critic(obs, masks=masks, hidden_state=hidden_state, stochastic_output=stochastic_output)

    def parameters(self, recurse: bool = True):
        return iter(self.policy.hover_critic_parameters())


class ServeHoverMaskedPPO(PPO):
    """PPO variant for Serve-Hover dual policy training.
    
    This class adapts the new rsl_rl PPO interface (which expects separate actor/critic)
    to work with ServeHoverDualActorCritic (which combines both roles).
    """
    
    def __init__(self, policy: ServeHoverDualActorCritic, *args, **kwargs) -> None:
        storage = kwargs.pop("storage", None)
        self.actor_freeze_iterations = int(kwargs.pop("actor_freeze_iterations", 0) or 0)
        self.actor_lr_scale = float(kwargs.pop("actor_lr_scale", 1.0) or 1.0)
        if self.actor_freeze_iterations < 0:
            raise ValueError("actor_freeze_iterations must be >= 0.")
        if self.actor_lr_scale < 0.0:
            raise ValueError("actor_lr_scale must be >= 0.")
        actor_model = ServeHoverActorAdapter(policy)
        critic_model = ServeHoverCriticAdapter(policy)

        ppo_signature = inspect.signature(PPO.__init__)
        ppo_param_names = [name for name in ppo_signature.parameters.keys() if name != "self"]
        ppo_accepts_storage = "storage" in ppo_signature.parameters
        ppo_uses_split_actor_critic = (
            len(ppo_param_names) >= 2
            and ppo_param_names[0] in {"actor", "actor_model"}
            and ppo_param_names[1] in {"critic", "critic_model"}
        )
        if storage is not None and ppo_accepts_storage:
            kwargs["storage"] = storage

        filtered_kwargs, dropped_keys = _filter_kwargs_for_callable(PPO.__init__, kwargs)
        if dropped_keys:
            print("ServeHoverMaskedPPO dropped unsupported PPO kwargs: " + str(sorted(dropped_keys)))

        if ppo_uses_split_actor_critic:
            super().__init__(actor_model, critic_model, *args, **filtered_kwargs)
        else:
            super().__init__(policy, *args, **filtered_kwargs)
        if storage is not None and not ppo_accepts_storage:
            self.storage = storage

        self.policy = policy
        self.actor = actor_model
        self.critic = critic_model

        if self.rnd or self.symmetry:
            raise ValueError("ServeHoverMaskedPPO currently supports plain PPO only.")
        self._hover_actor_parameters = policy.hover_actor_parameters()
        self._hover_critic_parameters = policy.hover_critic_parameters()
        self._hover_parameters = policy.hover_parameters()
        self._serve_hover_update_count = 0
        self._last_rollout_post_hit_samples = 0
        self._last_rollout_post_hit_fraction = 0.0
        self.optimizer = optim.Adam(
            [
                {"params": self._hover_actor_parameters, "lr": self.learning_rate, "name": "hover_actor"},
                {"params": self._hover_critic_parameters, "lr": self.learning_rate, "name": "hover_critic"},
            ],
            lr=self.learning_rate,
        )
        self._set_optimizer_lrs()

    def _is_actor_frozen(self) -> bool:
        return self._serve_hover_update_count < self.actor_freeze_iterations

    def _set_optimizer_lrs(self) -> None:
        actor_lr = 0.0 if self._is_actor_frozen() else self.learning_rate * self.actor_lr_scale
        for idx, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get("name")
            if group_name == "hover_actor" or (group_name is None and idx == 0):
                param_group["lr"] = actor_lr
            else:
                param_group["lr"] = self.learning_rate

    @staticmethod
    def _mask_obs_batch(obs_batch: TensorDict, mask: torch.Tensor) -> TensorDict:
        return TensorDict(
            {key: value[mask] for key, value in obs_batch.items()},
            batch_size=[int(mask.sum().item())],
            device=obs_batch.device,
        )

    @staticmethod
    def _get_batch_field(batch, *names: str):
        for name in names:
            if isinstance(batch, dict) and name in batch:
                return batch[name]
            if hasattr(batch, name):
                return getattr(batch, name)
            try:
                return batch[name]
            except Exception:
                pass
        available_fields = []
        if hasattr(batch, "keys"):
            try:
                available_fields = list(batch.keys())
            except Exception:
                available_fields = []
        if not available_fields:
            available_fields = [name for name in dir(batch) if not name.startswith("_")]
        raise AttributeError(
            f"Could not resolve any of batch fields: {names}. Available fields: {available_fields}"
        )

    def _unpack_batch(self, batch):
        if isinstance(batch, tuple):
            return batch

        obs_batch = self._get_batch_field(batch, "obs", "observations")
        actions_batch = self._get_batch_field(batch, "actions")
        target_values_batch = self._get_batch_field(batch, "target_values", "values", "old_values")
        advantages_batch = self._get_batch_field(batch, "advantages")
        returns_batch = self._get_batch_field(batch, "returns")
        old_actions_log_prob_batch = self._get_batch_field(
            batch,
            "old_actions_log_prob",
            "old_actions_log_prob_batch",
            "actions_log_prob",
            "old_log_prob",
            "old_log_probs",
        )
        try:
            old_distribution_params = self._get_batch_field(
                batch,
                "old_distribution_params",
                "distribution_params",
            )
            old_mu_batch, old_sigma_batch = old_distribution_params
        except AttributeError:
            old_mu_batch = self._get_batch_field(
                batch,
                "old_mu",
                "old_output_mean",
                "output_mean",
                "mu",
                "old_action_mean",
                "action_mean",
                "action_means",
                "mean",
                "actor_mean",
                "old_actor_mean",
                "distribution_mean",
                "old_distribution_mean",
            )
            old_sigma_batch = self._get_batch_field(
                batch,
                "old_sigma",
                "old_output_std",
                "output_std",
                "sigma",
                "std",
                "old_action_std",
                "action_std",
                "action_stds",
                "actor_std",
                "old_actor_std",
                "distribution_std",
                "old_distribution_std",
                "scale",
            )
        hidden_states_batch = None
        masks_batch = None
        try:
            hidden_states_batch = self._get_batch_field(batch, "hidden_states", "hidden_state")
        except AttributeError:
            hidden_states_batch = None
        try:
            masks_batch = self._get_batch_field(batch, "masks")
        except AttributeError:
            masks_batch = None

        return (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        )

    def _get_post_hit_storage_mask(self) -> torch.Tensor:
        mask_key = "serve_hover_mask" if "serve_hover_mask" in self.storage.observations.keys() else "post_hit_mask"
        post_hit_mask = self.storage.observations[mask_key].squeeze(-1) >= 0.5
        post_hit_count = int(post_hit_mask.sum().item())
        total_count = int(post_hit_mask.numel())
        self._last_rollout_post_hit_samples = post_hit_count
        self._last_rollout_post_hit_fraction = float(post_hit_count) / float(max(total_count, 1))
        return post_hit_mask

    def _normalize_post_hit_advantages(self) -> None:
        post_hit_mask = self._get_post_hit_storage_mask()
        if self._last_rollout_post_hit_samples == 0:
            return

        hover_advantages = self.storage.advantages[post_hit_mask]
        hover_std = hover_advantages.std(unbiased=False).clamp_min(1.0e-8)
        self.storage.advantages[post_hit_mask] = (hover_advantages - hover_advantages.mean()) / hover_std

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute raw GAE locally so post-hit advantages can be normalized only on
        # the samples that are actually used to update the hover policy.
        last_values = self.policy.evaluate(obs).detach()
        advantage = torch.zeros_like(last_values)

        for step in reversed(range(self.storage.rewards.shape[0])):
            next_values = last_values if step == self.storage.rewards.shape[0] - 1 else self.storage.values[step + 1]
            next_is_not_terminal = 1.0 - self.storage.dones[step].float()
            delta = (
                self.storage.rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - self.storage.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            self.storage.returns[step] = advantage + self.storage.values[step]

        self.storage.advantages = self.storage.returns - self.storage.values
        if not self.normalize_advantage_per_mini_batch:
            self._normalize_post_hit_advantages()
        else:
            self._get_post_hit_storage_mask()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        processed_batches = 0
        hover_sample_count = 0
        actor_frozen = self._is_actor_frozen()
        self._set_optimizer_lrs()

        if self.policy.is_recurrent:
            raise ValueError("ServeHoverMaskedPPO does not support recurrent policies.")

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                _hidden_states_batch,
                _masks_batch,
            ) = self._unpack_batch(batch)
            mask_key = "serve_hover_mask" if "serve_hover_mask" in obs_batch.keys() else "post_hit_mask"
            hover_mask = obs_batch[mask_key].squeeze(-1) >= 0.5
            batch_hover_samples = int(hover_mask.sum().item())
            if batch_hover_samples == 0:
                continue
            hover_sample_count += batch_hover_samples

            obs_batch = self._mask_obs_batch(obs_batch, hover_mask)
            actions_batch = actions_batch[hover_mask]
            target_values_batch = target_values_batch[hover_mask]
            advantages_batch = advantages_batch[hover_mask]
            returns_batch = returns_batch[hover_mask]
            old_actions_log_prob_batch = old_actions_log_prob_batch[hover_mask]
            old_mu_batch = old_mu_batch[hover_mask]
            old_sigma_batch = old_sigma_batch[hover_mask]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch - advantages_batch.mean()
                    ) / advantages_batch.std(unbiased=False).clamp_min(1.0e-8)

            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    self._set_optimizer_lrs()

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self._hover_parameters, self.max_grad_norm)
            self.optimizer.step()

            processed_batches += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        self.storage.clear()
        self._serve_hover_update_count += 1
        if processed_batches == 0:
            return {
                "value_function": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "hover_batches": 0.0,
                "hover_samples": float(hover_sample_count),
                "rollout_post_hit_samples": float(self._last_rollout_post_hit_samples),
                "rollout_post_hit_fraction": float(self._last_rollout_post_hit_fraction),
                "hover_action_std": 0.0,
                "actor_frozen": float(actor_frozen),
                "actor_freeze_remaining": float(max(self.actor_freeze_iterations - self._serve_hover_update_count, 0)),
            }

        return {
            "value_function": mean_value_loss / processed_batches,
            "surrogate": mean_surrogate_loss / processed_batches,
            "entropy": mean_entropy / processed_batches,
            "hover_batches": float(processed_batches),
            "hover_samples": float(hover_sample_count),
            "rollout_post_hit_samples": float(self._last_rollout_post_hit_samples),
            "rollout_post_hit_fraction": float(self._last_rollout_post_hit_fraction),
            "hover_action_std": float(self.policy.action_std.mean().item()),
            "actor_frozen": float(actor_frozen),
            "actor_freeze_remaining": float(max(self.actor_freeze_iterations - self._serve_hover_update_count, 0)),
        }


class ServeHoverOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner with custom ServeHover algorithm."""
        # Store basic attributes (copied from parent __init__)
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the custom algorithm using our own logic
        self.alg = self._construct_algorithm(obs)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def _get_alg_policy(self) -> ServeHoverDualActorCritic:
        # First check if we stored the policy directly
        if hasattr(self, "_serve_hover_policy") and self._serve_hover_policy is not None:
            return self._serve_hover_policy
        # Fallback: check alg attributes (new rsl_rl interface)
        if hasattr(self, "alg") and self.alg is not None:
            if hasattr(self.alg, "policy"):
                return self.alg.policy
            if hasattr(self.alg, "actor_critic"):
                return self.alg.actor_critic
            if hasattr(self.alg, "actor") and hasattr(self.alg, "critic"):
                # In ServeHoverMaskedPPO, actor is the ServeHoverDualActorCritic
                return self.alg.actor
        raise AttributeError(
            f"Unsupported PPO interface on {type(self.alg).__name__}: expected runner-held policy,"
            " 'policy', 'actor_critic', or separate 'actor'/'critic'."
        )

    def _construct_algorithm(self, obs: TensorDict) -> ServeHoverMaskedPPO:
        from rsl_rl.storage import RolloutStorage

        # Get configuration from self.cfg
        # New rsl_rl uses 'actor' and 'critic' keys instead of 'policy'
        actor_cfg = self.cfg.get("actor", {})
        critic_cfg = self.cfg.get("critic", {})
        policy_cfg = self.cfg.get("policy", {})  # Fallback for legacy config
        alg_cfg = self.cfg.get("algorithm", {})
        num_steps_per_env = self.cfg.get("num_steps_per_env", 24)

        # Build policy_kwargs by mapping new rsl_rl config names to ServeHoverDualActorCritic expected names
        policy_kwargs = {}
        
        # Map hidden_dims -> actor_hidden_dims, critic_hidden_dims
        if "hidden_dims" in actor_cfg:
            policy_kwargs["actor_hidden_dims"] = actor_cfg["hidden_dims"]
        if "hidden_dims" in critic_cfg:
            policy_kwargs["critic_hidden_dims"] = critic_cfg["hidden_dims"]
        
        # Map obs_normalization -> actor_obs_normalization, critic_obs_normalization
        if "obs_normalization" in actor_cfg:
            policy_kwargs["actor_obs_normalization"] = actor_cfg["obs_normalization"]
        if "obs_normalization" in critic_cfg:
            policy_kwargs["critic_obs_normalization"] = critic_cfg["obs_normalization"]
        
        # Handle activation
        if "activation" in actor_cfg:
            policy_kwargs["activation"] = actor_cfg["activation"]
        
        # Handle distribution_cfg -> init_noise_std, noise_std_type
        dist_cfg = actor_cfg.get("distribution_cfg", {})
        if dist_cfg:
            if "init_std" in dist_cfg:
                policy_kwargs["init_noise_std"] = dist_cfg["init_std"]
            if "std_type" in dist_cfg:
                policy_kwargs["noise_std_type"] = dist_cfg["std_type"]
        
        # Fallback: if policy_cfg exists (legacy), use it directly
        for key in ["actor_hidden_dims", "critic_hidden_dims", "actor_obs_normalization", 
                    "critic_obs_normalization", "activation", "init_noise_std", "noise_std_type"]:
            if key in policy_cfg and key not in policy_kwargs:
                policy_kwargs[key] = policy_cfg[key]

        policy = ServeHoverDualActorCritic(
            obs,
            self.env.num_actions,
            prehit_checkpoint_path=self.cfg.get("prehit_checkpoint_path"),
            **policy_kwargs,
        ).to(self.device)
        self._serve_hover_policy = policy

        # Create storage for new rsl_rl interface
        storage = RolloutStorage(
            training_type="rl",
            num_envs=self.env.num_envs,
            num_transitions_per_env=num_steps_per_env,
            obs=obs,
            actions_shape=[self.env.num_actions],
            device=self.device,
        )

        alg_kwargs = dict(alg_cfg)
        alg_kwargs.pop("class_name", None)
        alg_kwargs.pop("share_cnn_encoders", None)
        alg_kwargs.pop("actor", None)
        alg_kwargs.pop("critic", None)
        alg_kwargs.pop("storage", None)
        multi_gpu_cfg = self.cfg.get("multi_gpu")
        alg = ServeHoverMaskedPPO(
            policy,
            storage=storage,
            device=self.device,
            actor_freeze_iterations=self.cfg.get("serve_hover_actor_freeze_iterations", 0),
            actor_lr_scale=self.cfg.get("serve_hover_actor_lr_scale", 1.0),
            **alg_kwargs,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        return alg

    def save(self, path: str, infos: dict | None = None) -> None:
        policy = self._get_alg_policy()
        task2_actor_state_dict, task2_critic_state_dict = split_actor_critic_state_dict(policy.task2_actor_critic)
        hover_actor_state_dict, hover_critic_state_dict = split_actor_critic_state_dict(policy.hover_actor_critic)
        saved_dict = {
            "task2_actor_state_dict": task2_actor_state_dict,
            "task2_critic_state_dict": task2_critic_state_dict,
            "hover_actor_state_dict": hover_actor_state_dict,
            "hover_critic_state_dict": hover_critic_state_dict,
            "hover_optimizer_state_dict": self.alg.optimizer.state_dict(),
            "model_state_dict": policy.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        policy = self._get_alg_policy()
        if "task2_actor_state_dict" in loaded_dict:
            task2_checkpoint = {
                "actor_state_dict": loaded_dict["task2_actor_state_dict"],
                "critic_state_dict": loaded_dict["task2_critic_state_dict"],
            }
            hover_checkpoint = {
                "actor_state_dict": loaded_dict["hover_actor_state_dict"],
                "critic_state_dict": loaded_dict["hover_critic_state_dict"],
            }
            policy.task2_actor_critic.load_state_dict(
                build_combined_model_state_from_split_checkpoint(task2_checkpoint), strict=True
            )
            policy.hover_actor_critic.load_state_dict(
                build_combined_model_state_from_split_checkpoint(hover_checkpoint), strict=True
            )
            if load_optimizer and "hover_optimizer_state_dict" in loaded_dict:
                try:
                    self.alg.optimizer.load_state_dict(loaded_dict["hover_optimizer_state_dict"])
                except ValueError as exc:
                    print(f"[WARN] Skipping incompatible hover optimizer state: {exc}")
            self.current_learning_iteration = int(loaded_dict.get("iter", 0))
            if hasattr(self.alg, "_serve_hover_update_count"):
                self.alg._serve_hover_update_count = self.current_learning_iteration
                self.alg._set_optimizer_lrs()
            return loaded_dict.get("infos")

        model_state_dict = loaded_dict["model_state_dict"]
        policy.load_state_dict(model_state_dict, strict=True)
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except ValueError as exc:
                print(f"[WARN] Skipping incompatible optimizer state: {exc}")
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        if hasattr(self.alg, "_serve_hover_update_count"):
            self.alg._serve_hover_update_count = self.current_learning_iteration
            self.alg._set_optimizer_lrs()
        return loaded_dict.get("infos")
