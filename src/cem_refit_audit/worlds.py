from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def gaussian(x: Array, center: float, width: float) -> Array:
    """Unit-height Gaussian bump used for rewards and diagnostics."""
    width = max(float(width), 1e-8)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


@dataclass(frozen=True)
class ToyWorldConfig:
    """One-dimensional MPC world with a known hidden model-error pocket."""

    horizon: int = 8
    action_dim: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    init_state: float = 0.0
    state_clip: float = 1.35
    inertia: float = 0.72
    action_gain: float = 0.56
    dynamics_curve: float = 0.025
    goal: float = 0.82
    goal_width: float = 0.15
    goal_reward: float = 1.15
    trap: float = -0.56
    trap_width: float = 0.095
    trap_penalty: float = 0.95
    action_cost: float = 0.035
    state_cost: float = 0.015
    model_bonus: float = 1.28
    model_bonus_width_multiplier: float = 1.15
    ensemble_size: int = 7
    ensemble_seed: int = 17
    disagreement_scale: float = 0.75
    ood_scale: float = 0.45


class ToyWorld:
    """True environment plus an inspectable biased learned-model surrogate.

    The true world has a high-value goal and a narrow bad region. The learned
    model misses the bad reward and instead hallucinates a positive reward
    there. CEM can discover a partial pocket trajectory, refit toward it, and
    then generate increasingly unrealistic elites.
    """

    def __init__(self, config: ToyWorldConfig | None = None):
        self.config = config or ToyWorldConfig()
        rng = np.random.default_rng(self.config.ensemble_seed)
        self._bonus_scales = 1.0 + rng.normal(0.0, 0.12, self.config.ensemble_size)
        self._trap_offsets = rng.normal(0.0, self.config.trap_width * 0.22, self.config.ensemble_size)
        self._dynamics_offsets = rng.normal(0.0, 0.018, self.config.ensemble_size)

    @property
    def horizon(self) -> int:
        return self.config.horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def action_bounds(self) -> tuple[float, float]:
        return self.config.action_low, self.config.action_high

    def _as_batch(self, actions: Array) -> Array:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1:
            actions = actions[:, None]
        if actions.ndim == 2:
            actions = actions[None, :, :]
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape (batch, horizon, dim); got {actions.shape}")
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"expected action_dim={self.config.action_dim}; got {actions.shape[-1]}")
        return np.clip(actions, self.config.action_low, self.config.action_high)

    def next_state_true(self, x: Array, a: Array) -> Array:
        cfg = self.config
        curved = cfg.dynamics_curve * np.sin(3.0 * x)
        x_next = cfg.inertia * x + cfg.action_gain * np.tanh(a) + curved
        return np.clip(x_next, -cfg.state_clip, cfg.state_clip)

    def next_state_model(self, x: Array, a: Array, member: int | None = None) -> Array:
        cfg = self.config
        x_next = self.next_state_true(x, a)
        if member is not None:
            pocket = gaussian(x_next, cfg.trap + self._trap_offsets[member], cfg.trap_width * 1.5)
            x_next = x_next + self._dynamics_offsets[member] * pocket
        return np.clip(x_next, -cfg.state_clip, cfg.state_clip)

    def true_reward(self, x_next: Array, a: Array) -> Array:
        cfg = self.config
        goal = cfg.goal_reward * gaussian(x_next, cfg.goal, cfg.goal_width)
        trap = cfg.trap_penalty * gaussian(x_next, cfg.trap, cfg.trap_width)
        costs = cfg.action_cost * (a**2) + cfg.state_cost * (x_next**2)
        return goal - trap - costs

    def model_reward(self, x_next: Array, a: Array, member: int | None = None) -> Array:
        cfg = self.config
        goal = cfg.goal_reward * gaussian(x_next, cfg.goal, cfg.goal_width)
        center = cfg.trap
        bonus_scale = 1.0
        if member is not None:
            center = cfg.trap + self._trap_offsets[member]
            bonus_scale = self._bonus_scales[member]
        false_bonus = (
            cfg.model_bonus
            * bonus_scale
            * gaussian(x_next, center, cfg.trap_width * cfg.model_bonus_width_multiplier)
        )
        costs = cfg.action_cost * (a**2) + cfg.state_cost * (x_next**2)
        return goal + false_bonus - costs

    def rollout_states(self, actions: Array, x0: float | None = None, *, model_member: int | None = None) -> Array:
        actions = self._as_batch(actions)
        x = np.full(actions.shape[0], self.config.init_state if x0 is None else float(x0))
        states = [x.copy()]
        for t in range(actions.shape[1]):
            a = actions[:, t, 0]
            if model_member is None:
                x = self.next_state_true(x, a)
            else:
                x = self.next_state_model(x, a, model_member)
            states.append(x.copy())
        return np.stack(states, axis=1)

    def true_returns(self, actions: Array, x0: float | None = None) -> Array:
        actions = self._as_batch(actions)
        states = self.rollout_states(actions, x0)
        rewards = []
        for t in range(actions.shape[1]):
            rewards.append(self.true_reward(states[:, t + 1], actions[:, t, 0]))
        return np.sum(np.stack(rewards, axis=1), axis=1)

    def model_member_returns(self, actions: Array, x0: float | None = None) -> Array:
        actions = self._as_batch(actions)
        member_returns = []
        for member in range(self.config.ensemble_size):
            states = self.rollout_states(actions, x0, model_member=member)
            rewards = []
            for t in range(actions.shape[1]):
                rewards.append(self.model_reward(states[:, t + 1], actions[:, t, 0], member))
            member_returns.append(np.sum(np.stack(rewards, axis=1), axis=1))
        return np.stack(member_returns, axis=0)

    def model_returns(self, actions: Array, x0: float | None = None) -> Array:
        return np.mean(self.model_member_returns(actions, x0), axis=0)

    def trajectory_features(self, actions: Array, x0: float | None = None) -> dict[str, Array]:
        actions = self._as_batch(actions)
        states = self.rollout_states(actions, x0)
        cfg = self.config
        x = states[:, 1:]
        pocket = gaussian(x, cfg.trap, cfg.trap_width * 1.25)
        support_origin = gaussian(x, 0.0, 0.34)
        support_goal = gaussian(x, cfg.goal, 0.32)
        support = np.maximum(support_origin, support_goal)
        ood = 1.0 - np.mean(np.clip(support, 0.0, 1.0), axis=1)
        action_energy = np.mean(actions[:, :, 0] ** 2, axis=1)
        return {
            "pocket_occupancy": np.mean(pocket, axis=1),
            "pocket_max": np.max(pocket, axis=1),
            "ood": np.clip(ood, 0.0, 1.0),
            "action_energy": action_energy,
            "state_absmax": np.max(np.abs(x), axis=1),
        }

    def score_components(self, actions: Array, x0: float | None = None) -> dict[str, Array]:
        actions = self._as_batch(actions)
        member_returns = self.model_member_returns(actions, x0)
        model_return = np.mean(member_returns, axis=0)
        model_std = np.std(member_returns, axis=0)
        features = self.trajectory_features(actions, x0)
        disagreement = (
            model_std
            + self.config.disagreement_scale * features["pocket_occupancy"]
            + self.config.ood_scale * features["ood"]
        )
        components = dict(features)
        components.update(
            {
                "model_return": model_return,
                "disagreement": disagreement,
                "score": model_return,
            }
        )
        return components

    def step_true(self, x: float, action: Array | float) -> tuple[float, float]:
        a = float(np.asarray(action).reshape(-1)[0])
        a = float(np.clip(a, self.config.action_low, self.config.action_high))
        x_next = float(self.next_state_true(np.asarray([x]), np.asarray([a]))[0])
        reward = float(self.true_reward(np.asarray([x_next]), np.asarray([a]))[0])
        return x_next, reward
