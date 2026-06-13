from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cem_refit_audit.worlds import ToyWorld, gaussian


Array = np.ndarray


def _features(x: Array, a: Array) -> Array:
    return np.stack(
        [
            np.ones_like(x),
            x,
            a,
            x**2,
            a**2,
            x * a,
            x**3,
            a**3,
            np.sin(np.pi * x),
            np.cos(np.pi * x),
        ],
        axis=1,
    )


def _ridge_fit(x: Array, y: Array, ridge: float) -> Array:
    reg = ridge * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


@dataclass
class PolynomialEnsembleModel:
    """Small bootstrapped world model trained with sparse trap coverage."""

    true_world: ToyWorld
    transition_weights: Array
    reward_weights: Array
    hallucinated_bonus: Array
    hallucinated_centers: Array

    @classmethod
    def fit(
        cls,
        true_world: ToyWorld,
        *,
        ensemble_size: int = 7,
        n_transitions: int = 900,
        seed: int = 0,
        ridge: float = 1e-3,
    ) -> "PolynomialEnsembleModel":
        rng = np.random.default_rng(seed)
        cfg = true_world.config
        xs = []
        # Sparse data: mostly origin/goal support, very little in the trap.
        while len(xs) < n_transitions:
            proposal = rng.choice(
                [
                    rng.normal(0.05, 0.25),
                    rng.normal(cfg.goal, 0.18),
                    rng.uniform(-cfg.state_clip, cfg.state_clip),
                ],
                p=[0.58, 0.34, 0.08],
            )
            if abs(proposal - cfg.trap) > cfg.trap_width * 1.35 or rng.random() < 0.04:
                xs.append(float(np.clip(proposal, -cfg.state_clip, cfg.state_clip)))
        x = np.asarray(xs)
        a = rng.uniform(cfg.action_low, cfg.action_high, size=n_transitions)
        x_next = true_world.next_state_true(x, a)
        reward = true_world.true_reward(x_next, a)
        phi = _features(x, a)
        transition_weights = []
        reward_weights = []
        for _ in range(ensemble_size):
            idx = rng.integers(0, n_transitions, size=n_transitions)
            transition_weights.append(_ridge_fit(phi[idx], x_next[idx], ridge))
            reward_weights.append(_ridge_fit(phi[idx], reward[idx], ridge))
        bonus = rng.normal(0.42, 0.08, size=ensemble_size)
        centers = cfg.trap + rng.normal(0.0, cfg.trap_width * 0.25, size=ensemble_size)
        return cls(
            true_world=true_world,
            transition_weights=np.stack(transition_weights),
            reward_weights=np.stack(reward_weights),
            hallucinated_bonus=bonus,
            hallucinated_centers=centers,
        )

    @property
    def action_bounds(self) -> tuple[float, float]:
        return self.true_world.action_bounds

    def _as_batch(self, actions: Array) -> Array:
        return self.true_world._as_batch(actions)

    def member_returns(self, actions: Array, x0: float | None = None) -> Array:
        actions = self._as_batch(actions)
        cfg = self.true_world.config
        returns = []
        for member in range(self.transition_weights.shape[0]):
            x = np.full(actions.shape[0], cfg.init_state if x0 is None else float(x0))
            total = np.zeros(actions.shape[0])
            for t in range(actions.shape[1]):
                a = actions[:, t, 0]
                phi = _features(x, a)
                x_next = phi @ self.transition_weights[member]
                x_next = np.clip(x_next, -cfg.state_clip, cfg.state_clip)
                pred_reward = phi @ self.reward_weights[member]
                # Sparse-region hallucination: a compact basis unsupported by data.
                pred_reward = pred_reward + self.hallucinated_bonus[member] * gaussian(
                    x_next,
                    self.hallucinated_centers[member],
                    cfg.trap_width * 1.2,
                )
                total += pred_reward
                x = x_next
            returns.append(total)
        return np.stack(returns, axis=0)

    def score_components(self, actions: Array, x0: float | None = None) -> dict[str, Array]:
        actions = self._as_batch(actions)
        member_returns = self.member_returns(actions, x0)
        model_return = np.mean(member_returns, axis=0)
        model_std = np.std(member_returns, axis=0)
        features = self.true_world.trajectory_features(actions, x0)
        disagreement = model_std + 0.55 * features["pocket_occupancy"] + 0.35 * features["ood"]
        components = dict(features)
        components.update(
            {
                "model_return": model_return,
                "disagreement": disagreement,
                "score": model_return,
            }
        )
        return components
