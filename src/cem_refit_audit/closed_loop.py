from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ClosedLoopResult:
    total_reward: float
    states: list[float]
    actions: list[float]
    plan_true_returns: list[float | None]
    plan_model_returns: list[float]


def run_receding_horizon(
    world,
    planner_factory: Callable[[int], object],
    scorer_factory: Callable[[float], object],
    *,
    steps: int,
    seed: int,
    x0: float | None = None,
) -> ClosedLoopResult:
    """Run MPC by replanning after every real transition."""
    x = world.config.init_state if x0 is None else float(x0)
    states = [x]
    actions: list[float] = []
    true_returns: list[float | None] = []
    model_returns: list[float] = []
    total_reward = 0.0
    for t in range(steps):
        planner = planner_factory(seed + 1009 * t)
        scorer = scorer_factory(x)
        result = planner.plan(
            lambda batch, scorer=scorer, x=x: scorer(batch, x),
            utility_fn=lambda batch, x=x: world.true_returns(batch, x0=x),
        )
        action = float(result.actions[0, 0])
        x, reward = world.step_true(x, action)
        total_reward += reward
        states.append(x)
        actions.append(action)
        true_returns.append(result.selected_true_return)
        model_returns.append(result.selected_model_return)
    return ClosedLoopResult(
        total_reward=float(total_reward),
        states=states,
        actions=actions,
        plan_true_returns=true_returns,
        plan_model_returns=model_returns,
    )
