from __future__ import annotations

import numpy as np
import pandas as pd


Array = np.ndarray


def regret(optimal_return: float, achieved_return: float) -> float:
    return float(optimal_return - achieved_return)


def selected_tail_gap(model_return: float, true_return: float) -> float:
    return float(model_return - true_return)


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "optimal_true_return" in df and "selected_true_return" in df:
        df["regret"] = df["optimal_true_return"] - df["selected_true_return"]
    if "selected_model_return" in df and "selected_true_return" in df:
        df["selected_tail_gap"] = df["selected_model_return"] - df["selected_true_return"]
    return df


def estimate_true_optimum(
    world,
    horizon: int,
    action_dim: int,
    samples: int = 50_000,
    seed: int = 123,
    x0: float | None = None,
) -> tuple[float, Array]:
    """Monte Carlo plus deterministic probes for a reproducible oracle estimate."""
    rng = np.random.default_rng(seed)
    low, high = world.action_bounds
    random_actions = rng.uniform(low, high, size=(samples, horizon, action_dim))
    constants = np.linspace(low, high, 41)
    constant_actions = np.repeat(constants[:, None, None], horizon, axis=1)
    constant_actions = np.repeat(constant_actions, action_dim, axis=2)
    ramps = []
    for end in np.linspace(low, high, 21):
        ramp = np.linspace(0.0, end, horizon)[:, None]
        ramps.append(np.repeat(ramp[None, :, :], 1, axis=0))
    probe_actions = np.concatenate([constant_actions, *ramps], axis=0)
    candidates = np.concatenate([random_actions, probe_actions], axis=0)
    values = world.true_returns(candidates, x0=x0)
    idx = int(np.argmax(values))
    best_value = float(values[idx])
    best_actions = candidates[idx].copy()

    # Refine the oracle estimate with CEM on the true utility. This is used only
    # for evaluation, never by model-scored planners.
    cem_pop = int(np.clip(samples // 4, 1024, 4096))
    elite_count = max(8, cem_pop // 10)
    mean = np.zeros((horizon, action_dim), dtype=float)
    std = np.full_like(mean, 0.85)
    for _ in range(8):
        cem_actions = rng.normal(mean, std, size=(cem_pop, horizon, action_dim))
        cem_actions = np.clip(cem_actions, low, high)
        cem_values = world.true_returns(cem_actions, x0=x0)
        cem_idx = int(np.argmax(cem_values))
        if float(cem_values[cem_idx]) > best_value:
            best_value = float(cem_values[cem_idx])
            best_actions = cem_actions[cem_idx].copy()
        elite_idx = np.argsort(cem_values)[::-1][:elite_count]
        elites = cem_actions[elite_idx]
        mean = np.mean(elites, axis=0)
        std = np.maximum(np.std(elites, axis=0), 0.025)

    local_actions = rng.normal(best_actions, 0.06, size=(cem_pop, horizon, action_dim))
    local_actions = np.clip(local_actions, low, high)
    local_values = world.true_returns(local_actions, x0=x0)
    local_idx = int(np.argmax(local_values))
    if float(local_values[local_idx]) > best_value:
        best_value = float(local_values[local_idx])
        best_actions = local_actions[local_idx].copy()
    return best_value, best_actions


def trace_rows(seed: int, planner: str, history) -> list[dict]:
    rows = []
    for item in history:
        row = dict(item.__dict__)
        row["seed"] = seed
        row["planner"] = planner
        rows.append(row)
    return rows
