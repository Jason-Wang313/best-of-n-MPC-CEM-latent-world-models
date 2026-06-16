from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v4_gymnasium_cem_cards"
PAPER = ROOT / "paper" / "iclr_submission"
FIGURES = PAPER / "figures" / "v4"


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    horizon: int
    risk_penalty: float
    progress_bonus: float
    kwargs: dict[str, object] | None = None


TASKS = [
    TaskSpec(
        env_id="FrozenLake-v1",
        horizon=12,
        risk_penalty=0.10,
        progress_bonus=0.05,
        kwargs={"map_name": "4x4", "is_slippery": True},
    ),
    TaskSpec(
        env_id="CliffWalking-v1",
        horizon=18,
        risk_penalty=80.0,
        progress_bonus=0.07,
    ),
    TaskSpec(
        env_id="Taxi-v3",
        horizon=16,
        risk_penalty=4.0,
        progress_bonus=0.04,
    ),
]


PLANNER_LABELS = {
    "static": "static",
    "equal_static": "equal-budget static",
    "cem": "vanilla CEM",
    "risk_gated_cem": "risk-gated CEM",
}


def make_env(spec: TaskSpec):
    return gym.make(spec.env_id, **(spec.kwargs or {}))


def initial_state(env, seed: int) -> int:
    env.reset(seed=seed)
    return int(env.unwrapped.s)


def progress_feature(spec: TaskSpec, env, state: int, next_state: int) -> float:
    if spec.env_id.startswith("FrozenLake"):
        ncol = int(env.unwrapped.ncol)
        nrow = int(env.unwrapped.nrow)
        goal = nrow * ncol - 1
        goal_row, goal_col = divmod(goal, ncol)
        row, col = divmod(next_state, ncol)
        return -0.02 * (abs(goal_row - row) + abs(goal_col - col))
    if spec.env_id.startswith("CliffWalking"):
        row, col = divmod(next_state, 12)
        return -0.01 * (abs(3 - row) + abs(11 - col))
    taxi = env.unwrapped
    row, col, passenger_idx, destination_idx = taxi.decode(next_state)
    locations = taxi.locs
    target = locations[passenger_idx] if passenger_idx < 4 else locations[destination_idx]
    return -0.01 * (abs(row - target[0]) + abs(col - target[1]))


class TransitionEvaluator:
    def __init__(self, spec: TaskSpec, seed: int):
        self.spec = spec
        self.env = make_env(spec)
        self.transitions = self.env.unwrapped.P
        self.start_state = initial_state(self.env, seed)
        self.cache: dict[tuple[tuple[int, ...], str], tuple[float, float]] = {}

    def evaluate_one(self, sequence: np.ndarray, mode: str) -> tuple[float, float]:
        key = (tuple(int(x) for x in sequence), mode)
        if key in self.cache:
            return self.cache[key]

        distribution: dict[int, float] = {self.start_state: 1.0}
        expected_return = 0.0
        bad_event_probability = 0.0

        for action in key[0]:
            next_distribution: dict[int, float] = {}
            for state, state_probability in distribution.items():
                for probability, next_state, reward, done in self.transitions[state][action]:
                    mass = state_probability * float(probability)
                    true_reward = float(reward)
                    bad_event = 0.0

                    if self.spec.env_id.startswith("FrozenLake"):
                        goal_state = int(self.env.unwrapped.nrow * self.env.unwrapped.ncol - 1)
                        bad_event = float(bool(done) and true_reward == 0.0 and next_state != goal_state)
                        model_reward = (
                            true_reward
                            + self.spec.progress_bonus
                            + progress_feature(self.spec, self.env, state, next_state)
                        )
                    elif self.spec.env_id.startswith("CliffWalking"):
                        bad_event = float(true_reward <= -50.0)
                        model_reward = (
                            (3.0 if bad_event else true_reward)
                            + self.spec.progress_bonus
                            + progress_feature(self.spec, self.env, state, next_state)
                        )
                    else:
                        bad_event = float(true_reward <= -10.0)
                        model_reward = (
                            max(true_reward, -1.0)
                            + self.spec.progress_bonus
                            + progress_feature(self.spec, self.env, state, next_state)
                        )

                    if mode == "true":
                        scored_reward = true_reward
                    elif mode == "model":
                        scored_reward = model_reward
                    elif mode == "repair":
                        scored_reward = model_reward - self.spec.risk_penalty * bad_event
                    else:
                        raise ValueError(f"unknown evaluation mode: {mode}")

                    expected_return += mass * scored_reward
                    bad_event_probability += mass * bad_event
                    if not done:
                        next_distribution[next_state] = next_distribution.get(next_state, 0.0) + mass
            distribution = next_distribution
            if not distribution:
                break

        self.cache[key] = (float(expected_return), float(bad_event_probability))
        return self.cache[key]

    def evaluate_batch(self, sequences: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        values: list[float] = []
        risks: list[float] = []
        for sequence in sequences:
            value, risk = self.evaluate_one(sequence, mode)
            values.append(value)
            risks.append(risk)
        return np.asarray(values, dtype=float), np.asarray(risks, dtype=float)


def sample_sequences(rng: np.random.Generator, probabilities: np.ndarray, population: int) -> np.ndarray:
    horizon, action_count = probabilities.shape
    samples = np.empty((population, horizon), dtype=int)
    for step in range(horizon):
        samples[:, step] = rng.choice(action_count, size=population, p=probabilities[step])
    return samples


def probability_entropy(probabilities: np.ndarray) -> float:
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12), axis=1).mean())


def run_planner(
    evaluator: TransitionEvaluator,
    spec: TaskSpec,
    *,
    seed: int,
    planner: str,
    population: int,
    iterations: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    action_count = int(evaluator.env.action_space.n)
    rng = np.random.default_rng(seed + 1009)
    elite_count = max(2, population // 10)
    uniform = np.full((spec.horizon, action_count), 1.0 / action_count)
    score_mode = "repair" if planner == "risk_gated_cem" else "model"
    curves: list[dict[str, object]] = []

    if planner in {"static", "equal_static"}:
        count = population if planner == "static" else population * iterations
        sequences = sample_sequences(rng, uniform, count)
        scores, risks = evaluator.evaluate_batch(sequences, score_mode)
        selected = int(np.argmax(scores))
        true_return, _ = evaluator.evaluate_batch(sequences[[selected]], "true")
        model_return, _ = evaluator.evaluate_batch(sequences[[selected]], "model")
        row = {
            "task": spec.env_id,
            "seed": seed,
            "planner": planner,
            "label": PLANNER_LABELS[planner],
            "selected_true_return": float(true_return[0]),
            "selected_model_return": float(model_return[0]),
            "selected_tail_gap": float(model_return[0] - true_return[0]),
            "bad_event_probability": float(risks[selected]),
            "proposal_drift": 0.0,
            "final_entropy": float(np.log(action_count)),
            "sequence": " ".join(str(int(x)) for x in sequences[selected]),
        }
        return row, curves

    probabilities = uniform.copy()
    best_sequence: np.ndarray | None = None
    best_score = -np.inf
    best_risk = 0.0
    initial_probabilities = probabilities.copy()

    for iteration in range(iterations):
        sequences = sample_sequences(rng, probabilities, population)
        scores, risks = evaluator.evaluate_batch(sequences, score_mode)
        order = np.argsort(scores)[::-1]
        selected = int(order[0])
        if float(scores[selected]) > best_score:
            best_score = float(scores[selected])
            best_sequence = sequences[selected].copy()
            best_risk = float(risks[selected])

        elites = sequences[order[:elite_count]]
        smoothing = 0.50 if planner == "risk_gated_cem" else 0.15
        counts = np.full((spec.horizon, action_count), smoothing)
        for step in range(spec.horizon):
            counts[step] += np.bincount(elites[:, step], minlength=action_count)
        probabilities = counts / counts.sum(axis=1, keepdims=True)
        mix = 0.25 if planner == "risk_gated_cem" else 0.03
        probabilities = (1.0 - mix) * probabilities + mix * (1.0 / action_count)

        assert best_sequence is not None
        best_true, _ = evaluator.evaluate_batch(best_sequence[None, :], "true")
        best_model, _ = evaluator.evaluate_batch(best_sequence[None, :], "model")
        curves.append(
            {
                "task": spec.env_id,
                "seed": seed,
                "planner": planner,
                "label": PLANNER_LABELS[planner],
                "iteration": iteration,
                "best_true_return": float(best_true[0]),
                "best_model_return": float(best_model[0]),
                "best_tail_gap": float(best_model[0] - best_true[0]),
                "best_bad_event_probability": best_risk,
                "proposal_drift": float(np.linalg.norm(probabilities - initial_probabilities)),
                "entropy": probability_entropy(probabilities),
                "elite_bad_event_probability": float(np.mean(risks[order[:elite_count]])),
            }
        )

    if best_sequence is None:
        raise RuntimeError("categorical CEM never selected a sequence")
    true_return, _ = evaluator.evaluate_batch(best_sequence[None, :], "true")
    model_return, _ = evaluator.evaluate_batch(best_sequence[None, :], "model")
    row = {
        "task": spec.env_id,
        "seed": seed,
        "planner": planner,
        "label": PLANNER_LABELS[planner],
        "selected_true_return": float(true_return[0]),
        "selected_model_return": float(model_return[0]),
        "selected_tail_gap": float(model_return[0] - true_return[0]),
        "bad_event_probability": best_risk,
        "proposal_drift": float(np.linalg.norm(probabilities - initial_probabilities)),
        "final_entropy": probability_entropy(probabilities),
        "sequence": " ".join(str(int(x)) for x in best_sequence),
    }
    return row, curves


def estimate_oracle(evaluator: TransitionEvaluator, spec: TaskSpec, seed: int) -> float:
    rows = []
    for planner in ("static", "equal_static", "cem", "risk_gated_cem"):
        row, _ = run_planner(
            evaluator,
            spec,
            seed=seed,
            planner=planner,
            population=160,
            iterations=5,
        )
        rows.append(float(row["selected_true_return"]))

    action_count = int(evaluator.env.action_space.n)
    probabilities = np.full((spec.horizon, action_count), 1.0 / action_count)
    rng = np.random.default_rng(seed + 77_001)
    best = -np.inf
    for _ in range(5):
        sequences = sample_sequences(rng, probabilities, 192)
        values, _ = evaluator.evaluate_batch(sequences, "true")
        order = np.argsort(values)[::-1]
        best = max(best, float(values[order[0]]))
        elites = sequences[order[: max(2, 192 // 10)]]
        counts = np.full((spec.horizon, action_count), 0.35)
        for step in range(spec.horizon):
            counts[step] += np.bincount(elites[:, step], minlength=action_count)
        probabilities = counts / counts.sum(axis=1, keepdims=True)
    rows.append(best)
    return max(rows)


def write_figures(records: pd.DataFrame, curves: pd.DataFrame) -> list[Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    planners = ["static", "equal_static", "cem", "risk_gated_cem"]
    colors = {
        "static": "#4C78A8",
        "equal_static": "#72B7B2",
        "cem": "#E45756",
        "risk_gated_cem": "#54A24B",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    for axis, (task, group) in zip(axes, records.groupby("task", sort=False)):
        means = group.groupby("planner")["regret"].mean().reindex(planners)
        axis.bar(np.arange(len(planners)), means.values, color=[colors[p] for p in planners])
        axis.set_title(task)
        axis.set_xticks(np.arange(len(planners)))
        axis.set_xticklabels([PLANNER_LABELS[p] for p in planners], rotation=28, ha="right", fontsize=7)
        axis.set_ylabel("regret vs card oracle")
        axis.grid(axis="y", alpha=0.25)
        if means.max() > 100:
            axis.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    path = RESULTS / "v4_gymnasium_cem_cards.pdf"
    fig.savefig(path)
    plt.close(fig)
    shutil.copy2(path, FIGURES / path.name)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    for axis, (task, group) in zip(axes, curves.groupby("task", sort=False)):
        for planner in ["cem", "risk_gated_cem"]:
            planner_rows = group[group["planner"] == planner]
            means = planner_rows.groupby("iteration")["best_true_return"].mean()
            axis.plot(means.index, means.values, marker="o", label=PLANNER_LABELS[planner], color=colors[planner])
        axis.set_title(task)
        axis.set_xlabel("CEM refit iteration")
        axis.set_ylabel("best true return")
        axis.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = RESULTS / "v4_gymnasium_cem_traces.pdf"
    fig.savefig(path)
    plt.close(fig)
    shutil.copy2(path, FIGURES / path.name)
    paths.append(path)
    return paths


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def write_macros(summary: dict[str, object]) -> None:
    text = ""
    text += macro("VFourGymBenchmarks", f"{summary['benchmark_count']}")
    text += macro("VFourGymSeeds", f"{summary['seeds_per_benchmark']}")
    text += macro("VFourGymRecordRows", f"{summary['record_rows']:,}")
    text += macro("VFourGymCurveRows", f"{summary['curve_rows']:,}")
    text += macro("VFourGymCEMWorseCount", f"{summary['cem_worse_than_best_static_count']}")
    text += macro("VFourGymRepairCount", f"{summary['repair_beats_cem_count']}")
    text += macro("VFourGymBoundaryCount", f"{summary['boundary_count']}")
    text += macro("VFourGymCliffCEMRegret", f"{summary['cliff_cem_regret']:.2f}")
    text += macro("VFourGymCliffRepairRegret", f"{summary['cliff_repair_regret']:.2f}")
    text += macro("VFourGymTaxiCEMRegret", f"{summary['taxi_cem_regret']:.2f}")
    text += macro("VFourGymTaxiRepairRegret", f"{summary['taxi_repair_regret']:.2f}")
    text += macro("VFourGymFrozenBoundaryRegret", f"{summary['frozen_cem_regret']:.3f}")
    (PAPER / "v4_gymnasium_macros.tex").write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3]
    planners = ["static", "equal_static", "cem", "risk_gated_cem"]
    records: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []

    for spec in TASKS:
        for seed in seeds:
            evaluator = TransitionEvaluator(spec, seed)
            oracle = estimate_oracle(evaluator, spec, seed)
            for planner in planners:
                row, planner_curves = run_planner(
                    evaluator,
                    spec,
                    seed=seed,
                    planner=planner,
                    population=96,
                    iterations=4,
                )
                row["oracle_true_return"] = oracle
                row["regret"] = float(oracle - float(row["selected_true_return"]))
                records.append(row)
                curves.extend(planner_curves)

    record_df = pd.DataFrame(records)
    curve_df = pd.DataFrame(curves)
    effects = (
        record_df.groupby(["task", "planner", "label"], as_index=False)[
            ["selected_true_return", "regret", "selected_tail_gap", "bad_event_probability", "proposal_drift"]
        ]
        .mean()
        .sort_values(["task", "regret"])
    )

    record_df.to_csv(RESULTS / "candidate_records.csv", index=False)
    curve_df.to_csv(RESULTS / "selection_curves.csv", index=False)
    effects.to_csv(RESULTS / "benchmark_effects.csv", index=False)
    figures = write_figures(record_df, curve_df)

    pivot = record_df.pivot_table(index=["task", "seed"], columns="planner", values="regret")
    task_means = record_df.groupby(["task", "planner"])["regret"].mean().unstack("planner")
    best_static = task_means[["static", "equal_static"]].min(axis=1)
    summary = {
        "benchmark_count": len(TASKS),
        "seeds_per_benchmark": len(seeds),
        "record_rows": int(len(record_df)),
        "curve_rows": int(len(curve_df)),
        "cem_worse_than_best_static_count": int((task_means["cem"] > best_static).sum()),
        "cem_worse_than_equal_static_count": int((task_means["cem"] > task_means["equal_static"]).sum()),
        "repair_beats_cem_count": int((task_means["risk_gated_cem"] < task_means["cem"]).sum()),
        "boundary_count": int((task_means["risk_gated_cem"] >= task_means["cem"]).sum()),
        "paired_cem_worse_than_best_static": int(
            (pivot["cem"] > pivot[["static", "equal_static"]].min(axis=1)).sum()
        ),
        "paired_cem_worse_than_equal_static": int((pivot["cem"] > pivot["equal_static"]).sum()),
        "paired_repair_beats_cem": int((pivot["risk_gated_cem"] < pivot["cem"]).sum()),
        "cliff_cem_regret": float(task_means.loc["CliffWalking-v1", "cem"]),
        "cliff_repair_regret": float(task_means.loc["CliffWalking-v1", "risk_gated_cem"]),
        "taxi_cem_regret": float(task_means.loc["Taxi-v3", "cem"]),
        "taxi_repair_regret": float(task_means.loc["Taxi-v3", "risk_gated_cem"]),
        "frozen_cem_regret": float(task_means.loc["FrozenLake-v1", "cem"]),
        "figures": [str(path.relative_to(ROOT)) for path in figures],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_macros(summary)
    print(f"v4 Gymnasium CEM cards complete: {RESULTS}")
    print(
        "benchmarks={benchmark_count} rows={record_rows} curves={curve_rows} "
        "cem_worse={cem_worse_than_best_static_count} repair={repair_beats_cem_count} "
        "boundary={boundary_count}".format(**summary)
    )


if __name__ == "__main__":
    main()
