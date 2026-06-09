from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boncem.learned_models import PolynomialEnsembleModel
from boncem.metrics import estimate_true_optimum, summarize_rows, trace_rows
from boncem.planners import BestOfNPlanner, CEMPlanner, PlannerConfig
from boncem.scoring import ModelScorer, PilotCalibrator, RepairConfig
from boncem.worlds import ToyWorld


def run_learned(seed: int, population: int, iterations: int, train_transitions: int) -> tuple[list[dict], list[dict]]:
    world = ToyWorld()
    learned = PolynomialEnsembleModel.fit(
        world,
        ensemble_size=7,
        n_transitions=train_transitions,
        seed=seed + 700,
    )
    optimum, _ = estimate_true_optimum(
        world,
        horizon=world.horizon,
        action_dim=world.action_dim,
        samples=30_000,
        seed=seed + 900,
    )
    cfg = PlannerConfig(
        horizon=world.horizon,
        action_dim=world.action_dim,
        action_low=world.action_bounds[0],
        action_high=world.action_bounds[1],
        population=population,
        iterations=iterations,
        seed=seed,
        init_std=0.82,
    )
    rng = np.random.default_rng(seed + 123)
    pilot = rng.uniform(world.action_bounds[0], world.action_bounds[1], size=(96, world.horizon, world.action_dim))
    calibrator = PilotCalibrator(ridge=1e-2).fit(learned.score_components(pilot), world.true_returns(pilot))
    planners = {
        "learned_best_of_n": (BestOfNPlanner(cfg), RepairConfig(name="none"), None),
        "learned_cem": (CEMPlanner(cfg, name="learned_cem"), RepairConfig(name="none"), None),
        "learned_cem_uncertainty": (
            CEMPlanner(
                cfg,
                repair=RepairConfig(uncertainty_penalty=0.9, conservative_temperature=1.3, name="uncertainty"),
                name="learned_cem_uncertainty",
            ),
            RepairConfig(uncertainty_penalty=0.9, conservative_temperature=1.3, name="uncertainty"),
            None,
        ),
        "learned_cem_calibrated": (
            CEMPlanner(
                cfg,
                repair=RepairConfig(uncertainty_penalty=0.45, shadow_realism_penalty=0.35, name="calibrated"),
                name="learned_cem_calibrated",
            ),
            RepairConfig(uncertainty_penalty=0.45, shadow_realism_penalty=0.35, name="calibrated"),
            calibrator,
        ),
    }
    rows = []
    traces = []
    for name, (planner, repair, maybe_calibrator) in planners.items():
        scorer = ModelScorer(learned, repair=repair, calibrator=maybe_calibrator)
        result = planner.plan(scorer, utility_fn=lambda batch: world.true_returns(batch))
        row = result.to_row()
        row.update({"seed": seed, "planner": name, "optimal_true_return": optimum})
        rows.append(row)
        traces.extend(trace_rows(seed, name, result.history))
    return rows, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--train-transitions", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("results/learned_ensemble"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/learned_ensemble"))
    args = parser.parse_args()

    rows: list[dict] = []
    traces: list[dict] = []
    for seed in args.seeds:
        seed_rows, seed_traces = run_learned(seed, args.population, args.iterations, args.train_transitions)
        rows.extend(seed_rows)
        traces.extend(seed_traces)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    df = summarize_rows(rows)
    trace_df = pd.DataFrame(traces)
    df.to_csv(args.output_dir / "learned_rollouts.csv", index=False)
    trace_df.to_csv(args.output_dir / "learned_traces.csv", index=False)
    summary = {
        "seeds": args.seeds,
        "mean_regret_by_planner": df.groupby("planner")["regret"].mean().sort_values().to_dict(),
        "mean_tail_gap_by_planner": df.groupby("planner")["selected_tail_gap"].mean().sort_values().to_dict(),
    }
    (args.output_dir / "learned_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    means = df.groupby("planner")["regret"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(np.arange(len(means)), means.values, color=["#4c78a8", "#e45756", "#54a24b", "#72b7b2"])
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(means.index, rotation=25, ha="right")
    ax.set_ylabel("Regret vs true oracle estimate")
    ax.set_title("Learned sparse ensemble reproduces model-optimism amplification")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "learned_regret.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
