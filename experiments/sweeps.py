from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cem_refit_audit.metrics import estimate_true_optimum, summarize_rows
from cem_refit_audit.planners import CEMPlanner, PlannerConfig
from cem_refit_audit.scoring import ModelScorer, RepairConfig
from cem_refit_audit.worlds import ToyWorld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--populations", nargs="+", type=int, default=[96, 192, 384])
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--elite-fractions", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--output-dir", type=Path, default=Path("results/sweeps"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/sweeps"))
    args = parser.parse_args()

    world = ToyWorld()
    optimum, _ = estimate_true_optimum(world, world.horizon, world.action_dim, samples=30_000, seed=404)
    rows = []
    for seed in args.seeds:
        for population in args.populations:
            for iterations in args.iterations:
                for elite_fraction in args.elite_fractions:
                    cfg = PlannerConfig(
                        horizon=world.horizon,
                        action_dim=world.action_dim,
                        action_low=world.action_bounds[0],
                        action_high=world.action_bounds[1],
                        population=population,
                        iterations=iterations,
                        elite_fraction=elite_fraction,
                        seed=seed,
                        init_std=0.82,
                    )
                    for name, repair in {
                        "cem": RepairConfig(name="none"),
                        "cem_repaired": RepairConfig(
                            uncertainty_penalty=0.8,
                            shadow_realism_penalty=0.6,
                            conservative_temperature=1.45,
                            elite_diversity_floor=0.4,
                            name="combined",
                        ),
                    }.items():
                        scorer = ModelScorer(world, repair=repair)
                        result = CEMPlanner(cfg, repair=repair, name=name).plan(
                            scorer,
                            utility_fn=lambda batch: world.true_returns(batch),
                        )
                        row = result.to_row()
                        row.update(
                            {
                                "seed": seed,
                                "planner": name,
                                "population": population,
                                "iterations": iterations,
                                "elite_fraction": elite_fraction,
                                "optimal_true_return": optimum,
                            }
                        )
                        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    df = summarize_rows(rows)
    df.to_csv(args.output_dir / "cem_sweep.csv", index=False)

    pivot = (
        df[df["planner"] == "cem"]
        .groupby(["population", "iterations"])["regret"]
        .mean()
        .unstack("iterations")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("CEM iterations")
    ax.set_ylabel("Population per iteration")
    ax.set_title("More adaptive refits can increase regret in the pocket world")
    fig.colorbar(im, ax=ax, label="Mean regret")
    fig.tight_layout()
    fig.savefig(args.figure_dir / "cem_budget_iteration_heatmap.png", dpi=180)
    plt.close(fig)

    repair_gain = (
        df.pivot_table(
            index=["seed", "population", "iterations", "elite_fraction"],
            columns="planner",
            values="regret",
        )
        .assign(repair_regret_reduction=lambda x: x["cem"] - x["cem_repaired"])
        .reset_index()
    )
    repair_gain.to_csv(args.output_dir / "repair_gain.csv", index=False)
    print(df.groupby("planner")["regret"].mean().to_string())


if __name__ == "__main__":
    main()
