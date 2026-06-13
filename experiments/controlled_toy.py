from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cem_refit_audit.metrics import estimate_true_optimum, summarize_rows, trace_rows
from cem_refit_audit.planners import StaticProposalPlanner, CEMPlanner, PlannerConfig, RandomShootingPlanner
from cem_refit_audit.scoring import ModelScorer, PilotCalibrator, RepairConfig
from cem_refit_audit.worlds import ToyWorld, ToyWorldConfig


def _seed_list(values: list[str]) -> list[int]:
    seeds: list[int] = []
    for value in values:
        if ".." in value:
            start, end = value.split("..", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(value))
    return seeds


def make_repairs() -> dict[str, tuple[RepairConfig, bool]]:
    return {
        "cem": (RepairConfig(name="none"), False),
        "cem_uncertainty": (RepairConfig(uncertainty_penalty=1.4, name="uncertainty_pessimism"), False),
        "cem_diversity": (RepairConfig(elite_diversity_floor=0.75, name="elite_diversity_floor"), False),
        "cem_veto": (RepairConfig(disagreement_veto_quantile=0.82, name="model_disagreement_veto"), False),
        "cem_shadow": (RepairConfig(shadow_realism_penalty=1.1, name="shadow_realism"), False),
        "cem_conservative": (RepairConfig(conservative_temperature=1.8, name="conservative_refit_temperature"), False),
        "cem_pilot_calibrated": (RepairConfig(name="pilot_label_calibration"), True),
        "cem_combined_repair": (
            RepairConfig(
                uncertainty_penalty=0.8,
                shadow_realism_penalty=0.7,
                disagreement_veto_quantile=0.9,
                conservative_temperature=1.45,
                elite_diversity_floor=0.45,
                name="combined",
            ),
            True,
        ),
    }


def fit_calibrator(world: ToyWorld, seed: int, n: int, horizon: int, action_dim: int) -> PilotCalibrator:
    rng = np.random.default_rng(seed + 55_001)
    low, high = world.action_bounds
    pilot = rng.uniform(low, high, size=(n, horizon, action_dim))
    components = world.score_components(pilot)
    true_returns = world.true_returns(pilot)
    return PilotCalibrator(ridge=1e-2).fit(components, true_returns)


def run_once(
    *,
    seed: int,
    population: int,
    iterations: int,
    optimum_samples: int,
    calibration_samples: int,
    closed_loop_steps: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    world = ToyWorld(ToyWorldConfig())
    optimum, optimum_actions = estimate_true_optimum(
        world,
        horizon=world.horizon,
        action_dim=world.action_dim,
        samples=optimum_samples,
        seed=99_000 + seed,
    )
    rows: list[dict] = []
    traces: list[dict] = []
    closed_loop_rows: list[dict] = []
    score = ModelScorer(world)
    utility_fn = lambda batch: world.true_returns(batch)

    base_cfg = PlannerConfig(
        horizon=world.horizon,
        action_dim=world.action_dim,
        action_low=world.action_bounds[0],
        action_high=world.action_bounds[1],
        population=population,
        iterations=iterations,
        seed=seed,
        elite_fraction=0.1,
        init_std=0.82,
        min_std=0.035,
    )
    planners = {
        "random_shooting": RandomShootingPlanner(base_cfg),
        "static_proposal": StaticProposalPlanner(base_cfg),
        "static_equal_budget": StaticProposalPlanner(
            PlannerConfig(**{**base_cfg.__dict__, "population": population * iterations})
        ),
    }
    for name, planner in planners.items():
        result = planner.plan(score, utility_fn=utility_fn)
        row = result.to_row()
        row.update(
            {
                "seed": seed,
                "planner": name,
                "optimal_true_return": optimum,
                "optimal_first_action": float(optimum_actions[0, 0]),
            }
        )
        rows.append(row)
        traces.extend(trace_rows(seed, name, result.history))

    calibrator = fit_calibrator(world, seed, calibration_samples, world.horizon, world.action_dim)
    for name, (repair, needs_calibration) in make_repairs().items():
        scorer = ModelScorer(world, repair=repair, calibrator=calibrator if needs_calibration else None)
        planner = CEMPlanner(base_cfg, repair=repair, name=name)
        result = planner.plan(scorer, utility_fn=utility_fn)
        row = result.to_row()
        row.update(
            {
                "seed": seed,
                "planner": name,
                "optimal_true_return": optimum,
                "optimal_first_action": float(optimum_actions[0, 0]),
            }
        )
        rows.append(row)
        traces.extend(trace_rows(seed, name, result.history))

    if closed_loop_steps > 0:
        subset = {
            "static_proposal": (StaticProposalPlanner, RepairConfig(name="none"), False),
            "cem": (CEMPlanner, RepairConfig(name="none"), False),
            "cem_combined_repair": (CEMPlanner, make_repairs()["cem_combined_repair"][0], True),
        }
        for name, (planner_cls, repair, needs_calibration) in subset.items():
            x = world.config.init_state
            total = 0.0
            for t in range(closed_loop_steps):
                cfg = PlannerConfig(**{**base_cfg.__dict__, "seed": seed + 2000 * t})
                scorer = ModelScorer(world, repair=repair, calibrator=calibrator if needs_calibration else None)
                if planner_cls is CEMPlanner:
                    planner = planner_cls(cfg, repair=repair, name=name)
                else:
                    planner = planner_cls(cfg)
                result = planner.plan(
                    lambda batch, scorer=scorer, x=x: scorer(batch, x),
                    utility_fn=lambda batch, x=x: world.true_returns(batch, x0=x),
                )
                x, reward = world.step_true(x, result.actions[0])
                total += reward
            closed_loop_rows.append(
                {
                    "seed": seed,
                    "planner": name,
                    "closed_loop_steps": closed_loop_steps,
                    "closed_loop_return": float(total),
                }
            )

    return rows, traces, closed_loop_rows


def plot_results(df: pd.DataFrame, traces: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [
        "random_shooting",
        "static_proposal",
        "static_equal_budget",
        "cem",
        "cem_uncertainty",
        "cem_veto",
        "cem_shadow",
        "cem_conservative",
        "cem_pilot_calibrated",
        "cem_combined_repair",
    ]
    plot_df = df[df["planner"].isin(order)].copy()
    means = plot_df.groupby("planner", sort=False)["regret"].mean().reindex(order).dropna()
    sems = plot_df.groupby("planner", sort=False)["regret"].sem().reindex(means.index).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#4c78a8" if "cem" not in idx else "#e45756" for idx in means.index]
    colors = ["#54a24b" if "repair" in idx or "uncertainty" in idx or "shadow" in idx or "veto" in idx else c for idx, c in zip(means.index, colors)]
    ax.bar(np.arange(len(means)), means.values, yerr=sems.values, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_ylabel("Open-loop regret vs true oracle estimate")
    ax.set_title("CEM refits can amplify model-error pockets; repairs reduce regret")
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(means.index, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "regret_bars.png", dpi=180)
    plt.close(fig)

    cem_trace = traces[traces["planner"].str.startswith("cem")].copy()
    if not cem_trace.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for planner, group in cem_trace.groupby("planner"):
            means_trace = group.groupby("iteration")[["proposal_drift", "elite_model_error_mean"]].mean()
            axes[0].plot(means_trace.index, means_trace["proposal_drift"], marker="o", label=planner)
            axes[1].plot(means_trace.index, means_trace["elite_model_error_mean"], marker="o", label=planner)
        axes[0].set_ylabel("Proposal drift from initial Gaussian")
        axes[1].set_ylabel("Mean elite model error")
        for ax in axes:
            ax.set_xlabel("CEM iteration")
            ax.grid(alpha=0.25)
        axes[1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(figure_dir / "elite_refit_drift.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(df["selected_tail_gap"], df["selected_true_return"], c=df["regret"], cmap="viridis_r", s=48)
    ax.set_xlabel("Selected-tail gap: model return - true return")
    ax.set_ylabel("Executed true return")
    ax.set_title("Tail optimism concentrates in low-utility selections")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "tail_gap_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--seeds", nargs="+", default=["0..9"])
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--optimum-samples", type=int, default=50_000)
    parser.add_argument("--calibration-samples", type=int, default=96)
    parser.add_argument("--closed-loop-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/controlled"))
    args = parser.parse_args()

    if args.mode == "smoke":
        args.population = min(args.population, 96)
        args.iterations = min(args.iterations, 3)
        args.optimum_samples = min(args.optimum_samples, 8_000)
        args.closed_loop_steps = min(args.closed_loop_steps, 2)

    rows: list[dict] = []
    traces: list[dict] = []
    closed_loop_rows: list[dict] = []
    for seed in _seed_list(args.seeds):
        seed_rows, seed_traces, seed_closed = run_once(
            seed=seed,
            population=args.population,
            iterations=args.iterations,
            optimum_samples=args.optimum_samples,
            calibration_samples=args.calibration_samples,
            closed_loop_steps=args.closed_loop_steps,
        )
        rows.extend(seed_rows)
        traces.extend(seed_traces)
        closed_loop_rows.extend(seed_closed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = summarize_rows(rows)
    trace_df = pd.DataFrame(traces)
    closed_df = pd.DataFrame(closed_loop_rows)
    df.to_csv(args.output_dir / "controlled_rollouts.csv", index=False)
    trace_df.to_csv(args.output_dir / "controlled_traces.csv", index=False)
    if not closed_df.empty:
        closed_df.to_csv(args.output_dir / "closed_loop.csv", index=False)
    summary = {
        "mode": args.mode,
        "seeds": _seed_list(args.seeds),
        "population": args.population,
        "iterations": args.iterations,
        "mean_regret_by_planner": df.groupby("planner")["regret"].mean().sort_values().to_dict(),
        "mean_tail_gap_by_planner": df.groupby("planner")["selected_tail_gap"].mean().sort_values().to_dict(),
        "mean_true_return_by_planner": df.groupby("planner")["selected_true_return"].mean().sort_values(ascending=False).to_dict(),
    }
    if not closed_df.empty:
        summary["closed_loop_return_by_planner"] = (
            closed_df.groupby("planner")["closed_loop_return"].mean().sort_values(ascending=False).to_dict()
        )
    (args.output_dir / "controlled_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(df, trace_df, args.figure_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
