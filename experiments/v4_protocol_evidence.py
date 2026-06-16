from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v4_protocol_evidence"
PAPER = ROOT / "paper" / "iclr_submission"
FIGURES = PAPER / "figures" / "protocol"


LABELS = {
    "cem": "vanilla CEM",
    "static_proposal": "static proposal",
    "static_equal_budget": "equal-budget static",
    "random_shooting": "random shooting",
    "cem_uncertainty": "uncertainty CEM",
    "cem_veto": "disagreement veto",
    "cem_pilot_calibrated": "pilot calibrated",
    "cem_conservative": "conservative CEM",
    "cem_shadow": "shadow realism",
    "cem_combined_repair": "combined repair",
    "cem_diversity": "diversity floor",
    "learned_cem": "learned CEM",
    "learned_cem_calibrated": "calibrated learned CEM",
    "learned_cem_uncertainty": "uncertainty learned CEM",
    "learned_static_proposal": "learned static",
}


def fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_inputs() -> dict[str, pd.DataFrame | dict]:
    return {
        "controlled_summary": load_json("results/controlled/controlled_summary.json"),
        "learned_summary": load_json("results/learned_ensemble/learned_summary.json"),
        "controlled_rollouts": pd.read_csv(ROOT / "results" / "controlled" / "controlled_rollouts.csv"),
        "controlled_traces": pd.read_csv(ROOT / "results" / "controlled" / "controlled_traces.csv"),
        "closed_loop": pd.read_csv(ROOT / "results" / "controlled" / "closed_loop.csv"),
        "learned_rollouts": pd.read_csv(ROOT / "results" / "learned_ensemble" / "learned_rollouts.csv"),
        "sweep": pd.read_csv(ROOT / "results" / "sweeps" / "cem_sweep.csv"),
        "repair_gain": pd.read_csv(ROOT / "results" / "sweeps" / "repair_gain.csv"),
    }


def write_tables(data: dict[str, pd.DataFrame | dict]) -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    controlled_summary = data["controlled_summary"]
    learned_summary = data["learned_summary"]
    controlled_rollouts = data["controlled_rollouts"]
    controlled_traces = data["controlled_traces"]
    closed_loop = data["closed_loop"]
    learned_rollouts = data["learned_rollouts"]
    sweep = data["sweep"]
    repair_gain = data["repair_gain"]

    assert isinstance(controlled_summary, dict)
    assert isinstance(learned_summary, dict)
    assert isinstance(controlled_rollouts, pd.DataFrame)
    assert isinstance(controlled_traces, pd.DataFrame)
    assert isinstance(closed_loop, pd.DataFrame)
    assert isinstance(learned_rollouts, pd.DataFrame)
    assert isinstance(sweep, pd.DataFrame)
    assert isinstance(repair_gain, pd.DataFrame)

    planner_rows = []
    for planner, regret in controlled_summary["mean_regret_by_planner"].items():
        planner_rows.append(
            {
                "planner": planner,
                "label": LABELS.get(planner, planner),
                "regret": float(regret),
                "true_return": float(controlled_summary["mean_true_return_by_planner"][planner]),
                "tail_gap": float(controlled_summary["mean_tail_gap_by_planner"][planner]),
            }
        )
    planner_table = pd.DataFrame(planner_rows).sort_values("regret")

    learned_rows = []
    for planner, regret in learned_summary["mean_regret_by_planner"].items():
        learned_rows.append(
            {
                "planner": planner,
                "label": LABELS.get(planner, planner),
                "regret": float(regret),
                "tail_gap": float(learned_summary["mean_tail_gap_by_planner"][planner]),
            }
        )
    learned_table = pd.DataFrame(learned_rows).sort_values("regret")

    cem_trace = (
        controlled_traces[controlled_traces["planner"] == "cem"]
        .groupby("iteration", as_index=False)[
            ["selected_tail_gap", "elite_model_error_mean", "elite_pocket_mean", "proposal_drift", "proposal_std_mean"]
        ]
        .mean()
    )

    closed_loop_table = (
        closed_loop.groupby("planner", as_index=False)["closed_loop_return"]
        .mean()
        .assign(label=lambda df: df["planner"].map(LABELS).fillna(df["planner"]))
        .sort_values("closed_loop_return", ascending=False)
    )

    sweep_summary = (
        sweep.groupby(["planner", "population", "iterations"], as_index=False)["regret"]
        .mean()
        .pivot_table(index=["population", "iterations"], columns="planner", values="regret")
        .reset_index()
    )
    sweep_summary["repair_gain"] = sweep_summary["cem"] - sweep_summary["cem_repaired"]

    repair_gain_summary = repair_gain.describe().T.reset_index().rename(columns={"index": "metric"})
    controlled_rollouts["label"] = controlled_rollouts["planner"].map(LABELS).fillna(controlled_rollouts["planner"])
    learned_rollouts["label"] = learned_rollouts["planner"].map(LABELS).fillna(learned_rollouts["planner"])

    tables = {
        "controlled_planner_table": planner_table,
        "learned_planner_table": learned_table,
        "cem_trace_progression": cem_trace,
        "closed_loop_table": closed_loop_table,
        "sweep_summary": sweep_summary,
        "repair_gain_summary": repair_gain_summary,
        "controlled_rollout_audit": controlled_rollouts,
        "learned_rollout_audit": learned_rollouts,
    }
    for name, table in tables.items():
        table.to_csv(RESULTS / f"{name}.csv", index=False)
    return tables


def write_figures(tables: dict[str, pd.DataFrame]) -> list[Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    planner_table = tables["controlled_planner_table"].sort_values("regret", ascending=False)
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    colors = ["#D95F02" if p == "cem" else "#1B9E77" if "cem_" in p else "#4C78A8" for p in planner_table["planner"]]
    axis.barh(planner_table["label"], planner_table["regret"], color=colors)
    axis.set_xlabel("controlled regret (lower is better)")
    axis.set_title("Controlled pocket-world planner ranking")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = RESULTS / "protocol_controlled_regret_rank.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    trace = tables["cem_trace_progression"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for axis, col, title in [
        (axes[0], "selected_tail_gap", "selected tail gap"),
        (axes[1], "elite_pocket_mean", "elite pocket mass"),
        (axes[2], "proposal_drift", "proposal drift"),
    ]:
        axis.plot(trace["iteration"], trace[col], marker="o", color="#D95F02")
        axis.set_title(title)
        axis.set_xlabel("CEM iteration")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean value")
    fig.tight_layout()
    path = RESULTS / "protocol_cem_trace_progression.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    sweep = tables["sweep_summary"]
    pivot = sweep.pivot_table(index="population", columns="iterations", values="repair_gain").sort_index()
    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    image = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    axis.set_xlabel("CEM iterations")
    axis.set_ylabel("population")
    axis.set_title("Combined repair regret reduction")
    fig.colorbar(image, ax=axis, label="CEM regret - repaired regret")
    fig.tight_layout()
    path = RESULTS / "protocol_sweep_repair_gain.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    learned = tables["learned_planner_table"].sort_values("regret", ascending=False)
    closed = tables["closed_loop_table"].sort_values("closed_loop_return")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    axes[0].barh(learned["label"], learned["regret"], color="#4C78A8")
    axes[0].set_xlabel("learned-ensemble regret")
    axes[0].set_title("Sparse learned ensemble")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(closed["label"], closed["closed_loop_return"], color="#1B9E77")
    axes[1].set_xlabel("closed-loop return")
    axes[1].set_title("Receding-horizon sanity check")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = RESULTS / "protocol_learned_closed_loop.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    rollouts = tables["controlled_rollout_audit"]
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for planner, group in rollouts.groupby("planner"):
        if planner in {"cem", "static_proposal", "cem_uncertainty", "cem_veto", "cem_pilot_calibrated", "cem_combined_repair"}:
            axis.scatter(group["selected_tail_gap"], group["regret"], label=LABELS.get(planner, planner), alpha=0.75)
    axis.set_xlabel("selected tail gap")
    axis.set_ylabel("controlled regret")
    axis.set_title("Selected optimism predicts regret")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = RESULTS / "protocol_tail_gap_vs_regret.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    for path in paths:
        shutil.copy2(path, FIGURES / path.name)
    return paths


def write_macros(data: dict[str, pd.DataFrame | dict], tables: dict[str, pd.DataFrame]) -> dict:
    controlled = data["controlled_summary"]
    learned = data["learned_summary"]
    closed = tables["closed_loop_table"].set_index("planner")
    sweep = tables["sweep_summary"]
    trace = tables["cem_trace_progression"]
    rollouts = tables["controlled_rollout_audit"]
    assert isinstance(controlled, dict)
    assert isinstance(learned, dict)

    cem_regret = float(controlled["mean_regret_by_planner"]["cem"])
    static_regret = float(controlled["mean_regret_by_planner"]["static_proposal"])
    equal_static_regret = float(controlled["mean_regret_by_planner"]["static_equal_budget"])
    best_repair_name = min(
        [name for name in controlled["mean_regret_by_planner"] if name.startswith("cem_")],
        key=lambda name: controlled["mean_regret_by_planner"][name],
    )
    best_repair_regret = float(controlled["mean_regret_by_planner"][best_repair_name])
    learned_cem = float(learned["mean_regret_by_planner"]["learned_cem"])
    learned_calibrated = float(learned["mean_regret_by_planner"]["learned_cem_calibrated"])
    sweep_cem = float(sweep["cem"].mean())
    sweep_repaired = float(sweep["cem_repaired"].mean())
    closed_cem = float(closed.loc["cem", "closed_loop_return"])
    closed_repair = float(closed.loc["cem_combined_repair", "closed_loop_return"])
    tail_gap_corr = float(rollouts["selected_tail_gap"].corr(rollouts["regret"]))

    summary = {
        "controlled_rows": int(len(rollouts)),
        "trace_rows": int(len(data["controlled_traces"])),
        "sweep_rows": int(len(data["sweep"])),
        "repair_gain_rows": int(len(data["repair_gain"])),
        "learned_rows": int(len(data["learned_rollouts"])),
        "cem_regret": cem_regret,
        "static_regret": static_regret,
        "equal_static_regret": equal_static_regret,
        "cem_minus_static_regret": cem_regret - static_regret,
        "cem_minus_equal_static_regret": cem_regret - equal_static_regret,
        "best_repair_name": best_repair_name,
        "best_repair_regret": best_repair_regret,
        "best_repair_gain": cem_regret - best_repair_regret,
        "cem_tail_gap": float(controlled["mean_tail_gap_by_planner"]["cem"]),
        "static_tail_gap": float(controlled["mean_tail_gap_by_planner"]["static_proposal"]),
        "learned_cem_regret": learned_cem,
        "learned_calibrated_regret": learned_calibrated,
        "learned_gain": learned_cem - learned_calibrated,
        "sweep_cem_regret": sweep_cem,
        "sweep_repaired_regret": sweep_repaired,
        "sweep_gain": sweep_cem - sweep_repaired,
        "closed_loop_cem": closed_cem,
        "closed_loop_repair": closed_repair,
        "closed_loop_gain": closed_repair - closed_cem,
        "trace_tail_gap_start": float(trace.loc[trace["iteration"] == trace["iteration"].min(), "selected_tail_gap"].iloc[0]),
        "trace_tail_gap_end": float(trace.loc[trace["iteration"] == trace["iteration"].max(), "selected_tail_gap"].iloc[0]),
        "trace_pocket_start": float(trace.loc[trace["iteration"] == trace["iteration"].min(), "elite_pocket_mean"].iloc[0]),
        "trace_pocket_end": float(trace.loc[trace["iteration"] == trace["iteration"].max(), "elite_pocket_mean"].iloc[0]),
        "tail_gap_regret_corr": tail_gap_corr,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    text = ""
    text += macro("VFourControlledRows", f"{summary['controlled_rows']:,}")
    text += macro("VFourTraceRows", f"{summary['trace_rows']:,}")
    text += macro("VFourSweepRows", f"{summary['sweep_rows']:,}")
    text += macro("VFourRepairGainRows", f"{summary['repair_gain_rows']:,}")
    text += macro("VFourLearnedRows", f"{summary['learned_rows']:,}")
    text += macro("VFourCEMRegret", fmt(cem_regret))
    text += macro("VFourStaticRegret", fmt(static_regret))
    text += macro("VFourEqualStaticRegret", fmt(equal_static_regret))
    text += macro("VFourCEMMinusStatic", fmt(cem_regret - static_regret))
    text += macro("VFourBestRepairRegret", fmt(best_repair_regret))
    text += macro("VFourBestRepairGain", fmt(cem_regret - best_repair_regret))
    text += macro("VFourCEMTailGap", fmt(summary["cem_tail_gap"]))
    text += macro("VFourStaticTailGap", fmt(summary["static_tail_gap"]))
    text += macro("VFourLearnedCEMRegret", fmt(learned_cem))
    text += macro("VFourLearnedCalibratedRegret", fmt(learned_calibrated))
    text += macro("VFourLearnedGain", fmt(learned_cem - learned_calibrated))
    text += macro("VFourSweepCEMRegret", fmt(sweep_cem))
    text += macro("VFourSweepRepairedRegret", fmt(sweep_repaired))
    text += macro("VFourSweepGain", fmt(sweep_cem - sweep_repaired))
    text += macro("VFourClosedLoopCEM", fmt(closed_cem))
    text += macro("VFourClosedLoopRepair", fmt(closed_repair))
    text += macro("VFourClosedLoopGain", fmt(closed_repair - closed_cem))
    text += macro("VFourTraceTailGapStart", fmt(summary["trace_tail_gap_start"]))
    text += macro("VFourTraceTailGapEnd", fmt(summary["trace_tail_gap_end"]))
    text += macro("VFourTracePocketStart", fmt(summary["trace_pocket_start"], 3))
    text += macro("VFourTracePocketEnd", fmt(summary["trace_pocket_end"], 3))
    text += macro("VFourTailGapRegretCorr", fmt(tail_gap_corr, 2))
    (PAPER / "v4_results_macros.tex").write_text(text, encoding="utf-8")
    return summary


def main() -> None:
    data = load_inputs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = write_tables(data)
    figures = write_figures(tables)
    summary = write_macros(data, tables)
    print(f"v4 cached evidence complete: {RESULTS}")
    print(f"controlled_rows={summary['controlled_rows']} sweep_rows={summary['sweep_rows']} figures={len(figures)}")


if __name__ == "__main__":
    main()
