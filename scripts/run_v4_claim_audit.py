from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "OneDrive" / "Desktop"
FINAL_NAME = "best-of-n-MPC-CEM-latent-world-models-v4.pdf"
OLD_FINAL_NAME = "best-of-n-MPC-CEM-latent-world-models-v3.pdf"
REPO_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = DESKTOP / FINAL_NAME
OLD_REPO_PDF = ROOT / "paper" / "final" / OLD_FINAL_NAME
OLD_DESKTOP_PDF = DESKTOP / OLD_FINAL_NAME
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SUMMARY = ROOT / "results" / "v4_protocol_evidence" / "summary.json"
GYM_SUMMARY = ROOT / "results" / "v4_gymnasium_cem_cards" / "summary.json"
LOG = ROOT / "paper" / "iclr_submission" / "main.log"


EXPECTED_FILES = [
    "results/v4_protocol_evidence/controlled_planner_table.csv",
    "results/v4_protocol_evidence/learned_planner_table.csv",
    "results/v4_protocol_evidence/cem_trace_progression.csv",
    "results/v4_protocol_evidence/closed_loop_table.csv",
    "results/v4_protocol_evidence/sweep_summary.csv",
    "results/v4_protocol_evidence/repair_gain_summary.csv",
    "results/v4_protocol_evidence/controlled_rollout_audit.csv",
    "results/v4_protocol_evidence/learned_rollout_audit.csv",
    "results/v4_protocol_evidence/protocol_controlled_regret_rank.pdf",
    "results/v4_protocol_evidence/protocol_cem_trace_progression.pdf",
    "results/v4_protocol_evidence/protocol_sweep_repair_gain.pdf",
    "results/v4_protocol_evidence/protocol_learned_closed_loop.pdf",
    "results/v4_protocol_evidence/protocol_tail_gap_vs_regret.pdf",
    "results/v4_gymnasium_cem_cards/candidate_records.csv",
    "results/v4_gymnasium_cem_cards/selection_curves.csv",
    "results/v4_gymnasium_cem_cards/benchmark_effects.csv",
    "results/v4_gymnasium_cem_cards/summary.json",
    "results/v4_gymnasium_cem_cards/v4_gymnasium_cem_cards.pdf",
    "results/v4_gymnasium_cem_cards/v4_gymnasium_cem_traces.pdf",
    "paper/iclr_submission/figures/v4/v4_gymnasium_cem_cards.pdf",
    "paper/iclr_submission/figures/v4/v4_gymnasium_cem_traces.pdf",
    "paper/iclr_submission/v4_results_macros.tex",
    "paper/iclr_submission/v4_gymnasium_macros.tex",
]


STALE_TRACKED = {
    "PAPER.md",
    "PAPER.pdf",
    "paper/main.md",
    "paper/paper.md",
    "paper/claims_table.md",
    "paper/iclr_submission/main.pdf",
}


LOG_BLOCKERS = [
    re.compile(r"Citation .* undefined", re.IGNORECASE),
    re.compile(r"Reference .* undefined", re.IGNORECASE),
    re.compile(r"There were undefined", re.IGNORECASE),
    re.compile(r"Label\(s\) may have changed", re.IGNORECASE),
    re.compile(r"Rerun to get", re.IGNORECASE),
    re.compile(r"Fatal error", re.IGNORECASE),
    re.compile(r"Emergency stop", re.IGNORECASE),
    re.compile(r"LaTeX Error", re.IGNORECASE),
    re.compile(r"Overfull \\\\hbox", re.IGNORECASE),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_ls_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
    if not match:
        raise AssertionError(f"could not read page count from {path}")
    return int(match.group(1))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_summary(errors: list[str]) -> None:
    require(SUMMARY.exists(), f"missing {SUMMARY}", errors)
    if not SUMMARY.exists():
        return
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    exact = {
        "controlled_rows": 132,
        "trace_rows": 516,
        "sweep_rows": 360,
        "repair_gain_rows": 180,
        "learned_rows": 24,
    }
    for key, expected in exact.items():
        require(data.get(key) == expected, f"{key} expected {expected}, got {data.get(key)}", errors)
    thresholds = {
        "cem_regret": (10.0, None),
        "cem_minus_static_regret": (4.0, None),
        "best_repair_gain": (10.0, None),
        "sweep_gain": (5.0, None),
        "closed_loop_gain": (5.0, None),
        "tail_gap_regret_corr": (0.9, None),
    }
    for key, (low, high) in thresholds.items():
        value = data.get(key)
        require(isinstance(value, (int, float)), f"{key} missing numeric value", errors)
        if isinstance(value, (int, float)):
            require(value > low, f"{key} expected > {low}, got {value}", errors)
            if high is not None:
                require(value < high, f"{key} expected < {high}, got {value}", errors)
    require(data.get("best_repair_name") == "cem_pilot_calibrated", "unexpected best repair", errors)


def check_gym_summary(errors: list[str]) -> None:
    require(GYM_SUMMARY.exists(), f"missing {GYM_SUMMARY}", errors)
    if not GYM_SUMMARY.exists():
        return
    data = json.loads(GYM_SUMMARY.read_text(encoding="utf-8"))
    exact = {
        "benchmark_count": 3,
        "seeds_per_benchmark": 4,
        "record_rows": 48,
        "curve_rows": 96,
        "cem_worse_than_best_static_count": 2,
        "repair_beats_cem_count": 2,
        "boundary_count": 1,
    }
    for key, expected in exact.items():
        require(data.get(key) == expected, f"{key} expected {expected}, got {data.get(key)}", errors)
    require(float(data.get("cliff_cem_regret", 0.0)) > 1000.0, "CliffWalking stress card no longer exposes CEM failure", errors)
    require(float(data.get("cliff_repair_regret", 1.0)) <= 0.25, "CliffWalking repair gate regressed", errors)
    require(float(data.get("taxi_cem_regret", 0.0)) > 30.0, "Taxi stress card no longer exposes CEM failure", errors)
    require(float(data.get("taxi_repair_regret", 1.0)) <= 0.25, "Taxi repair gate regressed", errors)


def check_log(errors: list[str]) -> None:
    require(LOG.exists(), f"missing LaTeX log {LOG}", errors)
    if not LOG.exists():
        return
    text = LOG.read_text(encoding="utf-8", errors="replace")
    for pattern in LOG_BLOCKERS:
        match = pattern.search(text)
        if match:
            errors.append(f"LaTeX log blocker: {match.group(0)}")


def check_source_map(errors: list[str]) -> None:
    require(SOURCE_MAP.exists(), f"missing source map {SOURCE_MAP}", errors)
    if not SOURCE_MAP.exists():
        return
    text = SOURCE_MAP.read_text(encoding="utf-8")
    expected = (
        f"| `{FINAL_NAME}` | `C:\\Users\\wangz\\best-of-n-MPC-CEM-latent-world-models` | "
        "`Jason-Wang313/best-of-n-MPC-CEM-latent-world-models` |"
    )
    require(expected in text, "source map does not contain the v4 MPC-CEM row", errors)
    require("best-of-n-MPC-CEM-latent-world-models-v2.pdf" not in text, "source map still references v2 MPC-CEM PDF", errors)
    require("best-of-n-MPC-CEM-latent-world-models-v3.pdf" not in text, "source map still references v3 MPC-CEM PDF", errors)


def main() -> None:
    errors: list[str] = []

    for rel in EXPECTED_FILES:
        require((ROOT / rel).exists(), f"missing expected v4 artifact {rel}", errors)

    require(REPO_PDF.exists(), f"missing repo final PDF {REPO_PDF}", errors)
    require(DESKTOP_PDF.exists(), f"missing Desktop final PDF {DESKTOP_PDF}", errors)
    if REPO_PDF.exists():
        require(pdf_pages(REPO_PDF) >= 25, f"repo final PDF has fewer than 25 pages", errors)
    if REPO_PDF.exists() and DESKTOP_PDF.exists():
        require(sha256(REPO_PDF) == sha256(DESKTOP_PDF), "repo and Desktop PDFs have different SHA-256 hashes", errors)
    require(not OLD_REPO_PDF.exists(), f"old repo v3 PDF still exists: {OLD_REPO_PDF}", errors)
    require(not OLD_DESKTOP_PDF.exists(), f"old Desktop v3 PDF still exists: {OLD_DESKTOP_PDF}", errors)

    tracked = git_ls_files()
    for rel in STALE_TRACKED:
        require(rel not in tracked, f"stale draft artifact is still tracked: {rel}", errors)
    for rel in ["PAPER.md", "PAPER.pdf", "paper/main.md", "paper/paper.md", "paper/claims_table.md"]:
        require(not (ROOT / rel).exists(), f"stale draft artifact still exists: {rel}", errors)

    check_summary(errors)
    check_gym_summary(errors)
    check_log(errors)
    check_source_map(errors)

    claim_audit = subprocess.run(
        ["python", "scripts/run_claim_audit.py", "--repo-root", ".", "--claim-file", "docs/claims.json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    require(claim_audit.returncode == 0, claim_audit.stderr.strip() or claim_audit.stdout.strip(), errors)

    if errors:
        print("V4 claim audit failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("V4 claim audit passed.")


if __name__ == "__main__":
    main()
