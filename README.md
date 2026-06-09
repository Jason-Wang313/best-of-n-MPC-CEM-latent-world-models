# Best-of-N MPC/CEM with Learned World Models

This repo studies a narrow failure mode in planning with learned world models:

> CEM is an adaptive model-error amplifier. A one-shot Best-of-N planner can exploit model optimism, but CEM repeatedly refits its proposal distribution to elite imagined rollouts. If those elites are selected partly because of model error, the next proposal can place more mass on the error pocket and make the optimism self-reinforcing.

The first-pass contribution is deliberately modest: a controlled CPU benchmark, a sparse learned-ensemble variant, diagnostics for elite-refit drift and tail optimism, repair baselines, and a claim audit that blocks overclaiming.

## Why This Angle

The novelty map positions this against PlaNet latent CEM planning, Dreamer latent imagination, PETS uncertainty-aware MPC, and hybrid CEM/gradient MPC:

- PlaNet: https://arxiv.org/abs/1811.04551
- Dreamer: https://arxiv.org/abs/1912.01603
- PETS: https://arxiv.org/abs/1805.12114
- CEM plus gradient MPC: https://proceedings.mlr.press/v120/bharadhwaj20a.html

Those works establish strong learned-model planning systems and uncertainty-aware MPC. This repo focuses on the mechanism by which CEM's elite refits can amplify learned-model optimism across planning iterations.

## Quickstart

```powershell
cd "C:\Users\wangz\Downloads\best of n mpc cem learned world models"
python -m pip install -e . pytest
pytest
.\scripts\run_smoke.ps1
```

For the fuller local evidence package:

```powershell
.\scripts\run_all.ps1
```

Outputs are written to `results/` and `figures/`. The smoke run is intentionally fast and should create at least:

- `results/smoke/controlled_summary.json`
- `results/smoke/controlled_rollouts.csv`
- `figures/smoke/regret_bars.png`
- `figures/smoke/elite_refit_drift.png`

## Paper Draft

The paper draft is intentionally visible at the repo root:

- `PAPER.md`
- `PAPER.pdf`

There is also a pointer in `paper/paper.md`.

## Repo Layout

- `src/boncem/`: toy worlds, sparse learned ensemble, planners, scoring repairs, metrics.
- `experiments/`: controlled world, learned ensemble, and CEM sweep entrypoints.
- `scripts/`: smoke/full runs and claim audit.
- `tests/`: deterministic planner, score/utility separation, metric, and label-leakage checks.
- `docs/`: novelty map, theory, adversarial proof check, claim registry, final audit.
- `paper/`: paper skeleton and claim table.

## Implemented Planners

- Random shooting.
- One-shot Best-of-N.
- Equal-budget one-shot Best-of-N.
- Vanilla CEM.
- CEM with uncertainty pessimism.
- CEM with elite diversity floor.
- CEM with model-disagreement veto.
- CEM with pilot-label calibration.
- CEM with conservative refit temperature.
- CEM with shadow-realism scoring.
- Combined repaired CEM.

## Diagnostics

- Elite-refit drift: proposal mean movement from the initial Gaussian.
- Tail optimism: selected model return minus true return.
- Elite model-error concentration: model error among CEM elites per iteration.
- Pocket occupancy: how much elite mass enters the known model-error pocket.
- Closed-loop return: short receding-horizon MPC sanity check.

## Claim Boundary

The repo supports controlled-mechanism claims only. It does not claim MuJoCo-scale or Dreamer-scale validation. See `docs/final_audit.md` and `docs/claims.json` for the current evidence status.

## Latest Verified Local Run

On the full CPU run in this checkout, vanilla CEM had mean controlled regret `11.64`, one-shot Best-of-N had `6.16`, and the best repaired CEM variants were near `0.22-0.30`. In the sparse learned-ensemble experiment, vanilla CEM regret was `4.64` versus `2.96` for calibrated CEM. In the sweep, combined repair reduced mean CEM regret from `9.08` to `2.33`.
