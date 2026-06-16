# Elite-Refit Audits for Learned World-Model MPC

This repo studies a narrow failure mode in planning with learned world models:

> CEM is an adaptive model-error amplifier. A static-proposal planner can exploit
> model optimism once, but CEM repeatedly refits its proposal distribution to
> elite imagined rollouts. If those elites are selected partly because of model
> error, the next proposal can place more mass on the error pocket and make
> optimism self-reinforcing.

The contribution is deliberately modest: a controlled CPU benchmark, a sparse
learned-ensemble variant, diagnostics for elite-refit drift and tail optimism,
repair baselines, and a claim audit that blocks overclaiming.

## Why This Angle

The novelty map positions this against PlaNet latent CEM planning, Dreamer
latent imagination, PETS uncertainty-aware MPC, and hybrid CEM/gradient MPC:

- PlaNet: https://arxiv.org/abs/1811.04551
- Dreamer: https://arxiv.org/abs/1912.01603
- PETS: https://arxiv.org/abs/1805.12114
- CEM plus gradient MPC: https://proceedings.mlr.press/v120/bharadhwaj20a.html

Those works establish strong learned-model planning systems and
uncertainty-aware MPC. This repo focuses on the mechanism by which CEM's elite
refits can amplify learned-model optimism across planning iterations.

## Quickstart

```powershell
cd "C:\Users\wangz\best-of-n-MPC-CEM-latent-world-models"
python -m pip install -e . pytest
pytest
.\scripts\run_smoke.ps1
```

For the fuller local evidence package:

```powershell
.\scripts\run_all.ps1
```

Outputs are written to `results/` and `figures/`. The smoke run is intentionally
fast and should create at least:

- `results/smoke/controlled_summary.json`
- `results/smoke/controlled_rollouts.csv`
- `figures/smoke/regret_bars.png`
- `figures/smoke/elite_refit_drift.png`

## V4 Paper Artifact

The ICLR-style paper source lives at `paper/iclr_submission/main.tex`. The
submission-ready v4 artifact is versioned as:

- `paper/final/best-of-n-MPC-CEM-latent-world-models-v4.pdf`
- `C:\Users\wangz\OneDrive\Desktop\best-of-n-MPC-CEM-latent-world-models-v4.pdf`

Regenerate the v4 evidence layer and both PDF copies with:

```powershell
python scripts\build_v4_paper.py
python scripts\run_v4_claim_audit.py
```

The unversioned LaTeX output `paper/iclr_submission/main.pdf` is a local build
product, not the final artifact.

## Repo Layout

- `src/cem_refit_audit/`: toy worlds, sparse learned ensemble, planners,
  scoring repairs, and metrics.
- `experiments/`: controlled world, learned ensemble, CEM sweep, and v4
  Gymnasium transition-card entrypoints.
- `scripts/`: smoke/full runs, v4 paper build, and claim audits.
- `tests/`: deterministic planner, score/utility separation, metric, and
  label-leakage checks.
- `docs/`: novelty map, theory, adversarial proof check, claim registry, final
  audit.
- `paper/`: paper skeleton, ICLR submission package, and claim table.

## Implemented Planners

- Random shooting.
- Static-proposal baseline.
- Equal-budget static-proposal baseline.
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

The repo supports controlled-mechanism claims and scoped Gymnasium
transition-card stress claims only. It does not claim MuJoCo-scale,
D4RL-scale, or Dreamer-scale validation. See `docs/final_audit.md` and
`docs/claims.json` for the current evidence status.

## Latest Verified V4 Run

The v4 cached evidence layer reads 132 controlled rows, 516 CEM trace rows, 360
sweep rows, 180 paired repair-gain rows, and 24 learned-ensemble rows. Vanilla
CEM had mean controlled regret `11.64`, the static-proposal baseline had
`6.16`, and the best repaired CEM variant had regret `0.22`. In the sparse
learned-ensemble bridge, vanilla CEM regret was `4.64` versus `2.96` for
calibrated CEM. In the sweep, combined repair reduced mean CEM regret from
`9.08` to `2.33`; in the short closed-loop check, return improved from `-2.55`
to `3.94`.

The v4 Gymnasium transition-card layer adds 3 standard exact-transition
stress cards over 4 seeds each. Vanilla categorical CEM is worse than the best
non-adaptive static baseline on 2 cards, the risk-gated repair improves 2
cards, and FrozenLake is reported as the boundary card where vanilla CEM is
already competitive.
