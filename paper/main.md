# CEM as an Adaptive Model-Error Amplifier in Learned World-Model MPC

Canonical paper draft: `../PAPER.md`.

PDF copy: `../PAPER.pdf`.

## Abstract

Planning with learned world models often optimizes imagined rollouts using random shooting, static proposal, or CEM. This paper studies a specific failure mode: CEM can adaptively amplify model optimism because each elite refit changes the proposal distribution toward trajectories selected by the learned model. In controlled worlds with known sparse error pockets, we measure proposal drift, elite model-error concentration, tail optimism, and closed-loop regret. We compare uncertainty pessimism, diversity floors, disagreement vetoes, pilot calibration, conservative temperatures, and shadow-realism scoring. The current evidence supports a mechanism claim and motivates benchmark-scale validation.

## 1. Introduction

Learned world-model planners are attractive because they turn real interaction into cheap imagined search. PlaNet uses learned latent dynamics for online planning, Dreamer improves behavior through latent imagination, PETS adds ensemble uncertainty to MPC, and hybrid CEM/gradient methods improve action-sequence optimization efficiency.

This paper asks a narrower question: what does CEM's adaptive refit do when model error is structured?

## 2. Mechanism

static proposal samples once from a fixed proposal and selects the highest-scoring imagined trajectory. CEM samples, selects elites, refits the proposal, and repeats. If model error helps trajectories enter the elite set, the refit can move the next proposal toward the error source.

Diagnostics:

- Proposal drift from the initial action Gaussian.
- Elite model-error mean per CEM iteration.
- Pocket occupancy among elites in controlled worlds.
- Selected-tail gap between imagined and executed return.
- Open-loop and short receding-horizon regret.

## 3. Theory

The core proposition is an elite-conditional refit identity. For pocket indicator `P` and elite event `L`, exact refit gives:

`q_{t+1}(P=1) / q_t(P=1) = Pr(L=1 | P=1) / rho`.

When a model-error pocket is overrepresented among elites, the next proposal increases pocket mass. The result is diagnostic and finite-pool friendly; it is not a broad regret guarantee.

## 4. Experiments

### Controlled Pocket World

The true environment has a goal and a narrow harmful region. The learned model hallucinates positive reward in the harmful sparse region. This makes the true optimum and the error pocket inspectable.

### Sparse Learned Ensemble

A bootstrapped polynomial ensemble is trained on transitions that sparsely cover the harmful region. The experiment checks whether the same diagnostics remain useful when optimism is produced by learned sparse-region bias rather than only by an analytic model.

### Sweeps

We sweep population, CEM iterations, and elite fraction. The expected signature is that additional adaptive refits increase regret when they increase elite model error and proposal drift together.

## 5. Repairs

Repairs are model-side or pilot-label-side:

- Uncertainty pessimism.
- Elite diversity floor.
- Model-disagreement veto.
- Pilot-label calibration.
- Conservative refit temperature.
- Shadow-realism scoring.

The paper treats these as mechanism probes. Strong benchmark claims require stronger experiments.

## 6. Limitations

The current repo is a first autonomous pass. It is honest about what it does not show: high-dimensional latent control, image-based planning, and large benchmark transfer remain open validation work.
