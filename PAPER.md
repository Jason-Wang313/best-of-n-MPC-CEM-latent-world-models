# CEM as an Adaptive Model-Error Amplifier in Learned World-Model MPC

## Abstract

Model-predictive control with learned world models commonly chooses action sequences by optimizing imagined returns. Random shooting and Best-of-N select from a fixed proposal, while the cross-entropy method (CEM) repeatedly selects elite imagined rollouts and refits its proposal distribution. This paper studies a narrow failure mode of that adaptation: when learned-model errors create optimistic sparse regions, CEM can make the optimizer's next samples more likely to revisit those regions. The result is not merely tail overfitting from a large sample; it is self-reinforcing proposal drift caused by elite refits.

We introduce a controlled one-dimensional planning benchmark with a known true optimum and a hidden model-error pocket, a sparse learned-ensemble variant, diagnostics for elite-refit drift and tail optimism, and repair baselines that target the refit mechanism. The current evidence supports a bounded mechanism claim: CEM can amplify learned-model optimism and can underperform static proposals in controlled settings. The repo does not claim high-dimensional benchmark transfer.

## 1. Introduction

Learned world models make planning cheap by replacing real interaction with imagined rollouts. This pattern appears in latent CEM planning, latent imagination, uncertainty-aware model-based reinforcement learning, and hybrid sampling-gradient MPC. The optimizer is often treated as a detachable search routine: if the model is good enough, better search should improve behavior.

This paper isolates a case where that intuition can fail. A static Best-of-N planner can certainly pick trajectories that exploit model error, but it samples once from a fixed proposal. CEM samples, scores, selects elites, refits the proposal to those elites, and repeats. If the elites are selected partly because of model error, the next proposal may move toward the error source. Optimism can become an attractor of the optimizer itself.

The central thesis is:

> CEM with a learned model can be worse than static Best-of-N because each elite refit changes the proposal toward regions selected by model error, not necessarily real utility.

The project is designed to expose, measure, repair, and audit this mechanism without overclaiming.

## 2. Related Work And Novelty Boundary

PlaNet learns compact latent dynamics and performs online planning with CEM-style shooting. Dreamer learns policies from latent imagination rather than relying primarily on online CEM. PETS uses probabilistic ensembles and trajectory sampling to reduce model-bias damage in MPC. Hybrid CEM/gradient MPC improves action-sequence optimization efficiency by combining sampling with gradient refinement.

These works establish the value of learned-model planning and uncertainty-aware MPC. They also make clear that model bias matters. The contribution here is narrower: CEM's repeated elite refits are treated as a mechanism that can amplify model-error pockets. The repo's diagnostics ask whether proposal drift, elite model error, pocket occupancy, and executed regret rise together.

This is not a claim that CEM is generally bad. It is also not a claim that uncertainty penalties are new. The paper's novelty is the adaptive-refit framing and the measurement protocol around that framing.

## 3. Mechanism

Let `a` denote an action sequence, `U(a)` the true return, and `S(a)` the learned-model score. Write the model error in return space as `E(a) = S(a) - U(a)`. A one-shot Best-of-N planner draws candidates from a fixed proposal and selects the best score. CEM draws candidates from proposal `q_t`, selects the top `rho` fraction by `S`, refits `q_{t+1}` to the elites, and repeats.

Suppose a sparse region of action-sequence space enters a model-error pocket. If trajectories in that pocket are overrepresented among model-scored elites, CEM's refit moves the next proposal toward the pocket. That movement can increase the chance of sampling even more pocket trajectories on the next iteration.

The repo measures this with four primary diagnostics:

- Elite-refit drift: movement of the CEM proposal mean from the initial Gaussian.
- Elite model-error concentration: mean `S(a) - U(a)` among elites.
- Pocket occupancy: how much elite mass enters the known sparse error pocket in the controlled world.
- Selected-tail gap: selected imagined return minus selected true return.

## 4. Theory

Consider an idealized CEM update over a binary pocket feature `P(a)`. Let `L_t(a)` be the event that `a` is elite under proposal `q_t`, and let the elite fraction be `rho = Pr(L_t=1)`.

Assume the next proposal's pocket mass is exactly refit to the elite conditional mass:

`q_{t+1}(P=1) = Pr_{a ~ q_t}(P(a)=1 | L_t(a)=1)`.

Then

`q_{t+1}(P=1) / q_t(P=1) = Pr(L_t=1 | P=1) / rho`.

Thus if `Pr(L_t=1 | P=1) > rho`, the next proposal places more mass on the pocket. For any moment-matched feature `phi`,

`E_{q_{t+1}}[phi] - E_{q_t}[phi] = Cov_{q_t}(phi, L_t) / rho`

under the exact conditional refit.

This proposition is deliberately modest. It is not a regret theorem, and diagonal Gaussian CEM only approximates the exact conditional update through elite moment matching. The result justifies the diagnostic: if a model-error pocket is elite-enriched, proposal drift toward that pocket is the expected behavior of the refit.

## 5. Experiments

### Controlled Pocket World

The main experiment uses a one-dimensional world with a true goal and a narrow harmful region. The learned model hallucinates positive reward in the harmful region. This makes the true optimum, model-error pocket, and model-versus-true return gap inspectable.

Compared planners:

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

The verified full CPU run in this checkout produced the following mean regrets against a strengthened true-oracle estimate:

| Planner | Mean regret |
|---|---:|
| CEM, pilot calibrated | 0.22 |
| CEM, disagreement veto | 0.23 |
| CEM, uncertainty pessimism | 0.30 |
| CEM, combined repair | 0.77 |
| CEM, conservative temperature | 2.57 |
| CEM, shadow realism | 5.17 |
| Best-of-N | 6.16 |
| Equal-budget Best-of-N | 7.99 |
| Random shooting | 9.64 |
| Vanilla CEM | 11.64 |
| CEM, diversity floor only | 13.31 |

The important result is not just that vanilla CEM had high regret. The traces also record whether elite model error and proposal drift increase during refits, which is the direct signature predicted by the mechanism.

### Sparse Learned Ensemble

The second experiment trains a small bootstrapped polynomial ensemble on transition data with sparse coverage of the harmful region. This checks whether the effect can appear when optimism comes from a learned sparse-region model rather than only from an analytic hallucinated bonus.

In the verified run, vanilla learned CEM had mean regret `4.64`, while calibrated learned CEM had `2.96`. Best-of-N had `3.19`, and uncertainty-penalized learned CEM had `3.35`. This supports the direction of the mechanism but remains a small synthetic test.

### Sweeps

The sweep varies CEM population, iteration count, and elite fraction. In the verified sweep, combined repair reduced mean CEM regret from `9.08` to `2.33`. This is evidence that interrupting uncertain/OOD elite concentration can help in the controlled setting.

## 6. Repairs

The repair methods are intentionally simple and auditable:

- Uncertainty pessimism subtracts ensemble disagreement from the score.
- Model-disagreement veto removes candidates above a disagreement quantile.
- Pilot-label calibration fits a small optimism penalty using an explicitly supplied pilot set.
- Conservative refit temperature keeps proposal variance from collapsing too fast.
- Elite diversity floor prevents all elites from being near-duplicates.
- Shadow-realism scoring penalizes trajectories that look out-of-support under simple support proxies.

Tests ensure that repaired planners do not call true evaluation labels during planning. Pilot calibration is the one method that uses real labels, and it uses only labels explicitly supplied by the pilot set.

## 7. Limitations

The main benchmark is controlled and synthetic. The sparse learned-ensemble experiment is still low-dimensional. The theorem is a diagnostic elite-refit identity, not a general regret guarantee. Equal-budget static Best-of-N can also fail when it samples the bad pocket. The current evidence should therefore be read as a mechanism scaffold, not a completed benchmark paper.

The next step is to port these diagnostics into a PlaNet/PETS-style continuous-control benchmark and test whether latent uncertainty, disagreement, and realism proxies produce the same repair pattern.

## 8. Conclusion

CEM's strength is adaptation, but adaptation can also amplify the wrong signal. In learned world-model MPC, elite imagined rollouts may be elite because they are truly useful, because they exploit model error, or both. This repo shows a controlled setting where refitting to those elites concentrates search into model-error pockets and damages real return. The repair results suggest that slowing, vetoing, or calibrating uncertain elite concentration is a promising direction, while the final claim remains bounded to controlled CPU evidence.

## References

- Hafner et al. PlaNet: Learning Latent Dynamics for Planning from Pixels. https://arxiv.org/abs/1811.04551
- Hafner et al. Dream to Control: Learning Behaviors by Latent Imagination. https://arxiv.org/abs/1912.01603
- Chua et al. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. https://arxiv.org/abs/1805.12114
- Bharadhwaj et al. Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization. https://proceedings.mlr.press/v120/bharadhwaj20a.html
