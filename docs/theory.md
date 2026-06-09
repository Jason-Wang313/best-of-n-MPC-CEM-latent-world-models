# Theory

## Setup

Let `a` be an action sequence drawn from proposal distribution `q_t(a)` at CEM iteration `t`.

- True utility: `U(a)`.
- Learned-model score: `S(a) = U(a) + E(a)`, where `E(a)` is model error in return space.
- Elite event: `L_t(a) = 1{S(a) >= tau_t}`, where `tau_t` is the top-`rho` threshold under the sampled or population proposal.
- Error pocket: `P(a) = 1` when the trajectory enters a sparse region where learned-model optimism is high.

CEM refits the next proposal to the elite sample. In the finite-sample implementation, this is a Gaussian moment match to elite action sequences.

## Proposition: Elite-Refit Pocket Amplification

Assume an idealized CEM update over a binary pocket feature `P` where the next proposal's pocket mass is refit exactly to the elite conditional mass:

`q_{t+1}(P=1) = Pr_{a ~ q_t}(P(a)=1 | L_t(a)=1)`.

If `Pr(L_t=1 | P=1) > Pr(L_t=1) = rho`, then

`q_{t+1}(P=1) > q_t(P=1)`,

and the multiplicative lift is

`q_{t+1}(P=1) / q_t(P=1) = Pr(L_t=1 | P=1) / rho`.

For any trajectory feature `phi(a)` that is moment-matched by the refit,

`E_{q_{t+1}}[phi] - E_{q_t}[phi] = Cov_{q_t}(phi, L_t) / rho`

in the ideal conditional-distribution update. In Gaussian CEM, the implemented mean update is the finite-sample analog with `phi(a)=a`.

## Interpretation

The result is small but useful. It says the CEM refit amplifies any feature that is overrepresented among model-scored elites. If a learned-model error pocket is more likely to pass the elite threshold than a random proposal sample, the next proposal places more mass there. This is different from one-shot Best-of-N, whose proposal distribution does not change after selecting a tail sample.

## What It Does Not Prove

- It does not prove that CEM has higher regret in every learned model.
- It does not prove that uncertainty penalties are sufficient in high-dimensional latent planners.
- It does not cover arbitrary parametric refits exactly; diagonal Gaussian CEM only moment-matches the elite sample.
- It does not remove the possibility that equal-budget static Best-of-N also finds the same bad pocket.

## Empirical Diagnostic

The proposition motivates three measurements:

- `proposal_drift`: action-mean movement after elite refits.
- `elite_model_error_mean`: whether elites are increasingly selected by model error.
- `pocket_occupancy`: whether the known error pocket is overrepresented among elites.

When these rise together while true return falls, the experiment has direct evidence for adaptive model-error amplification.
