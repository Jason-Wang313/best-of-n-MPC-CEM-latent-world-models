# Novelty Map

## Search Scope

I mapped the project against four close anchors:

- PlaNet learns latent dynamics from pixels and uses online planning in a compact latent space with CEM-style shooting: https://arxiv.org/abs/1811.04551
- Dreamer learns behaviors by latent imagination, using gradients through imagined trajectories rather than online CEM as the main optimizer: https://arxiv.org/abs/1912.01603
- PETS combines probabilistic ensembles, trajectory sampling, and MPC for uncertainty-aware model-based RL: https://arxiv.org/abs/1805.12114
- Hybrid CEM/gradient MPC improves action-sequence optimization efficiency by interleaving sampling and gradient steps: https://proceedings.mlr.press/v120/bharadhwaj20a.html

The search also checked the broader planner-overfitting idea: learned models can be exploited by optimizers, and uncertainty penalties are a known mitigation family. The repo's contribution is not "optimizers can exploit a model"; that would be incremental.

## Known

- Learned world models can support planning and policy improvement from imagined rollouts.
- CEM is a standard MPC optimizer for action sequences in learned dynamics, especially in PlaNet-style latent planning.
- Ensembles and trajectory sampling can reduce model-bias damage by representing epistemic uncertainty.
- Direct trajectory optimizers can overfit model errors, especially in sparse or out-of-distribution regions.
- Penalizing uncertainty, constraining search, or calibrating model predictions are established repair motifs.

## Incremental

- Showing that larger Best-of-N samples can select optimistic model errors.
- Adding an uncertainty penalty to CEM.
- Comparing random shooting, Best-of-N, and CEM on a toy learned-model failure.
- Plotting predicted return versus executed return.

Those are useful baselines, but they are not enough for a sharp paper.

## Genuinely New Candidate

The sharper thesis is:

> CEM with a learned model can be worse than a static one-shot proposal because elite refitting changes the proposal distribution toward regions selected by model error. The failure is not just extreme-value selection; it is adaptive concentration of future samples into model-error pockets.

The repo operationalizes this with:

- `elite_refit_drift`: movement of the proposal mean toward selected regions.
- `elite_model_error_mean`: model error among elites after each refit.
- `pocket_occupancy`: controlled diagnostic for known sparse-error pockets.
- `selected_tail_gap`: the gap between imagined and executed value of the selected tail.
- repair ablations that target the refit mechanism rather than only final selection.

## Reviewer Attack Surface

- The controlled world has an explicit known pocket. That is good for mechanism isolation but not benchmark validation.
- The learned ensemble is still a small synthetic model, not a high-dimensional latent world model.
- The theorem is an elite-conditioning identity and drift diagnostic, not a regret theorem.
- Repairs use proxies that are available in the toy world; stronger work should test whether analogous latent-space proxies work in PlaNet/Dreamer-like systems.
- Equal-budget Best-of-N can also fail when it samples the pocket, so the claim must be "CEM can amplify" rather than "CEM is dominated."

## Most Worth Pursuing

The publishable v1 direction is an audit-style paper:

1. Expose the elite-refit amplification mechanism in a setting where the true optimum and model-error pocket are known.
2. Prove a finite-pool elite-refit law for pocket-mass drift under explicit assumptions.
3. Measure whether CEM iterations increase elite model error and proposal drift.
4. Test repairs that directly interrupt refit amplification.
5. State the benchmark gap plainly.
