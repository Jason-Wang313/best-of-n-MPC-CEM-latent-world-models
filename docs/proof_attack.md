# Adversarial Proof Check

## Claim Under Attack

The theorem candidate was initially too strong: "CEM worsens learned-model exploitation relative to static proposal." That does not survive.

The surviving claim is narrower:

> Under an exact elite-conditional refit for a pocket indicator, pocket mass increases when the pocket is overrepresented among model-scored elites. Gaussian CEM inherits this as a finite-sample moment drift diagnostic, not as a universal regret theorem.

## Assumption Attacks

- Exact refit: real CEM uses a diagonal Gaussian over continuous action sequences. The binary pocket identity is exact only for a distribution family that can represent the pocket indicator directly.
- Elite threshold: finite samples make the threshold random. The identity is population-level; finite-sample traces estimate it.
- Score decomposition: `S = U + E` is an analysis device. In practice `U` is hidden and used only for evaluation.
- Persistent overrepresentation: repeated amplification requires the pocket to remain elite-enriched after the proposal moves.

## Counterexamples

- If the true utility advantage outside the pocket dominates the model error, `Pr(L | P)` need not exceed `rho`; no pocket amplification follows.
- If the pocket has high model uncertainty and the planner applies a strong penalty, the elite condition can underweight the pocket.
- If an equal-budget static proposal sample contains a fully optimized pocket trajectory, it can fail in the same way as CEM.
- If CEM initializes far from the pocket and never samples partial pocket trajectories, refit amplification is not triggered.

## Numerical Edge Checks

The tests and smoke experiment check the following:

- CEM updates remain deterministic for a fixed seed.
- Planner selection uses score, while true utility is only a diagnostic.
- Repairs can run without calling true labels during planning.
- Smoke runs generate traces so pocket occupancy and elite model error can be inspected per iteration.

## Weakened Final Statement

The paper should claim:

> CEM can adaptively amplify learned-model optimism because each elite refit conditions the next proposal on model-scored elites. In controlled worlds, this can produce worse real returns than non-adaptive proposals, and model-only repairs that reduce uncertain/OOD elite concentration reduce the failure.

It should not claim benchmark-general dominance, broad latent-world-model validation, or a complete solution.
