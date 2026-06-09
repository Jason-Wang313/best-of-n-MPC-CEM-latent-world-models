# Final Audit

## 1. What Is The Discovered Main Thesis?

CEM is an adaptive model-error amplifier in learned-model MPC. Unlike static Best-of-N, CEM repeatedly refits its proposal to imagined elites. When elite status is partly caused by learned-model optimism, the next proposal can concentrate around that error pocket and intensify the mismatch between imagined and executed return.

## 2. What Is Genuinely New?

The new part is the elite-refit framing and diagnostic package: proposal drift, elite model-error concentration, pocket occupancy, selected-tail gap, and repair ablations aimed at interrupting refit amplification. The repo treats Best-of-N overfitting as the baseline and asks what adaptation adds.

## 3. What Theorem/Proof Survived Adversarial Checking?

The surviving proposition is a finite elite-refit law: if a pocket is overrepresented among model-scored elites, an exact elite-conditional refit increases next-iteration pocket mass by `Pr(elite | pocket) / rho`. The Gaussian CEM version is a diagnostic moment-drift analog, not a regret theorem.

## 4. What Is The Strongest Empirical Result?

The controlled toy experiment is the strongest result. In the verified full run, vanilla CEM had mean regret `11.64`, one-shot Best-of-N had `6.16`, equal-budget Best-of-N had `7.99`, and random shooting had `9.64`. The CEM traces record proposal drift, elite model error, and pocket occupancy per iteration, making the mechanism inspectable rather than only outcome-based.

## 5. What Is The Strongest Repair Result?

The strongest controlled repair result is the family of model-side veto/pessimism/calibration repairs. In the full run, pilot-calibrated CEM had mean regret `0.22`, disagreement-veto CEM had `0.23`, uncertainty-pessimistic CEM had `0.30`, and combined repaired CEM improved short closed-loop return from vanilla CEM's `-2.55` to `3.94`. In the sweep, combined repair reduced mean CEM regret from `9.08` to `2.33`.

## 6. What Are The Biggest Weaknesses?

- The main benchmark is controlled and synthetic.
- The sparse learned ensemble is still low-dimensional.
- The theory is diagnostic rather than a regret guarantee.
- Pilot calibration spends real labels and should be accounted for in any data-efficiency comparison.
- High-dimensional latent planners may need better uncertainty and realism proxies.

## 7. Paper-Worthy Status

This is a paper-worthy v1 mechanism scaffold, not a finished benchmark paper. The next validation step is to port the diagnostics into a PlaNet/PETS-style continuous-control benchmark and test whether latent-space uncertainty or realism proxies reproduce the controlled repair behavior.

The sparse learned-ensemble result is encouraging but still preliminary: vanilla learned CEM had regret `4.64`, while calibrated learned CEM had `2.96`.
