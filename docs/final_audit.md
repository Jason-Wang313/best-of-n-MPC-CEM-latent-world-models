# Final Audit

## 1. What Is The Discovered Main Thesis?

CEM is an adaptive model-error amplifier in learned-model MPC. Unlike a static-proposal baseline, CEM repeatedly refits its proposal to imagined elites. When elite status is partly caused by learned-model optimism, the next proposal can concentrate around that error pocket and intensify the mismatch between imagined and executed return.

## 2. What Is Genuinely New?

The new part is the elite-refit framing and diagnostic package: proposal drift, elite model-error concentration, pocket occupancy, selected-tail gap, and repair ablations aimed at interrupting refit amplification. The repo treats static-proposal overfitting as the baseline and asks what adaptation adds.

## 3. What Theorem/Proof Survived Adversarial Checking?

The surviving proposition is a finite elite-refit law: if a pocket is overrepresented among model-scored elites, an exact elite-conditional refit increases next-iteration pocket mass by `Pr(elite | pocket) / rho`. The Gaussian CEM version is a diagnostic moment-drift analog, not a regret theorem.

## 4. What Is The Strongest Empirical Result?

The controlled toy experiment is the strongest result. In the verified full run, vanilla CEM had mean regret `11.64`, the static-proposal baseline had `6.16`, the equal-budget static baseline had `7.99`, and random shooting had `9.64`. The CEM traces record proposal drift, elite model error, and pocket occupancy per iteration, making the mechanism inspectable rather than only outcome-based.

## 5. What Is The Strongest Repair Result?

The strongest controlled repair result is the family of model-side veto/pessimism/calibration repairs. In the full run, pilot-calibrated CEM had mean regret `0.22`, disagreement-veto CEM had `0.23`, uncertainty-pessimistic CEM had `0.30`, and combined repaired CEM improved short closed-loop return from vanilla CEM's `-2.55` to `3.94`. In the sweep, combined repair reduced mean CEM regret from `9.08` to `2.33`.

## 6. What Are The Biggest Weaknesses?

- The main benchmark is controlled and synthetic.
- The sparse learned ensemble is still low-dimensional.
- The Gymnasium transition cards are recognized benchmark interfaces but use
  deliberately misspecified model rewards; they are stress cards, not
  leaderboard-scale validation.
- The theory is diagnostic rather than a regret guarantee.
- Pilot calibration spends real labels and should be accounted for in any data-efficiency comparison.
- High-dimensional latent planners may need better uncertainty and realism proxies.

## 7. V4 Paper Status

This is a submission-ready v4 bounded mechanism paper, not a broad benchmark paper. The v4 manuscript writes a versioned final PDF to `paper/final/best-of-n-MPC-CEM-latent-world-models-v4.pdf`, and mirrors that PDF to the visible Desktop. The v4 protocol evidence layer adds planner-ranking tables, trace progression, sweep repair-gain summaries, learned-ensemble ledgers, closed-loop summaries, five paper-facing figures, and LaTeX macros that keep the manuscript synchronized with repository artifacts.

The v4 benchmark upgrade adds exact Gymnasium transition-table CEM cards for FrozenLake, CliffWalking, and Taxi. Vanilla categorical CEM is worse than the best non-adaptive static baseline on 2 of 3 cards; risk-gated CEM improves 2 of 3; FrozenLake is deliberately kept as the boundary card where the repair is unnecessary or conservative.

The sparse learned-ensemble result remains a bridge rather than benchmark-scale transfer: vanilla learned CEM had regret `4.64`, while calibrated learned CEM had `2.96`. The next validation step is to port the diagnostics into a PlaNet/PETS-style continuous-control benchmark and test whether latent-space uncertainty or realism proxies reproduce the controlled repair behavior.

## 8. V4 Acceptance Gates

- Final PDF is versioned and at least 25 pages.
- Desktop PDF and repository PDF must have identical SHA-256 hashes.
- `results/v4_protocol_evidence/summary.json` must match the manuscript macros.
- `results/v4_gymnasium_cem_cards/summary.json` must report 3 cards, 48
  records, 96 curve rows, 2 CEM-worse cards, 2 repair-improved cards, and 1
  boundary card.
- `docs/claims.json` must cite v4 evidence for supported or partial claims.
- Unsupported benchmark-scale transfer must remain unsupported.
- Old root draft artifacts must not be treated as the final paper.
- The source map must point to the v4 Desktop PDF, this local source folder, and the GitHub repository.
- Final PDF SHA256:
  `4AAF7DF7B94479B8F1F64980FC5CC187007F867F1D0915761F84A27FA1A01676`
- Visual QA inspected rendered pages 1, 5, 7, 8, 16, 24, and 26.
