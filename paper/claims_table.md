# Claims Table

| Claim | Status | Evidence |
|---|---|---|
| CEM can amplify model-error pockets through elite refits. | Supported in controlled smoke/full runs. | `results/*/controlled_traces.csv`, `figures/*/elite_refit_drift.png` |
| CEM can underperform static-proposal baselines in the pocket world. | Partial but strong in the current full run: CEM regret `11.64`, static-proposal regret `6.16`. | `results/controlled/controlled_rollouts.csv` |
| Repairs reduce controlled regret. | Partial but strong in controlled runs: best repaired variants are near `0.22-0.30` regret. | `results/controlled/controlled_summary.json`, `figures/controlled/regret_bars.png` |
| The method transfers to high-dimensional latent control. | Unsupported. | Future benchmark work required. |
