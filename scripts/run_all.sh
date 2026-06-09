#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
python -m experiments.controlled_toy --mode full --seeds 0..11 --population 256 --iterations 5 --optimum-samples 50000 --closed-loop-steps 5 --output-dir results/controlled --figure-dir figures/controlled
python -m experiments.learned_ensemble --seeds 0 1 2 3 4 5 --population 192 --iterations 4 --output-dir results/learned_ensemble --figure-dir figures/learned_ensemble
python -m experiments.sweeps --seeds 0 1 2 3 4 --output-dir results/sweeps --figure-dir figures/sweeps
python scripts/run_claim_audit.py --repo-root . --claim-file docs/claims.json
