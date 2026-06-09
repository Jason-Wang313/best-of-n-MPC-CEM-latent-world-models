#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
python -m experiments.controlled_toy --mode smoke --seeds 0 1 --population 96 --iterations 3 --optimum-samples 8000 --closed-loop-steps 2 --output-dir results/smoke --figure-dir figures/smoke
python scripts/run_claim_audit.py --repo-root . --claim-file docs/claims.json
