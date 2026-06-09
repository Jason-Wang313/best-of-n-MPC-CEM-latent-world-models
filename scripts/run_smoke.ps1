$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
  $env:PYTHONPATH = "$Root\src"
  python -m experiments.controlled_toy --mode smoke --seeds 0 1 --population 96 --iterations 3 --optimum-samples 8000 --closed-loop-steps 2 --output-dir results/smoke --figure-dir figures/smoke
  python scripts/run_claim_audit.py --repo-root . --claim-file docs/claims.json
} finally {
  Pop-Location
}
