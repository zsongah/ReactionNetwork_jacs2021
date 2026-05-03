#!/usr/bin/env bash
# Long-running FlowER-as-generator demo on Apple Silicon CPU.
# Populates cache/flower/ with real FlowER priors for size-1 and size-2
# organic combos drawn from the LEDC fragrec pool (~200 species).
#
# Expected runtime: 1-3 hours on M-series CPU.
# Cache writes are atomic, so this script is safe to interrupt and resume.
#
# Usage:
#   bash scripts/run_long_demo.sh           # foreground
#   nohup bash scripts/run_long_demo.sh > results/long_demo.log 2>&1 &
#   echo $! > /tmp/flower_demo.pid          # save PID for later kill
#
# Tail progress:
#   tail -f results/long_demo.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# FlowER subprocess wiring. MODEL_PATH is the *directory* containing the
# .pt checkpoint, not the file itself.
export FLOWER_REPO_PATH="$REPO_ROOT/vendor/FlowER"
export FLOWER_MODEL_PATH="$REPO_ROOT/vendor/FlowER/checkpoints/flower_new_dataset/best_large_hyperparam"
export FLOWER_PYTHON_EXECUTABLE="$HOME/miniforge3/envs/flower/bin/python"

# Activate the main jacs2021 env (the FlowER env is invoked via
# FLOWER_PYTHON_EXECUTABLE only).
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jacs2021

mkdir -p results

# Reactive whitelist for size-2 combos. Without this filter, size-2
# enumeration is O(N^2) over a 197-species pool ~= 19500 combos,
# taking ~10 hours on Apple Silicon CPU. With this whitelist, only
# pairs containing at least one reactive small species remain
# (~K * N ~= 2000 combos). The whitelist captures the species that
# actually carry reactivity in the EC + Li -> LEDC mechanism: the
# substrates (EC, VC, DMC), small radicals (H., OH., HO2.), small
# closed-shell molecules (H2O, H2, CO2, formaldehyde, O2), and the
# parent peroxide. Large oligomeric carbonate fragments still appear
# as size-1 (rearrangement) and as the *partner* of a reactive species.
REACTIVE_SMILES='[H]C1([H])OC(=O)OC1([H])[H],[H]c1oc(=O)oc1[H],[H]C([H])([H])OC(=O)OC([H])([H])[H],[H]O[H],[H][H],[H][O],[H]OO[H],O=C=O,[H]C([H])=O,[H][C]([H])[H],O=O'

echo "=== Long demo started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Cache size before: $(ls cache/flower/ 2>/dev/null | wc -l | tr -d ' ') entries"

python -m src.run_demo \
    --backend both \
    --flower-as-generator \
    --combo-sizes 1,2 \
    --reactive-smiles "$REACTIVE_SMILES" \
    --lam 0.5 \
    --use-tier-bias \
    --n-paths 10 \
    --verbose

# Snapshot the result JSON with a timestamped name so subsequent demo
# runs don't clobber it.
if [[ -f results/ledc_paths.json ]]; then
    cp results/ledc_paths.json "results/long_demo_$(date -u +%Y%m%dT%H%M%SZ).json"
fi

echo "=== Long demo finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Cache size after: $(ls cache/flower/ 2>/dev/null | wc -l | tr -d ' ') entries"
