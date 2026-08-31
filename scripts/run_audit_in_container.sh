#!/bin/bash
set -euo pipefail

# Run one reward_retrieval audit inside the training container.
#
# WHY THIS EXISTS
#   Several audit scripts document themselves as running under "bare login-node
#   python3" (advantage_mass.py:91, and memtype_payoff.py imports the same
#   helpers). That is not true on this cluster and cost a debugging cycle on
#   2026-08-30. Measured there:
#
#     /usr/bin/python3                              no torch
#     anaconda/bin/python                           no torch
#     anaconda/envs/retriever/bin/python            torch 2.4.1, NO sentence_transformers
#     sglang_25.10-py3-tls-fixed.sif                torch 2.9.0 + sentence_transformers 5.1.2  <-- only one that works
#
#   The audits need BOTH: torch for the cdist/argsort GT ranking, and
#   sentence-transformers to embed the answer strings with
#   paraphrase-MiniLM-L3-v2 (the same encoder eval.py's embeddings.pt was built
#   with -- swapping it silently changes every rank).
#
# THREE THINGS THIS WRAPPER GETS RIGHT THAT AN AD-HOC `singularity exec` DOES NOT
#   1. --pwd must be the PROJECT ROOT. memtype_payoff.CATALOG is the RELATIVE
#      path ./data/amazon_data/CDs_and_Vinyl, so running from the audits/
#      directory dies with FileNotFoundError on embeddings.pt.
#   2. HF_HOME must be forwarded. The model cache lives at
#      /scratch/11138/pranavbelligundu/huggingface, NOT in $HOME/.cache; with
#      HF_HOME unset inside the container, SentenceTransformer tries the network
#      and compute nodes have none. HF_HUB_OFFLINE=1 makes that failure loud
#      instead of a silent hang.
#   3. LD_LIBRARY_PATH must have cuda/compat stripped -- the container's CUDA 13.0
#      compat libs shadow the host driver (CUDA error 803). These audits are
#      CPU-only so it is not fatal here, but it produces alarming warnings that
#      have been mistaken for real failures before.
#
#   PYTHONPATH is pinned to the project root for the same reason every other
#   offline script in this repo pins it: a stale installed verl at
#   ~/.local/lib/python3.12/site-packages/verl silently returns bare 0.0.
#
# CPU-ONLY BY DESIGN. No --nv. advantage_mass.py hardcodes device="cpu", the
# work is a few seconds of cdist, and dropping --nv means this runs on a login
# node as happily as on a compute node.
#
# Usage:
#   bash scripts/run_audit_in_container.sh <audit-name> [args...]
#
#   <audit-name> is the script stem under
#   verl/utils/reward_score/reward_retrieval/audits/ -- e.g. advantage_mass,
#   memtype_payoff, paired_arm_test, decode_behavior, selection_capability_probe.
#   Everything after it is passed through verbatim.
#
# Examples:
#   bash scripts/run_audit_in_container.sh advantage_mass --label base200-TRAIN \
#     outputs/eval/<arm>/global_step_200/traindiag*/test_predictions.json \
#     --json verl/utils/reward_score/reward_retrieval/audits/advantage_mass_base200_TRAIN.json
#
#   bash scripts/run_audit_in_container.sh memtype_payoff \
#     outputs/eval/<arm>/global_step_200/decode*/test_predictions.json --label base200-9dec
#
# NOTE ON GLOBS: they are expanded by YOUR shell before the container starts, so
# they resolve against the host filesystem -- which is what you want, since /work
# and /scratch are bind-mounted at the same paths inside.

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONTAINER="$PROJECT_DIR/sglang_25.10-py3-tls-fixed.sif"
AUDIT_DIR="verl/utils/reward_score/reward_retrieval/audits"

if [ "$#" -lt 1 ]; then
    echo "usage: bash scripts/run_audit_in_container.sh <audit-name> [args...]" >&2
    echo "       audit-name is a script stem under $AUDIT_DIR" >&2
    exit 2
fi

AUDIT="${1%.py}"; shift
AUDIT_PATH="$AUDIT_DIR/$AUDIT.py"

if [ ! -f "$PROJECT_DIR/$AUDIT_PATH" ]; then
    echo "[fatal] no such audit: $PROJECT_DIR/$AUDIT_PATH" >&2
    echo "        available:" >&2
    ls "$PROJECT_DIR/$AUDIT_DIR"/*.py 2>/dev/null | sed 's|.*/|          |;s|\.py$||' >&2
    exit 2
fi
[ -f "$CONTAINER" ] || { echo "[fatal] container not found: $CONTAINER" >&2; exit 2; }

# tacc-apptainer is not in the default PATH on compute nodes; on login nodes it
# usually is. Load it only if `singularity` is missing, so an already-correct
# environment is left alone.
#
# `set -euo pipefail` MUST be relaxed around this block. Lmod's init script and
# its `module` shell function are not written for `set -u`, and under it they
# abort the whole script with status 1 and NO output -- which is exactly how
# this looked when first written on 2026-08-30: a silent `exit=1` even for
# `--help`. Do not fold these two lines back in.
if ! command -v singularity >/dev/null 2>&1; then
    set +eu
    # shellcheck disable=SC1091
    source /etc/profile.d/z00_lmod.sh 2>/dev/null
    module load tacc-apptainer/1.4.1 >/dev/null 2>&1 || module load tacc-apptainer >/dev/null 2>&1
    set -eu
fi
command -v singularity >/dev/null 2>&1 || {
    echo "[fatal] singularity/apptainer unavailable; try: module load tacc-apptainer" >&2; exit 2; }

echo "[audit] $AUDIT_PATH"
echo "[audit] args: $*"

exec singularity exec \
    --bind /work:/work \
    --bind /scratch:/scratch \
    --env HF_HOME="${HF_HOME:-/scratch/11138/pranavbelligundu/huggingface}" \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONPATH="$PROJECT_DIR" \
    --env TOKENIZERS_PARALLELISM=false \
    --pwd "$PROJECT_DIR" \
    "$CONTAINER" \
    bash -c '
        set -euo pipefail
        export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v "cuda/compat" | paste -sd ":" -)
        exec python3 "$@"
    ' _ "$AUDIT_PATH" "$@"
