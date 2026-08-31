#!/bin/bash
# Held-out eval SWEEP for the outcome-only control arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 @ steps 200,250,...,500
#
# Same arm and same code path as test_in_container_baseline_n8.sh -- that script
# evaluates ONE checkpoint, this one walks the whole plateau in a single job so
# the checkpoints are decoded back-to-back against one warm retriever.
#
# WHY a sweep rather than the locked step 200
#   The step-200 lock exists to compare ARMS at equal training. It says nothing
#   about where this arm's own held-out curve actually sits. The training-side
#   val reward (greedy, deterministic) only crosses zero around step 250:
#       step  100     200      300      400      500      550
#       val  -0.0167 -0.0001  +0.0062  +0.0085  +0.0083  +0.0107
#   so step 200 is measured on the last pre-takeoff checkpoint. This sweep gives
#   the temperature-1.0 HR/NDCG curve over the same range, which is what says
#   whether 200 is a fair comparison point or an artificially low bar.
#
# WHAT THIS DOES NOT DO
#   It does not make v7/v7b/v8 comparable past step 200 -- those arms only have
#   checkpoints through 200. Only the step-200 row here is an A/B comparator;
#   250-500 characterize the control's own trajectory.
#
# Usage (submit the sbatch wrapper from a LOGIN node, not this file):
#   sbatch sbatch_run_test_baseline_n8_sweep.sh
#   EVAL_REPEATS=1 sbatch sbatch_run_test_baseline_n8_sweep.sh                  # quick pass
#   CHECKPOINT_STEPS="300 400 500" sbatch sbatch_run_test_baseline_n8_sweep.sh  # subset
#
# Env vars
#   CHECKPOINT_STEPS  space-separated steps (default "200 250 300 350 400 450 500").
#                     Steps without a checkpoint on disk are reported and skipped.
#   EVAL_REPEATS      independent decodes per checkpoint (default 3). Decode is
#                     unseeded at temperature=1.0 and this arm's own three decodes
#                     of step 200 returned HR@5 0.002/0.003/0.004 -- the spread
#                     across repeats IS the significance bar, so 1 is not a
#                     measurement. Cost is ~20 min per decode.
#   PURGE_MERGED      1 = delete this sweep's merged HF models on the way out
#                     (~3.8 GB per step; default 0, they make a re-decode cheap).
#   Everything test_in_container_rthink.sh accepts (FORCE_MERGE, GEN_BATCH_SIZE,
#   TOOL_CONFIG, EVAL_CATEGORY, ...) works here unchanged.
#
# KNOWN ARM ASYMMETRY (carry into any writeup)
#   This arm's checkpoints up to ~step 220 were produced by jobs that started
#   BEFORE the tool_call_repair JSON fix (commit e2a492e8, 2026-07-20 19:03);
#   steps ~250 onward were trained with it. v7/v7b/v8 trained entirely with the
#   fix. So the step-200 row carries a substrate asymmetry that the later rows
#   do not -- which is a second reason not to read step 200 as the whole story.
#   The parity-matched control at step 200 is gpu-baseline-n8-v2 (trained
#   2026-07-25/27, post-fix, never decoded).

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

SWEEP_EXPERIMENT_NAME=${EXPERIMENT_NAME:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8"}
EXPERIMENT_NAME="$SWEEP_EXPERIMENT_NAME"
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"200 250 300 350 400 450 500"}
export EVAL_REPEATS=${EVAL_REPEATS:-3}
PURGE_MERGED=${PURGE_MERGED:-0}
SWEEP_DECODE_TEMPERATURE=${DECODE_TEMPERATURE:-1.0}

CKPT_ROOT="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"
EVAL_ROOT="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME"

# ---------------------------------------------------------------------------
# Resolve the step list against what is actually on disk, and fail BEFORE the
# first 20-minute decode if none of it exists.
# ---------------------------------------------------------------------------
RUNS=""
FOUND_STEPS=""
MISSING_STEPS=""
for step in $CHECKPOINT_STEPS; do
    if [ -d "$CKPT_ROOT/global_step_${step}" ]; then
        RUNS="$RUNS ${EXPERIMENT_NAME}:${step}"
        FOUND_STEPS="$FOUND_STEPS $step"
    else
        MISSING_STEPS="$MISSING_STEPS $step"
    fi
done

if [ -z "$RUNS" ]; then
    echo "[fatal] none of the requested steps exist under $CKPT_ROOT" >&2
    echo "        requested:$CHECKPOINT_STEPS" >&2
    echo "        available:" >&2
    ls -d "$CKPT_ROOT"/global_step_* >&2 2>/dev/null
    exit 1
fi
if [ -n "$MISSING_STEPS" ]; then
    echo "[warn] no checkpoint for step(s):$MISSING_STEPS -- skipping those" >&2
fi

# The core walks this list; every entry carries an explicit ":step" so it never
# falls back to CHECKPOINT_STEP (whose default in the core is "latest").
export RTHINK_RUNS="${RUNS# }"

N_STEPS=$(echo $FOUND_STEPS | wc -w)
echo "######################################################################"
echo "# M3 control arm -- held-out eval SWEEP"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   steps      :$FOUND_STEPS   ($N_STEPS checkpoints)"
echo "#   decodes    : $EVAL_REPEATS per checkpoint, temperature=$SWEEP_DECODE_TEMPERATURE"
echo "#   total      : $((N_STEPS * EVAL_REPEATS)) decodes  (~20 min each)"
echo "#   iso-step   : only step 200 is an A/B comparator vs v7/v7b/v8"
echo "######################################################################"
echo ""
date

# Sentinel: everything this job writes is newer than this file, which is how the
# summary below tells today's decodes apart from the 07-21 / 07-24 decode dirs
# that already sit under global_step_200.
SENTINEL="$(mktemp -t sweep_start.XXXXXX)"
trap 'rm -f "$SENTINEL"' EXIT

# ---------------------------------------------------------------------------
# [1/3] merge -> generate -> eval.py for every step, via the shared core, so
# this arm runs byte-identical evaluation code to every other arm.
# ---------------------------------------------------------------------------
unset EXPERIMENT_NAME   # else the core takes its single-run back-compat branch
bash test_in_container_rthink.sh
CORE_RC=$?
EXPERIMENT_NAME="$SWEEP_EXPERIMENT_NAME"

# ---------------------------------------------------------------------------
# [2/3] Decode-time retrieval behavior, per step. Same measurement the v7/v7b/v8
# arms report, so the tables line up. For this arm it is the reference level
# (~99% retrieval at step 200), not the finding.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [2/3] Decode-time retrieval behavior, per checkpoint"
echo "######################################################################"
for step in $FOUND_STEPS; do
    STEP_DIR="$EVAL_ROOT/global_step_${step}"
    # Only this job's decodes -- older decode dirs under step 200 would otherwise
    # be folded into today's behavior numbers.
    PREDS=$(find "$STEP_DIR" -name test_predictions.json -newer "$SENTINEL" 2>/dev/null | sort)
    echo ""
    echo "--- global_step_${step} ---"
    if [ -z "$PREDS" ]; then
        echo "[warn] no test_predictions.json newer than job start -- decode failed?" >&2
        continue
    fi
    # shellcheck disable=SC2086
    PYTHONPATH="$PROJECT_DIR" python3 \
        verl/utils/reward_score/reward_retrieval/audits/decode_behavior.py \
        --json "$STEP_DIR/decode_behavior.json" $PREDS
done

# ---------------------------------------------------------------------------
# [3/3] Curve across steps: per-decode rows plus mean/min/max per step. The core
# already printed one row per decode; this collapses them into the shape you
# actually read -- does the held-out metric climb, plateau, or decline, and is
# any step-to-step move bigger than the within-step decode spread?
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [3/3] Held-out curve across checkpoints (this job's decodes only)"
echo "######################################################################"
EVAL_ROOT="$EVAL_ROOT" SENTINEL="$SENTINEL" STEPS="$FOUND_STEPS" python3 - <<'PY'
import json, os, statistics as st

root = os.environ["EVAL_ROOT"]
t0 = os.path.getmtime(os.environ["SENTINEL"])
steps = [int(s) for s in os.environ["STEPS"].split()]

def metrics(path):
    """eval.py writes a 1-element list of dicts; HR/NDCG are 1-element lists."""
    with open(path) as fh:
        d = json.load(fh)[0]
    return float(d["HR"][0]), float(d["NDCG"][0]), float(d["ORRatio"])

rows = {}
for step in steps:
    step_dir = os.path.join(root, f"global_step_{step}")
    if not os.path.isdir(step_dir):
        continue
    # EVAL_REPEATS<=1 takes the core's untagged path and writes the metrics at the
    # step-dir root instead of in a decode<i>_<date> subdir, so scan both.
    candidates = [("(untagged)", step_dir)]
    candidates += [(s, os.path.join(step_dir, s)) for s in sorted(os.listdir(step_dir))]
    for sub, d in candidates:
        if not os.path.isdir(d):
            continue
        top1, top5 = (os.path.join(d, f"test_metrics_top{k}.json") for k in (1, 5))
        if not (os.path.isfile(top1) and os.path.isfile(top5)):
            continue
        if os.path.getmtime(top5) < t0:      # a decode from an earlier job
            continue
        try:
            hr1, _, orr1 = metrics(top1)
            hr5, ndcg5, _ = metrics(top5)
        except (KeyError, IndexError, ValueError) as exc:
            print(f"[warn] unreadable metrics in {d}: {exc}")
            continue
        rows.setdefault(step, []).append((sub, hr1, hr5, ndcg5, orr1))

if not rows:
    print("no metrics from this job -- every decode failed, check the log above")
    raise SystemExit(0)

hdr = f"{'step':>6} {'decode':<20} {'HR@1':>8} {'HR@5':>8} {'NDCG@5':>9} {'ORRatio@1':>10}"
print(hdr)
print("-" * len(hdr))
for step in sorted(rows):
    for sub, hr1, hr5, ndcg5, orr1 in rows[step]:
        print(f"{step:>6} {sub:<20} {hr1:>8.3f} {hr5:>8.3f} {ndcg5:>9.5f} {orr1:>10.4f}")

print()
print("mean over decodes (spread = max-min within the step):")
hdr = (f"{'step':>6} {'n':>3} {'HR@1':>8} {'spread':>8} {'HR@5':>8} {'spread':>8} "
       f"{'NDCG@5':>9} {'spread':>9}")
print(hdr)
print("-" * len(hdr))
for step in sorted(rows):
    v = rows[step]
    def col(i):
        xs = [r[i] for r in v]
        return st.mean(xs), (max(xs) - min(xs))
    (m1, s1), (m5, s5), (mn, sn) = col(1), col(2), col(3)
    print(f"{step:>6} {len(v):>3} {m1:>8.4f} {s1:>8.4f} {m5:>8.4f} {s5:>8.4f} "
          f"{mn:>9.5f} {sn:>9.5f}")

print()
print("Read it this way: a step-to-step move smaller than the within-step spread")
print("is not a result. Prior evidence for this bar -- six decodes of step 200")
print("(2026-07-21 + 07-24) gave HR@5 0.002 0.003 0.004 0.004 0.003 0.002.")
PY

# ---------------------------------------------------------------------------
# Optional cleanup. Merged HF models are ~3.8 GB per checkpoint and are what
# makes a re-decode cheap, so they are kept unless you ask otherwise.
# ---------------------------------------------------------------------------
if [ "$PURGE_MERGED" = "1" ]; then
    echo ""
    echo "[cleanup] removing merged HF models for this sweep (PURGE_MERGED=1)"
    for step in $FOUND_STEPS; do
        rm -rf "/scratch/11138/pranavbelligundu/verl/merged_models/$EXPERIMENT_NAME/global_step_${step}"
    done
fi

echo ""
date
echo "Decodes live under $EVAL_ROOT/global_step_<step>/<decode-tag><i>_<date>/"
echo "Append the per-step means to TEST_OUTPUT.md (workspace root)."
exit $CORE_RC
