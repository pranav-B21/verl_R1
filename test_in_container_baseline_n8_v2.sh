#!/bin/bash
# Held-out evaluation for the PARITY-MATCHED control arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-v2
#
# Outcome-only reward (reward_SPRec.py, USE_RTHINK=0), GRPO n=8. Its resolved
# verl config is byte-identical to gpu-baseline-n8 and to v7/v7b/v8 apart from
# the experiment name -- same rollout.n=8, train_batch=56, ppo_mini=56, lr,
# kl_loss_coef, temperature, 1024/2048 lengths, max_assistant_turns=4.
#
# WHY THIS ARM EXISTS
#   gpu-baseline-n8's step-200 checkpoint was produced by a job that started
#   2026-07-20 18:03, one hour BEFORE the tool_call_repair JSON fix landed
#   (commit e2a492e8, 19:03). v7, v7b and v8 all trained entirely with the fix.
#   So the locked step-200 A/B row compares arms that ran different rollout code.
#   This arm is a fresh run started 2026-07-25 17:30 -- after the fix, alongside
#   v7b -- so it is the control that removes that asymmetry. It was trained and
#   then never decoded; this script is the missing measurement.
#
# PRE-REGISTERED EXPECTATION (state it before looking, then check it)
#   The two controls already agree on the deterministic greedy val reward:
#       step        50        100       150       200
#       n8     -0.03349  -0.01666  -0.00219  -0.00008
#       n8-v2  -0.03456  -0.02069  -0.00419  -0.00019
#   The step-200 gap is 0.0001, roughly 20x below the temperature-1.0 decode
#   spread (gpu-baseline-n8's six step-200 decodes span HR@5 0.002-0.004). So the
#   expectation is that this arm lands INSIDE that band. If it does, the JSON-fix
#   asymmetry is confirmed immaterial and the existing v7/v7b/v8-vs-baseline-n8
#   comparisons stand as published. If it lands outside, the step-200 row of every
#   one of those comparisons has to be re-read against THIS arm instead.
#
# SCOPE
#   Checkpoints exist only through step 200 (50/100/150/200), which is exactly the
#   locked comparison step -- there is nothing to sweep here. For the control's
#   longer trajectory use test_in_container_baseline_n8_sweep.sh on the n8 arm.
#
# Usage:
#   bash test_in_container_baseline_n8_v2.sh                      # step 200, 3 decodes
#   CHECKPOINT_STEP=150 bash test_in_container_baseline_n8_v2.sh  # earlier ckpt
#   EVAL_REPEATS=1 bash test_in_container_baseline_n8_v2.sh       # quick single decode
#
# Env vars: every variable accepted by test_in_container_rthink.sh works here.
# This script only changes the defaults; it delegates the merge -> generate ->
# json -> eval.py pipeline to that script so all arms run byte-identical code.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-v2"
REFERENCE_ARM="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8"
export CHECKPOINT_STEP=${CHECKPOINT_STEP:-200}
export EVAL_REPEATS=${EVAL_REPEATS:-3}
export RTHINK_RUNS="${EXPERIMENT_NAME}:${CHECKPOINT_STEP}"

CKPT_DIR="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME/global_step_${CHECKPOINT_STEP}"
if [ ! -d "$CKPT_DIR" ]; then
    echo "[fatal] checkpoint not found: $CKPT_DIR" >&2
    echo "        available:" >&2
    ls -d "/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"/global_step_* >&2 2>/dev/null
    exit 1
fi

echo "######################################################################"
echo "# Parity-matched control arm (post JSON-fix outcome-only baseline)"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   checkpoint : global_step_$CHECKPOINT_STEP"
echo "#   decodes    : $EVAL_REPEATS  (unseeded, temperature=1.0)"
echo "#   reference  : $REFERENCE_ARM @ same step"
echo "#   answers    : does the tool_call_repair asymmetry move the step-200 bar?"
echo "######################################################################"
echo ""
date

# Everything this job writes is newer than the sentinel, which keeps the
# comparison below from folding a rerun's decodes into the wrong column.
SENTINEL="$(mktemp -t bn8v2_start.XXXXXX)"
trap 'rm -f "$SENTINEL"' EXIT

# Delegate merge/generate/json/eval to the shared core. RTHINK_RUNS already wins
# over the core's EXPERIMENT_NAME back-compat branch; unsetting it is
# belt-and-braces so the core cannot fall through to its v6 default.
unset EXPERIMENT_NAME
bash test_in_container_rthink.sh
CORE_RC=$?
EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-v2"

EVAL_ROOT="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/global_step_${CHECKPOINT_STEP}"

# ---------------------------------------------------------------------------
# [5/6] Retrieval behavior -- same measurement as every other arm, so the tables
# line up. Read it against gpu-baseline-n8's ~99% at step 200.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [5/6] Decode-time retrieval behavior (parity control)"
echo "######################################################################"
PREDS=$(find "$EVAL_ROOT" -name test_predictions.json -newer "$SENTINEL" 2>/dev/null | sort)
if [ -z "$PREDS" ]; then
    echo "[warn] no test_predictions.json newer than job start -- decode failed?" >&2
else
    # shellcheck disable=SC2086
    PYTHONPATH="$PROJECT_DIR" python3 \
        verl/utils/reward_score/reward_retrieval/audits/decode_behavior.py \
        --json "$EVAL_ROOT/decode_behavior.json" $PREDS
fi

# ---------------------------------------------------------------------------
# [6/6] The actual question: this arm's decode band vs the reference arm's, at
# the same step. Reads the metrics off disk rather than quoting numbers, so it
# stays correct as decodes accumulate.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [6/6] Parity check -- post-fix control vs pre-fix control @ step $CHECKPOINT_STEP"
echo "######################################################################"
ARM_DIR="$EVAL_ROOT" \
REF_DIR="$PROJECT_DIR/outputs/eval/$REFERENCE_ARM/global_step_${CHECKPOINT_STEP}" \
SENTINEL="$SENTINEL" STEP="$CHECKPOINT_STEP" python3 - <<'PY'
import json, os, statistics as st

arm_dir = os.environ["ARM_DIR"]
ref_dir = os.environ["REF_DIR"]
t0 = os.path.getmtime(os.environ["SENTINEL"])
step = os.environ["STEP"]

def metrics(path):
    """eval.py writes a 1-element list of dicts; HR/NDCG are 1-element lists."""
    with open(path) as fh:
        d = json.load(fh)[0]
    return float(d["HR"][0]), float(d["NDCG"][0]), float(d["ORRatio"])

def collect(root, newer_than=None):
    out = []
    if not os.path.isdir(root):
        return out
    # EVAL_REPEATS<=1 takes the core's untagged path and writes the metrics at the
    # step-dir root instead of in a decode<i>_<date> subdir, so scan both.
    candidates = [("(untagged)", root)]
    candidates += [(s, os.path.join(root, s)) for s in sorted(os.listdir(root))]
    for sub, d in candidates:
        if not os.path.isdir(d):
            continue
        top1, top5 = (os.path.join(d, f"test_metrics_top{k}.json") for k in (1, 5))
        if not (os.path.isfile(top1) and os.path.isfile(top5)):
            continue
        if newer_than is not None and os.path.getmtime(top5) < newer_than:
            continue
        try:
            hr1, _, orr1 = metrics(top1)
            hr5, ndcg5, _ = metrics(top5)
        except (KeyError, IndexError, ValueError) as exc:
            print(f"[warn] unreadable metrics in {d}: {exc}")
            continue
        out.append((sub, hr1, hr5, ndcg5, orr1))
    return out

arm = collect(arm_dir, newer_than=t0)          # this job only
ref = collect(ref_dir)                          # every decode the reference has

if not arm:
    print("no metrics from this job -- the decode failed, check the log above")
    raise SystemExit(0)

hdr = f"{'arm':<12} {'decode':<20} {'HR@1':>8} {'HR@5':>8} {'NDCG@5':>9} {'ORRatio@1':>10}"
print(hdr)
print("-" * len(hdr))
for label, rows in (("n8-v2 (post)", arm), ("n8 (pre)", ref)):
    for sub, hr1, hr5, ndcg5, orr1 in rows:
        print(f"{label:<12} {sub:<20} {hr1:>8.3f} {hr5:>8.3f} {ndcg5:>9.5f} {orr1:>10.4f}")

def agg(rows, i):
    xs = [r[i] for r in rows]
    return st.mean(xs), min(xs), max(xs)

print()
hdr = f"{'arm':<12} {'n':>3} {'HR@1 mean':>10} {'HR@5 mean':>10} {'HR@5 range':>18} {'NDCG@5 mean':>12}"
print(hdr)
print("-" * len(hdr))
for label, rows in (("n8-v2 (post)", arm), ("n8 (pre)", ref)):
    if not rows:
        print(f"{label:<12} {'0':>3}  (no decodes on disk)")
        continue
    m1, _, _ = agg(rows, 1)
    m5, lo5, hi5 = agg(rows, 2)
    mn, _, _ = agg(rows, 3)
    print(f"{label:<12} {len(rows):>3} {m1:>10.4f} {m5:>10.4f} "
          f"{f'{lo5:.3f} - {hi5:.3f}':>18} {mn:>12.5f}")

if ref and arm:
    _, lo5, hi5 = agg(ref, 2)
    a5, alo, ahi = agg(arm, 2)
    inside = lo5 <= a5 <= hi5
    print()
    print(f"VERDICT @ step {step}: post-fix control HR@5 mean {a5:.4f} vs pre-fix "
          f"reference band {lo5:.3f}-{hi5:.3f}")
    if inside:
        print("  INSIDE the reference band -> the tool_call_repair asymmetry does not")
        print("  move the step-200 bar; existing v7/v7b/v8 comparisons stand as published.")
    else:
        print("  OUTSIDE the reference band -> the asymmetry is material. Re-read the")
        print("  step-200 row of every v7/v7b/v8 comparison against THIS arm, and say so")
        print("  explicitly in TEST_OUTPUT.md rather than silently swapping the control.")
    print()
    print("  Caveat that survives either verdict: both bands are ~1000-prompt decodes at")
    print("  temperature 1.0 with HR@5 in the 0.002-0.004 range, i.e. 2-4 hits. 'Inside")
    print("  the band' means not distinguishable at this sample size, not identical.")
PY

echo ""
date
echo "Decodes under $EVAL_ROOT/decode<i>_<date>/"
echo "Append the result to TEST_OUTPUT.md (workspace root) as a baseline-n8-v2 row,"
echo "and record the verdict next to the M3 step-200 comparison."
exit $CORE_RC
