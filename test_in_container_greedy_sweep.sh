#!/bin/bash
# GREEDY (temperature=0) held-out eval sweep across ARMS x STEPS.
#
# WHY THIS EXISTS
#   Every held-out number in TEST_OUTPUT.md was decoded at temperature=1.0, and
#   the sampling noise of that regime is larger than any arm gap we have
#   measured: three decodes of ONE frozen checkpoint at step 200 gave HR@5
#   0.006 / 0.000 / 0.001. That is why the tables carry a "spread" column and
#   why no comparison has ever cleared it.
#
#   Deployment inference does not sample. A trained policy is served at
#   temperature 0, so the decode that matters for a recommendation metric is the
#   argmax one -- the realistic setting, and one that SHRINKS the noise term
#   instead of averaging it down.
#
#   MEASURED 2026-08-08, and the first draft of this header was wrong: greedy is
#   NOT deterministic here. Two greedy decodes of baseline-n8@200 reproduce only
#   35.1% of rollouts byte-for-byte (see "DETERMINISM IS MEASURED" below -- the
#   probe caught exactly what it was built to catch). Judge the METRIC, not the
#   text: 95% of rollouts sit past GT rank 500, so most divergence is invisible
#   to HR. Same pair, same instrument:
#
#       regime      identical text   identical answer   HR@5 repeat spread
#       greedy           35.1%            62.0%              0.00102
#       temp 1.0          0.0%             1.5%              0.00407
#
#   HR@1 was reproduced EXACTLY under greedy and not under temp-1.0. So greedy
#   buys ~4x on the metric and 41x on answer-level reproducibility -- a single
#   greedy decode beats a single sampled one, and since paired_arm_test.py pairs
#   the ARMS within prompt, a cross-arm greedy comparison is valid without
#   within-arm repeats. Keep >=2 greedy decodes where affordable to quote the
#   residual; do not claim the residual is zero.
#
#   The trainer already agrees: rollout.val_kwargs is temperature=0,
#   do_sample=False, n=1 (verl/trainer/config/rollout/rollout.yaml), so every
#   wandb val curve for every arm is ALREADY greedy. The temperature-1.0 eval
#   harness was the odd one out. This script closes that gap, which also means
#   the greedy HR/NDCG produced here is the metric that belongs next to the
#   val-curve story in the paper, not a second opinion on it.
#
# WHAT IT DOES NOT FIX
#   Greedy SHRINKS decode noise ~4x (it does not remove it -- above). It does not
#   remove the ~3.1% retrieval-coverage ceiling, and it does not make an
#   8-events-per-1000 metric well-powered. A greedy HR@5 difference of one or two
#   prompts is still one or two prompts. The paired per-prompt test
#   (audits/paired_arm_test.py --tag greedy) is what turns these decodes into a
#   significance statement; under greedy it is a McNemar over disagreeing prompts
#   with a small residual decode term (~1 prompt/1000 on HR@5), not zero.
#
#   Greedy also makes tool-call JSON failures REPRODUCIBLE rather than averaged:
#   at temperature 1.0 a prompt that malforms its query_list may succeed on the
#   next decode, so EVAL_REPEATS=3 partly averages the loss away; under greedy it
#   is a fixed per-prompt loss. Both arms hit it and the test is within-prompt so
#   it does not bias the comparison, but report the greedy JSON-failure rate
#   alongside the table rather than folding it into "decode noise".
#
# DETERMINISM IS MEASURED, NOT ASSUMED
#   sglang batching and the radix cache can reorder reductions, and this is a
#   multi-turn rollout: one flipped token inside a <search> query changes the
#   retrieved documents and can change the answer. So the FIRST step of each arm
#   is decoded TWICE and the two decodes are diffed prompt-by-prompt. If they
#   agree exactly, one decode per step is a measurement and the remaining steps
#   run once. If they do not, the residual spread is the real bar and gets
#   reported as such -- that is a finding, not a failure.
#
# Usage (submit the sbatch wrapper from a LOGIN node):
#   sbatch sbatch_run_test_greedy_sweep.sh
#   STEPS="200 300 400 500" sbatch sbatch_run_test_greedy_sweep.sh
#   ARMS="gpu-baseline-n8 gpu-rthink-v8-n8 gpu-rthink-v7b-n8" sbatch sbatch_run_test_greedy_sweep.sh
#
# Env vars
#   ARMS      space-separated arm suffixes (the part after
#             "nq-search-r1-grpo-qwen3-1.7b-sbatch-"), control FIRST.
#             Default: "gpu-baseline-n8 gpu-rthink-v8-n8".
#   STEPS     space-separated checkpoint steps (default "200 250 300 350 400 450 500",
#             the grid where baseline-n8 and v8-n8 both have checkpoints).
#             Steps missing for an arm are reported and skipped for that arm only.
#   PROBE_REPEATS  decodes of the first available step per arm (default 2, the
#             determinism probe). Set 1 to skip the probe once it has been run.
#   PURGE_MERGED   1 = delete this sweep's merged HF models on the way out
#             (~3.8 GB per step per arm).
#
# ORDERING NOTE: all arms are decoded inside ONE job against ONE warm retriever
# instance, so retriever state cannot differ between arms. That is a real
# confound in any design that gives each arm its own job.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

PREFIX="nq-search-r1-grpo-qwen3-1.7b-sbatch-"
ARMS=${ARMS:-"gpu-baseline-n8 gpu-rthink-v8-n8"}
STEPS=${STEPS:-"200 250 300 350 400 450 500"}
PROBE_REPEATS=${PROBE_REPEATS:-2}
PURGE_MERGED=${PURGE_MERGED:-0}
TODAY=$(date +%F)

# Greedy for every decode this script issues. Exported (not passed per call) so
# there is exactly one place the regime is set.
export DECODE_TEMPERATURE=0
export DECODE_TAG=greedy

CKPT_ROOT="/scratch/11138/pranavbelligundu/verl"
EVAL_ROOT="$PROJECT_DIR/outputs/eval"

# ---------------------------------------------------------------------------
# Resolve arms x steps against what is on disk. Fail BEFORE the first decode if
# nothing resolves, and record the first found step per arm as its probe step.
# ---------------------------------------------------------------------------
PROBE_RUNS=""
REST_RUNS=""
RESOLVED=""
LIVE_ARMS=""
PROBE_STEPS=""      # parallel to LIVE_ARMS: each arm's first available step
for arm in $ARMS; do
    exp="${PREFIX}${arm}"
    if [ ! -d "$CKPT_ROOT/$exp" ]; then
        echo "[warn] no checkpoint root for arm '$arm' ($CKPT_ROOT/$exp) -- skipping" >&2
        continue
    fi
    arm_steps=""
    for step in $STEPS; do
        if [ -d "$CKPT_ROOT/$exp/global_step_${step}" ]; then
            arm_steps="$arm_steps $step"
        fi
    done
    if [ -z "$arm_steps" ]; then
        echo "[warn] arm '$arm' has none of the requested steps -- skipping" >&2
        continue
    fi
    probe=$(echo $arm_steps | awk '{print $1}')
    LIVE_ARMS="$LIVE_ARMS $arm"
    PROBE_STEPS="$PROBE_STEPS $probe"
    RESOLVED="$RESOLVED ${arm}[${arm_steps# }]"
    echo "[arm] $arm -> steps$arm_steps (probe step $probe)"
done

# Build the decode order STEP-MAJOR (baseline:250, v8:250, baseline:300, ...)
# rather than arm-major. The retriever can crash and be restarted mid-job by
# sbatch_run_test_rthink.sh; decoding one whole arm and then the other would
# align that restart boundary with the arm boundary, making retriever epoch a
# confound in exactly the comparison this job exists to make. Interleaving
# spreads any such event across both arms.
for step in $STEPS; do
    for arm in $LIVE_ARMS; do
        exp="${PREFIX}${arm}"
        [ -d "$CKPT_ROOT/$exp/global_step_${step}" ] || continue
        probe=$(echo $LIVE_ARMS $PROBE_STEPS | awk -v a="$arm" \
            '{n=NF/2; for(i=1;i<=n;i++) if($i==a) print $(i+n)}')
        if [ "$step" = "$probe" ] && [ "$PROBE_REPEATS" -gt 1 ]; then
            PROBE_RUNS="$PROBE_RUNS ${exp}:${step}"
        else
            REST_RUNS="$REST_RUNS ${exp}:${step}"
        fi
    done
done

if [ -z "$PROBE_RUNS$REST_RUNS" ]; then
    echo "[fatal] nothing to decode: no arm/step combination exists on disk" >&2
    exit 1
fi

N_DECODES=$(( $(echo $PROBE_RUNS | wc -w) * PROBE_REPEATS + $(echo $REST_RUNS | wc -w) ))
echo "######################################################################"
echo "# GREEDY held-out eval sweep (temperature=0, top_p=1, top_k=1)"
echo "#   arms/steps :$RESOLVED"
echo "#   probe      : first step of each arm decoded ${PROBE_REPEATS}x (determinism)"
echo "#   decodes    : $N_DECODES total  (~7 min each)"
echo "#   output     : outputs/eval/<arm>/global_step_<n>/greedy<i>_${TODAY}/"
echo "#   NOTE       : greedy decodes never land in decode<i>_* dirs, so the"
echo "#                temperature-1.0 tables and audits are untouched."
echo "######################################################################"
date

SENTINEL="$(mktemp -t greedysweep_start.XXXXXX)"
trap 'rm -f "$SENTINEL"' EXIT

# ---------------------------------------------------------------------------
# [1/4] Decode. Probe steps first (so a determinism problem surfaces in the
# first ~30 min rather than after the whole sweep has been spent), then the rest
# at one decode each.
# ---------------------------------------------------------------------------
CORE_RC=0
PROBE_OK=1
if [ -n "$PROBE_RUNS" ]; then
    echo ""
    echo ">>> [1/4a] determinism probe: ${PROBE_REPEATS} greedy decodes of$PROBE_RUNS"
    env -u EXPERIMENT_NAME RTHINK_RUNS="${PROBE_RUNS# }" EVAL_REPEATS="$PROBE_REPEATS" \
        bash test_in_container_rthink.sh || CORE_RC=$?

    # ABORT GATE. temperature=0 is a decode path this harness has never run
    # before, so the probe is also its smoke test. If it produced no predictions
    # at all, the remaining steps would fail the same way and burn ~1.5 h of the
    # allocation proving it. Fail here instead. (A PARTIAL probe still proceeds:
    # one arm failing while the other works is a per-arm problem, and the
    # per-step warnings below will name it.)
    PROBE_OK=0
    for entry in $PROBE_RUNS; do
        exp="${entry%:*}"; step="${entry##*:}"
        if [ -s "$EVAL_ROOT/$exp/global_step_${step}/${DECODE_TAG}1_${TODAY}/test_predictions.json" ]; then
            PROBE_OK=1
        fi
    done
    if [ "$PROBE_OK" = "0" ]; then
        echo ""
        echo "[fatal] the determinism probe produced NO predictions for any arm." >&2
        echo "        Greedy decode is failing, not the checkpoints -- the same" >&2
        echo "        merged models decode fine at temperature 1.0. Check the" >&2
        echo "        [2/4] generation log above for the sglang/main_generation" >&2
        echo "        error before re-submitting; skipping the remaining" >&2
        echo "        $(echo $REST_RUNS | wc -w) decodes." >&2
        # Never exit 0 on a fatal: the core can return 0 while producing nothing
        # (every checkpoint skipped, for instance), and a 0 here reads as
        # "COMPLETED" in sacct -- the same masking that made the 2026-08-02
        # Lustre eviction look like a clean run for a day.
        [ "$CORE_RC" -ne 0 ] && exit "$CORE_RC"
        exit 1
    fi
fi
if [ -n "$REST_RUNS" ]; then
    echo ""
    echo ">>> [1/4b] remaining steps, one greedy decode each"
    env -u EXPERIMENT_NAME RTHINK_RUNS="${REST_RUNS# }" EVAL_REPEATS=1 \
        bash test_in_container_rthink.sh || CORE_RC=$?
fi

# ---------------------------------------------------------------------------
# [2/4] Determinism verdict. Compare the probe's two decodes prompt-by-prompt.
# Identical predictions => the greedy decode is a measurement and a single
# decode per step is enough. Any drift is reported with its size.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [2/4] Determinism probe -- do two greedy decodes of one checkpoint agree?"
echo "######################################################################"
if [ "$PROBE_REPEATS" -le 1 ]; then
    echo "PROBE_REPEATS=1, probe skipped."
else
    EVAL_ROOT="$EVAL_ROOT" PROBE_RUNS="$PROBE_RUNS" TODAY="$TODAY" python3 - <<'PY'
import json, os

root, today = os.environ["EVAL_ROOT"], os.environ["TODAY"]
for entry in os.environ["PROBE_RUNS"].split():
    exp, step = entry.rsplit(":", 1)
    d = os.path.join(root, exp, f"global_step_{step}")
    a = os.path.join(d, f"greedy1_{today}", "test_predictions.json")
    b = os.path.join(d, f"greedy2_{today}", "test_predictions.json")
    short = exp.replace("nq-search-r1-grpo-qwen3-1.7b-sbatch-", "")
    if not (os.path.isfile(a) and os.path.isfile(b)):
        print(f"{short} @ {step}: probe decodes missing -- check the decode log above")
        continue
    ra, rb = (json.load(open(p, encoding="utf-8")) for p in (a, b))
    n = min(len(ra), len(rb))
    diff = [i for i in range(n)
            if "".join(ra[i].get("predict") or []) != "".join(rb[i].get("predict") or [])]
    pct = 100.0 * len(diff) / n if n else float("nan")
    verdict = ("DETERMINISTIC -- one decode per step is a measurement"
               if not diff else
               f"NOT deterministic -- {len(diff)}/{n} prompts ({pct:.1f}%) differ; "
               "that residual IS the bar, keep >=2 decodes")
    print(f"{short} @ {step}: {verdict}")
PY
fi

# ---------------------------------------------------------------------------
# [3/4] Gate 0 (validity) + Gate 1 (mechanism) per arm/step, same audit scripts
# and same definitions as the temperature-1.0 sweeps.
# Gate 0 first: an arm that stopped retrieving has not won, it has opted out.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [3/4] Decode behavior + selection funnel (Gate 0, Gate 1)"
echo "######################################################################"
for entry in $PROBE_RUNS $REST_RUNS; do
    exp="${entry%:*}"; step="${entry##*:}"
    STEP_DIR="$EVAL_ROOT/$exp/global_step_${step}"
    PREDS=$(find "$STEP_DIR" -name test_predictions.json -newer "$SENTINEL" 2>/dev/null | sort)
    echo ""
    echo "--- ${exp#nq-search-r1-grpo-qwen3-1.7b-sbatch-} @ global_step_${step} ---"
    if [ -z "$PREDS" ]; then
        echo "[warn] no greedy decode from this job -- decode failed?" >&2
        continue
    fi
    # shellcheck disable=SC2086
    PYTHONPATH="$PROJECT_DIR" python3 \
        verl/utils/reward_score/reward_retrieval/audits/decode_behavior.py \
        --json "$STEP_DIR/greedy_decode_behavior.json" $PREDS
    # shellcheck disable=SC2086
    python3 verl/utils/reward_score/reward_retrieval/audits/decode_selection.py \
        --json "$STEP_DIR/greedy_decode_selection.json" $PREDS
done

# ---------------------------------------------------------------------------
# [4/4] The cross-arm greedy table -- the thing that goes in the paper.
# One row per arm/step, arms side by side at each step.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [4/4] Greedy held-out metrics, all arms (this job's decodes only)"
echo "######################################################################"
EVAL_ROOT="$EVAL_ROOT" RUNS="$PROBE_RUNS $REST_RUNS" SENTINEL="$SENTINEL" TODAY="$TODAY" \
python3 - <<'PY'
import json, os

root = os.environ["EVAL_ROOT"]
t0 = os.path.getmtime(os.environ["SENTINEL"])
PREFIX = "nq-search-r1-grpo-qwen3-1.7b-sbatch-"

def metrics(path):
    """eval.py writes a 1-element list of dicts; HR/NDCG are 1-element lists."""
    with open(path) as fh:
        d = json.load(fh)[0]
    return float(d["HR"][0]), float(d["NDCG"][0]), float(d["ORRatio"])

rows = {}          # (arm, step) -> list of (tag, hr1, hr5, ndcg5, orr1)
for entry in os.environ["RUNS"].split():
    exp, step = entry.rsplit(":", 1)
    step_dir = os.path.join(root, exp, f"global_step_{step}")
    if not os.path.isdir(step_dir):
        continue
    for tag in sorted(os.listdir(step_dir)):
        d = os.path.join(step_dir, tag)
        if not (tag.startswith("greedy") and os.path.isdir(d)):
            continue
        top1, top5 = (os.path.join(d, f"test_metrics_top{k}.json") for k in (1, 5))
        if not (os.path.isfile(top1) and os.path.isfile(top5)):
            continue
        if os.path.getmtime(top5) < t0:        # a greedy decode from an earlier job
            continue
        try:
            hr1, _, orr1 = metrics(top1)
            hr5, ndcg5, _ = metrics(top5)
        except (KeyError, IndexError, ValueError) as exc:
            print(f"[warn] unreadable metrics in {d}: {exc}")
            continue
        rows.setdefault((exp.replace(PREFIX, ""), int(step)), []).append(
            (tag, hr1, hr5, ndcg5, orr1))

if not rows:
    print("no greedy metrics from this job -- every decode failed, check the log above")
    raise SystemExit(0)

arms = sorted({a for a, _ in rows})
steps = sorted({s for _, s in rows})

hdr = f"{'arm':<22} {'step':>5} {'decode':<18} {'HR@1':>8} {'HR@5':>8} {'NDCG@5':>9} {'ORRatio@1':>10}"
print(hdr); print("-" * len(hdr))
for arm in arms:
    for step in steps:
        for tag, hr1, hr5, ndcg5, orr1 in rows.get((arm, step), []):
            print(f"{arm:<22} {step:>5} {tag:<18} {hr1:>8.4f} {hr5:>8.4f} "
                  f"{ndcg5:>9.5f} {orr1:>10.4f}")

# Arms side by side at each step, on HR@5 (the headline metric in Table 1).
print()
print("HR@5 by step, arms side by side (mean over this job's greedy decodes):")
hdr = f"{'step':>5} " + " ".join(f"{a[:20]:>20}" for a in arms)
print(hdr); print("-" * len(hdr))
for step in steps:
    cells = []
    for arm in arms:
        v = rows.get((arm, step))
        cells.append(f"{sum(r[2] for r in v) / len(v):>20.4f}" if v else f"{'-':>20}")
    print(f"{step:>5} " + " ".join(cells))

print()
print("Read in the pre-registered order -- Gate 0 (retr%/ans% in [3/4]) BEFORE")
print("any HR here; an arm that stopped retrieving is discarded, not compared.")
print()
print("Then run the paired per-prompt test, which is what makes a difference")
print("here a claim rather than a count (exact McNemar, no decode-noise term):")
print(f"  python3 verl/utils/reward_score/reward_retrieval/audits/paired_arm_test.py \\")
print(f"      --tag greedy --n-decodes 1 --steps {' '.join(str(s) for s in steps)} \\")
for arm in arms[:2]:
    label = arm.replace("gpu-", "").replace("rthink-", "")
    print(f"      --arm {label}={PREFIX}{arm}:{os.environ['TODAY']} \\")
print("      --json .../audits/paired_arm_test_greedy_<date>.json")
print()
print("Append the per-step rows to TEST_OUTPUT.md, labelled GREEDY -- they are")
print("NOT comparable to the temperature-1.0 rows already in that file.")
PY

if [ "$PURGE_MERGED" = "1" ]; then
    echo ""
    echo "[cleanup] removing merged HF models for this sweep (PURGE_MERGED=1)"
    for entry in $PROBE_RUNS $REST_RUNS; do
        exp="${entry%:*}"; step="${entry##*:}"
        rm -rf "$CKPT_ROOT/merged_models/$exp/global_step_${step}"
    done
fi

echo ""
date
exit $CORE_RC
