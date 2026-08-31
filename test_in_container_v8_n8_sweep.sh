#!/bin/bash
# Held-out eval SWEEP for the v8 SELECTION arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8 @ steps 200,250,...,600
#
# Same arm and same code path as test_in_container_v8_n8.sh -- that script
# evaluates ONE checkpoint, this one walks the whole post-takeoff range in a
# single job so every step is decoded back-to-back against one warm retriever.
# Structure mirrors test_in_container_baseline_n8_sweep.sh so the two sweeps
# produce directly overlayable tables.
#
# WHY a sweep rather than the locked step 200
#   v8 trained to step 725 without regressing (see the wandb decomposition
#   below), and the baseline arm now has a 3-decode sweep at 200,250,...,500.
#   So for the first time the two arms can be compared as CURVES over 200-500
#   rather than at a single step. 550/600 are v8-only and characterize this
#   arm's own trajectory past where the control was decoded.
#
# WHAT THE TRAINING CURVE ALREADY SAYS -- read this before the decodes land
#   Held-out val (data.val_files=test.parquet, greedy mean@1), decomposed out of
#   the two wandb segments. val-core is NOT comparable across arms because for
#   v8 it CONTAINS the shaping bonus; r_answer is the outcome-only term and IS
#   comparable to the baseline's val-core:
#
#     step   v8 val-core   v8 r_answer   v8 r_think   is_successor   cover   | baseline val-core
#      200     +0.0341       +0.0039       +0.0342       0.128       0.0488  |    -0.0002
#      300     +0.0424       +0.0056       +0.0395       0.158       0.0502  |    +0.0062
#      400     +0.0431       +0.0049       +0.0419       0.199       0.0513  |    +0.0085
#      500     +0.0504       +0.0063       +0.0464       0.269       0.0530  |    +0.0083
#      600     +0.0520       +0.0073       +0.0472       0.303       0.0540  |      n/a
#      700     +0.0542       +0.0074       +0.0487       0.344       0.0550  |      n/a
#
#   Two things follow, and this sweep exists to test the second one:
#     1. v8's val-core sits ~5x above the baseline's on the dashboard, but ~90%
#        of that gap is r_think -- the shaping constant the arm pays itself.
#        On outcome the arms are level (v8 +0.0074 @700 vs baseline +0.0107
#        @550). Do not read the wandb panel as a win; read r_answer.
#     2. is_successor on HELD-OUT data rose 0.006 -> 0.344 and was still
#        climbing at 700, while r_answer stayed flat. The reward's mechanism
#        transferred; the outcome did not follow. That is either the
#        pre-registered 13-18% cue ceiling binding, or a train/decode gap.
#        `decode_selection.py` per step is what separates those two.
#
# WHAT THIS SWEEP DOES NOT DO
#   It does not license picking a "best" step. val_files=test.parquet, so every
#   curve above is computed on the TEST set and any step chosen by looking at it
#   is selected on test. Report the whole curve, not its argmax. Step 200 stays
#   the pre-registered cross-arm comparison point; 250-500 are iso-step
#   comparisons of equal decode budget (3 vs 3) that were not pre-registered and
#   must be labelled as such.
#
# Usage (submit the sbatch wrapper from a LOGIN node, not this file):
#   sbatch sbatch_run_test_v8_n8_sweep.sh
#   EVAL_REPEATS=1 sbatch sbatch_run_test_v8_n8_sweep.sh                  # quick pass
#   CHECKPOINT_STEPS="600 650 700" sbatch sbatch_run_test_v8_n8_sweep.sh  # past the control
#
# Env vars
#   CHECKPOINT_STEPS  space-separated steps (default "200 250 300 350 400 450 500 550 600").
#                     The 50-step grid is deliberate: 200-500 lines up 1:1 with
#                     the baseline sweep's grid. Checkpoints also exist on a
#                     25-step grid from 275 and out to 725 if you want more.
#                     Steps without a checkpoint on disk are reported and skipped.
#   EVAL_REPEATS      independent decodes per checkpoint (default 3). Decode is
#                     unseeded at temperature=1.0; re-decoding ONE frozen v8
#                     checkpoint gave HR@5 0.005/0.004/0.002 at step 250, so the
#                     within-step spread IS the significance bar and 1 is not a
#                     measurement. ~7 min per decode on this arm.
#   PURGE_MERGED      1 = delete this sweep's merged HF models on the way out.
#                     ~3.8 GB per step, so the default 9 steps leave ~34 GB on
#                     /scratch (default 0; they make a re-decode cheap).
#   CONTROL_SUMMARY_JSON  optional audited control curve. Expected schema:
#                     {"arm":"name","rows":[{"step":200,"hr1":0.1,
#                     "hr5":0.2,"ndcg5":0.15}, ...]}. No control curve is
#                     printed when omitted; historical values are never baked in.
#   Everything test_in_container_rthink.sh accepts (FORCE_MERGE, GEN_BATCH_SIZE,
#   TOOL_CONFIG, EVAL_CATEGORY, ...) works here unchanged.
#
# SAFE TO RUN WHILE v8 IS STILL TRAINING: the checkpoints are frozen on disk and
# sbatch_run_test_rthink.sh writes a job-scoped copy of search_tool_config.yaml
# under /scratch/.../_tool_configs/, so it cannot repoint a running trainer's
# retriever. It does need its own 2 nodes on top of the training job's 2.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

SWEEP_EXPERIMENT_NAME=${EXPERIMENT_NAME:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8"}
EXPERIMENT_NAME="$SWEEP_EXPERIMENT_NAME"
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"200 250 300 350 400 450 500 550 600"}
export EVAL_REPEATS=${EVAL_REPEATS:-3}
PURGE_MERGED=${PURGE_MERGED:-0}
SWEEP_DECODE_TEMPERATURE=${DECODE_TEMPERATURE:-1.0}

CKPT_ROOT="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"
EVAL_ROOT="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME"

# ---------------------------------------------------------------------------
# Resolve the step list against what is actually on disk, and fail BEFORE the
# first decode if none of it exists.
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
echo "# v8 selection arm -- held-out eval SWEEP"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   steps      :$FOUND_STEPS   ($N_STEPS checkpoints)"
echo "#   decodes    : $EVAL_REPEATS per checkpoint, temperature=$SWEEP_DECODE_TEMPERATURE"
echo "#   total      : $((N_STEPS * EVAL_REPEATS)) decodes  (~7 min each)"
echo "#   control    : test_in_container_baseline_n8_sweep.sh, same 50-step grid"
echo "#                over 200-500 at the same 3 decodes/step"
echo "#   iso-step   : 200 is the PRE-REGISTERED comparator; 250-500 are"
echo "#                equal-budget but post-hoc; 550-600 are v8-only"
echo "######################################################################"
echo ""
date

# Sentinel: everything this job writes is newer than this file, which is how the
# summary below tells today's decodes apart from the decode dirs already sitting
# under global_step_200 (2026-08-01) and global_step_250 (2026-08-02).
SENTINEL="$(mktemp -t v8sweep_start.XXXXXX)"
trap 'rm -f "$SENTINEL"' EXIT

# ---------------------------------------------------------------------------
# [1/4] merge -> generate -> eval.py for every step, via the shared core, so
# this arm runs byte-identical evaluation code to every other arm.
# ---------------------------------------------------------------------------
unset EXPERIMENT_NAME   # else the core takes its single-run back-compat branch
bash test_in_container_rthink.sh
CORE_RC=$?
EXPERIMENT_NAME="$SWEEP_EXPERIMENT_NAME"

# ---------------------------------------------------------------------------
# [2/4] Gate 0 (validity) + the coverage branch of Gate 1, per step.
# Gate 0 is per-STEP here, not per-run: retrieval collapse late in training is
# the v7 failure mode, and a sweep is exactly where it would show up. Any step
# whose retr% drops below 95 is DISCARDED, not compared -- an arm that "wins"
# by not retrieving has not won.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [2/4] Decode behavior per checkpoint -- Gate 0 + coverage (Gate 1a)"
echo "######################################################################"
for step in $FOUND_STEPS; do
    STEP_DIR="$EVAL_ROOT/global_step_${step}"
    # Only this job's decodes -- the older 08-01/08-02 decode dirs under steps
    # 200 and 250 would otherwise be folded into today's behavior numbers.
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
        --json "$STEP_DIR/sweep_decode_behavior.json" $PREDS
done

# ---------------------------------------------------------------------------
# [3/4] The selection branch of Gate 1 -- v8's actual hypothesis, per step.
# decode_behavior.py stops at coverage; this is the stage v8 was built to move,
# and the wandb curve says is_successor more than doubled between 200 and 700.
# Whether that shows up here is the whole question the sweep asks.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [3/4] Selection funnel per checkpoint -- Gate 1b"
echo "######################################################################"
for step in $FOUND_STEPS; do
    STEP_DIR="$EVAL_ROOT/global_step_${step}"
    PREDS=$(find "$STEP_DIR" -name test_predictions.json -newer "$SENTINEL" 2>/dev/null | sort)
    echo ""
    echo "--- global_step_${step} ---"
    if [ -z "$PREDS" ]; then
        echo "[warn] no decodes for this step" >&2
        continue
    fi
    # shellcheck disable=SC2086
    python3 verl/utils/reward_score/reward_retrieval/audits/decode_selection.py \
        --json "$STEP_DIR/sweep_decode_selection.json" $PREDS
done

# ---------------------------------------------------------------------------
# [4/4] The curve. Per-decode HR/NDCG rows, then one row per step pooling the
# funnel counts across decodes. Per-decode selection events are ~1-3 out of ~35
# winnable cases -- one binomial draw wide -- so the POOLED column is the only
# one with enough events to read a trend off.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [4/4] Held-out curve across checkpoints (this job's decodes only)"
echo "######################################################################"
EVAL_ROOT="$EVAL_ROOT" SENTINEL="$SENTINEL" STEPS="$FOUND_STEPS" \
CONTROL_SUMMARY_JSON="${CONTROL_SUMMARY_JSON:-}" python3 - <<'PY'
import json, math, os, statistics as st

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
    # EVAL_REPEATS<=1 takes the core's untagged path and writes the metrics at
    # the step-dir root instead of in a decode<i>_<date> subdir, so scan both.
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

# --- the funnel, pooled across this step's decodes -----------------------
# Read back what [2/4] and [3/4] wrote instead of re-parsing the decodes: same
# numbers, and it keeps the audit scripts the single definition of each metric.
print()
print("funnel per step, POOLED over decodes (Gate 0 | Gate 1a | Gate 1b):")
hdr = (f"{'step':>6} {'retr%':>6} {'ans%':>6} {'d/ret':>6} {'q/call':>6} | "
       f"{'cover%':>7} | {'succ%':>6} {'rep%':>5} {'|succ|':>6} {'GTinSucc':>9} "
       f"{'pickGT':>10}")
print(hdr)
print("-" * len(hdr))
for step in sorted(steps):
    sd = os.path.join(root, f"global_step_{step}")
    bpath = os.path.join(sd, "sweep_decode_behavior.json")
    spath = os.path.join(sd, "sweep_decode_selection.json")
    if not (os.path.isfile(bpath) and os.path.isfile(spath)):
        continue

    with open(bpath) as fh:
        b = json.load(fh)
    b = [r for r in (b if isinstance(b, list) else b.get("rows", [])) if r.get("n")]
    if not b:
        continue
    wmean = lambda k: sum(r[k] * r["n"] for r in b) / sum(r["n"] for r in b)

    with open(spath) as fh:
        s = json.load(fh)
    s = s.get("rows", s) if isinstance(s, dict) else s
    tot = lambda k: sum(int(r.get(k, 0)) for r in s)
    n_all, n_cov = tot("n"), tot("n_cov")
    pick, cov_succ = tot("n_pick_gt"), tot("n_cov_succ")
    pct = lambda a, b_: (100.0 * a / b_) if b_ else float("nan")
    items = st.mean([r["succ_mean"] for r in s]) if s and "succ_mean" in s[0] else float("nan")

    print(f"{step:>6} {100*wmean('retrieval_rate'):>6.1f} {100*wmean('answer_rate'):>6.1f} "
          f"{wmean('docs_per_retrieval'):>6.2f} {wmean('queries_per_call_mean'):>6.2f} | "
          f"{pct(n_cov, n_all):>7.2f} | {pct(tot('n_successor'), n_all):>6.1f} "
          f"{pct(tot('n_repeat'), n_all):>5.1f} {items:>6.2f} "
          f"{pct(cov_succ, n_cov):>8.1f}% {pick:>4}/{n_cov:<5}")

print()
print("How to read this, in the pre-registered order:")
print("  Gate 0  retr% >= 95, ans% >= 95, d/ret >= 3. A step that fails is")
print("          DISCARDED, not compared -- v7 'tied' the baseline at 5% retr%.")
print("  Gate 1  coverage >= 4.0%  OR  pickGT/cover >= 15% (chance ~4-5%, since")
print("          a rollout retrieves ~14 items). Pooled over 3 decodes there are")
print("          only ~100 winnable cases per step, so a 1-2 event move is not a")
print("          trend; look for monotonicity across steps, not step-to-step.")
print("  Gate 2  HR@5. A step-to-step move smaller than the within-step spread")
print("          above is not a result. Prior bar on this arm: three decodes of")
print("          the FROZEN step-250 checkpoint gave HR@5 0.005/0.004/0.002.")
print()
print("The specific question this sweep answers: wandb says is_successor climbed")
print("0.128 -> 0.344 between steps 200 and 700 on held-out data while r_answer")
print("stayed ~flat at +0.005-0.008. If succ% here tracks that climb and pickGT")
print("stays at chance, the successor cue is exhausted (its pre-registered")
print("ceiling was coverage 3.2% x GT-in-cue 13-18% ~ HR@1 0.006 at PERFECT")
print("selection) and v8 has been answered -- the next iteration needs a")
print("different cue, not a bigger weight on this one. If succ% here is well")
print("BELOW the wandb curve, that is a train/decode gap (greedy val vs")
print("temperature-1.0 decode) and the reward has not actually been tested.")
print()
control_path = os.environ.get("CONTROL_SUMMARY_JSON", "").strip()
if not control_path:
    print("Control curve omitted (set CONTROL_SUMMARY_JSON to an audited arm summary).")
else:
    with open(control_path) as fh:
        control = json.load(fh)
    control_rows = control.get("rows", []) if isinstance(control, dict) else []
    required = {"step", "hr1", "hr5", "ndcg5"}
    if not control_rows or any(not required.issubset(row) for row in control_rows):
        raise ValueError(f"invalid control summary schema: {control_path}")
    arm = control.get("arm", os.path.basename(control_path))
    print(f"Control curve to overlay ({arm}; loaded from {control_path}):")
    print(f"{'step':>6} {'HR@1':>9} {'HR@5':>9} {'NDCG@5':>9}")
    for row in sorted(control_rows, key=lambda value: int(value["step"])):
        print(f"{int(row['step']):>6} {float(row['hr1']):>9.5f} "
              f"{float(row['hr5']):>9.5f} {float(row['ndcg5']):>9.5f}")
PY

# ---------------------------------------------------------------------------
# Optional cleanup. Merged HF models are ~3.8 GB per checkpoint (~34 GB for the
# default 9 steps) and are what makes a re-decode cheap, so they are kept unless
# you ask otherwise.
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
echo "Decodes live under $EVAL_ROOT/global_step_<step>/decode<i>_<date>/"
echo "Per-step audit JSON: <step dir>/sweep_decode_{behavior,selection}.json"
echo "Append the per-step means to TEST_OUTPUT.md (workspace root)."
exit $CORE_RC
