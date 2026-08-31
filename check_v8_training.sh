#!/bin/bash
# Status check for the v8 selection-reward TRAINING run (resume segments).
#
#   bash check_v8_training.sh          # auto-detects the running job
#   bash check_v8_training.sh 883844   # or pin a job id
#
# Safe to run any time, from anywhere, while the job is queued, running, or
# finished. Read-only: it inspects SLURM, the job log, the checkpoint dir, and
# the local .wandb datastore. It never touches the run.
#
# Section 2 is the one that matters on a RESUME. Three failures are silent --
# they produce a healthy-looking run that answers a different question:
#   * RTHINK_MODE unset  -> launcher defaults to v7, so the v8 checkpoint keeps
#     training under the v7 reward. No error is raised anywhere.
#   * EXPERIMENT_NAME typo -> resume_mode=auto finds no checkpoint and silently
#     starts from step 0 in a NEW directory.
#   * RTHINK_W_GROUND changed mid-run -> segment 2 optimizes a different reward
#     than segment 1 and the merged curve means nothing.
# All three are asserted below against the log itself, not against what the
# submit command was supposed to contain.

set -u
PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

EXP="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8"
CKPT_ROOT="/scratch/11138/pranavbelligundu/verl/$EXP"
LOG="${LOG:-output_rthink_v8_n8_resume.log}"
JOB="${1:-}"

echo "=============================================================="
echo "1. JOB"
echo "=============================================================="
if [ -z "$JOB" ]; then
    JOB=$(squeue -u "$USER" -h -n sbatch_run_dual_gpu_rthink.sh -o "%i" 2>/dev/null | tail -1)
fi
if [ -n "$JOB" ]; then
    squeue -j "$JOB" -o "%.10i %.30j %.9T %.11M %.11L %.5D %R" 2>/dev/null \
        || echo "  job $JOB is no longer in the queue"
    sacct -j "$JOB" -X -n -o "JobID%10,State%12,Elapsed%12,End%20" 2>/dev/null | sed 's/^/  /'
else
    echo "  (no training job in the queue -- showing on-disk state only)"
fi

echo
echo "=============================================================="
echo "2. WIRING  -- must all be OK before any curve below is meaningful"
echo "=============================================================="
if [ ! -f "$LOG" ]; then
    echo "  !! log not found: $LOG   (set LOG=<file> if you renamed it)"
else
    age_min=$(( ( $(date +%s) - $(stat -c%Y "$LOG") ) / 60 ))
    echo "  log      : $LOG  ($(stat -c%s "$LOG") bytes, last write ${age_min}min ago)"
    [ "$age_min" -gt 45 ] && echo "             !! no writes in ${age_min}min -- job may be wedged"

    # Reward version: read it off the reward's own debug print, which reports the
    # weights it actually used. This is the ground truth; the submit env is not.
    v8n=$(grep -ac "\[Rthink-v8\]" "$LOG" 2>/dev/null); v8n="${v8n:-0}"
    v7n=$(grep -ac "\[Rthink-v7\]" "$LOG" 2>/dev/null); v7n="${v7n:-0}"
    if [ "$v8n" -gt 0 ]; then
        echo "  reward   : v8 OK  ($v8n sampled prints)"
        echo "             $(grep -a "\[Rthink-v8\]" "$LOG" | tail -1 | grep -o "(w_sel=.*)")"
        # Segment 1 ran at w_grnd=0.4 (launcher export overrides v8's own 0.1
        # default). Segment 2 must match it or the curve splices two rewards.
        grep -a "\[Rthink-v8\]" "$LOG" | tail -1 | grep -q "w_grnd=0.4" \
            || echo "             !! w_grnd differs from segment 1 (0.4) -- curves are NOT comparable"
    elif [ "$v7n" -gt 0 ]; then
        echo "  reward   : !! v7 PRINTS FOUND -- RTHINK_MODE was not set to v8."
        echo "             The v8 checkpoint is being trained under the v7 reward. Kill and resubmit."
    else
        echo "  reward   : (no reward prints yet -- still initializing)"
    fi

    grep -am1 "\[pre-flight\]" "$LOG" | sed 's/^/  geometry : /'
    grep -am1 "rollout_n" "$LOG" | grep -o "'rollout_n': [0-9]*" | sed 's/^/  /'

    # Resume assertion: verl logs the checkpoint it loads. Starting from 0 on a
    # resume means it did not find the directory -- usually an EXPERIMENT_NAME typo.
    res=$(grep -am1 "Resuming from\|resume from\|load from checkpoint folder\|Checkpoint loaded" "$LOG" | cut -c1-160)
    echo "  resume   : ${res:-(not yet logged)}"

    # Failure signatures only. "retriever" alone matches the normal startup
    # chatter ("Starting retriever on ...") and cries wolf on every healthy run.
    for pat in "Traceback" "CUDA error" "out of memory" "pidfd_getfd" \
               "Error 803" "Retriever failed" "did not become ready" \
               "Restarting retriever" ; do
        n=$(grep -aci "$pat" "$LOG" 2>/dev/null); n="${n:-0}"
        [ "$n" -gt 0 ] 2>/dev/null && echo "  !! $pat  x$n"
    done
fi

echo
echo "=============================================================="
echo "3. CHECKPOINTS  (save_freq=50)"
echo "=============================================================="
ls -1dt "$CKPT_ROOT"/global_step_* 2>/dev/null | head -6 | sed 's/^/  /' || echo "  (none)"
echo "  latest_checkpointed_iteration.txt: $(cat "$CKPT_ROOT/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')"
echo "  NOTE: a wall-time kill loses everything since the last multiple of 50."

echo
echo "=============================================================="
echo "4. PROGRESS + MECHANISM  (from the local .wandb datastore)"
echo "=============================================================="
# The run logs to wandb online-only, so the console carries no numbers; the
# .wandb binary beside it holds every logged step. Newest one is this training
# run (eval jobs do not write wandb).
#
# Each SLURM segment gets its OWN local directory even though WANDB_RUN_ID pins
# them to one server-side run, so a freshly submitted resume shows the PREVIOUS
# segment's datastore until it writes its first step (~15min: container start,
# sglang init, checkpoint load). Say which segment is on screen -- otherwise the
# old segment's final step reads as live progress.
WB=$(ls -1t wandb/run-*/run-*.wandb 2>/dev/null | head -1)
if [ -z "$WB" ]; then
    echo "  (no .wandb datastore found)"
else
    wb_age_min=$(( ( $(date +%s) - $(stat -c%Y "$WB") ) / 60 ))
    echo "  datastore: $WB  (last write ${wb_age_min}min ago)"
    if [ "$wb_age_min" -gt 20 ]; then
        echo "             ^^ STALE: this is the PREVIOUS segment's datastore."
        echo "                The current job has not logged a step yet; numbers"
        echo "                below are where the last segment ended."
    fi
    TMP=$(mktemp /tmp/v8_check_XXXX.csv)
    python3 verl/utils/reward_score/reward_reasoning/diagnostics_scripts/wandb_dump.py \
        --csv "$TMP" "$WB" >/dev/null 2>&1
    python3 - "$TMP" <<'PY'
import csv, sys, time
rows = {}
for r in csv.DictReader(open(sys.argv[1])):
    s = r.get('training/global_step') or r.get('_step')
    try: s = int(float(s))
    except (TypeError, ValueError): continue
    rows.setdefault(s, {}).update({k: v for k, v in r.items() if v not in (None, '')})
if not rows:
    print("  (datastore empty -- run just started)"); raise SystemExit
steps = sorted(rows)
last = steps[-1]

def f(s, k):
    v = rows.get(s, {}).get(k)
    try: return float(v)
    except (TypeError, ValueError): return None

# steps/hour over the most recent stretch that has wall-clock stamps
rate = None
have = [s for s in steps if f(s, '_timestamp')]
if len(have) >= 6:
    a, b = have[-6], have[-1]
    dt = f(b, '_timestamp') - f(a, '_timestamp')
    if dt > 0: rate = (b - a) / (dt / 3600.0)

print(f"  step     : {last}   (epoch {f(last,'training/epoch') or 0:.2f} of 22; 73 steps/epoch)")
if rate:
    print(f"  rate     : {rate:.1f} steps/h  -> ~{int(rate*48)} steps per 48h segment")
    print(f"  next ckpt: step {((last//50)+1)*50} in ~{((((last//50)+1)*50)-last)/rate:.1f}h")

print("\n  TRAIN (watch for the v5 pattern: train reward climbing, held-out flat)")
for k, lbl in [('critic/score/mean', 'reward'), ('actor/entropy', 'entropy'),
               ('response_length/mean', 'resp len'), ('actor/grad_norm', 'grad norm'),
               ('response/aborted_ratio', 'aborted')]:
    pts = [s for s in steps if f(s, k) is not None][-5:]
    if pts:
        print(f"    {lbl:9s}" + "  ".join(f"{s}:{f(s,k):.3f}" for s in pts))

print("\n  HELD-OUT val-aux (NB: val_files=test.parquet -- these ARE the test set)")
vk = [('cover', 'cover*'), ('is_grounded', 'grounded'), ('is_successor', 'successor'),
      ('is_repeat', 'repeat'), ('r_answer', 'r_answer'), ('best_sim', 'best_sim'),
      ('n_retrieved_items', 'n_items')]
vsteps = [s for s in steps if f(s, 'val-aux/amazon_test/r_answer/mean@1') is not None]
if not vsteps:
    print("    (no validation yet -- test_freq=50)")
else:
    print("    step     " + " ".join(f"{s:>8d}" for s in vsteps))
    for key, lbl in vk:
        vals = [f(s, f'val-aux/amazon_test/{key}/mean@1') for s in vsteps]
        print(f"    {lbl:9s}" + " ".join(f"{v:8.4f}" if v is not None else "       -" for v in vals))
    print("\n    *cover is NOT coverage: 1.0 on an exact GT hit, else 0.3 x cosine.")
    print("     It upper-bounds true coverage; subtract a ~0.023 soft tail to estimate it.")
    print("    Convergence read: successor/grounded plateauing = the cue is spent.")
    print("    repeat RISING is the w_grnd=0.4 artifact (grounded+repeat nets +0.01).")
    print("    r_answer is the only held-out OUTCOME number here; it was 0.0039 @200.")
PY
    rm -f "$TMP"
fi

echo
echo "=============================================================="
echo "5. WHAT WOULD COUNT AS CONVERGED"
echo "=============================================================="
cat <<'EOF'
  Decode step 300 (pre-registered as the 2nd and final comparison point) with:
      sbatch sbatch_run_test_v8_n8.sh        # CHECKPOINT_STEP=300 to override
  Reference @200, pooled: coverage 3.4%, pick GT|covered 2/34 (chance 6.9%),
  successor 10.6%, repeat 7.2%, HR@1 0.0016 / HR@5 0.0034 over 5 decodes.
  The claim is Gate 1 (mechanism), not HR: at ~3% coverage, HR cannot resolve it.
  Do NOT pick the checkpoint where val-aux peaks -- val IS the test set.
EOF
