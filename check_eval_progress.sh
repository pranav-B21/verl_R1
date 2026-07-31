#!/bin/bash
# Status check for the M3 A/B eval jobs (test_v7_n8 / test_baseline_n8).
#
#   bash check_eval_progress.sh            # both arms
#   bash check_eval_progress.sh 861993     # one job id
#
# Safe to run any time, from anywhere, while the jobs are queued or running.
# Read-only: it inspects SLURM, the job logs, and whatever predictions exist.

set -u
PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

echo "=============================================================="
echo "1. JOBS"
echo "=============================================================="
squeue -u "$USER" -o "%.10i %.20j %.9T %.11M %.11L %.5D %R" 2>/dev/null
echo
sacct -u "$USER" -S "$(date +%F)" --format=JobID%10,JobName%20,State%12,Elapsed,NodeList%22 2>/dev/null \
  | grep -E "test_v7_n8|test_baseline_n8|JobID|^---" | grep -v '\.' || echo "(no eval jobs today yet)"

echo
echo "=============================================================="
echo "2. LOGS  (each job writes its OWN %j log -- check the id matches)"
echo "=============================================================="
shopt -s nullglob
for log in output_test_v7_n8_[0-9]*.log output_test_baseline_n8_[0-9]*.log; do
    # Only [0-9]* logs are matched above, so these are always %j job logs -- never
    # the pre-%j names (output_test_rthink.log, output_test_baseline_n8_s200.log)
    # that a previous run left behind. Flag anything stale anyway.
    age_min=$(( ( $(date +%s) - $(stat -c%Y "$log") ) / 60 ))
    stale=""
    [ "$age_min" -gt 120 ] && stale="   <-- NOT updated in ${age_min}min; is this job still running?"
    echo "--- $log  ($(stat -c%s "$log") bytes, mtime $(stat -c%y "$log" | cut -d. -f1))$stale"

    # Wiring: these three must appear, in this order, before generation starts.
    grep -am1 "\[entrypoint\]"   "$log" | sed 's/^/    /' || true
    grep -am1 "\[tool-config\]"  "$log" | sed 's/^/    /' || true
    grep -am1 "Retriever is ready\|Starting evaluation" "$log" | sed 's/^/    /' || true

    # Failure modes seen before, cheapest first.
    for pat in "can't open file" "Retriever failed to start" \
               "CUDA error" "out of memory" "Error 803" "Traceback"; do
        # grep -c prints "0" AND exits 1 on no match, so a `|| echo 0` fallback
        # would emit "0\n0" and break the numeric test. Swallow the status instead.
        n=$(grep -ac "$pat" "$log" 2>/dev/null); n="${n:-0}"
        [ "$n" -gt 0 ] 2>/dev/null && echo "    !! $pat  x$n"
    done

    # Progress markers from test_in_container_*.sh.
    for step in "1/4" "2/4" "3/4" "4/4"; do
        grep -am1 "\[$step\]" "$log" | sed 's/^/    /' || true
    done
    echo
done
[ -z "$(echo output_test_v7_n8_[0-9]*.log output_test_baseline_n8_[0-9]*.log)" ] && echo "(no eval logs yet -- jobs still pending)"

echo "=============================================================="
echo "3. RESULTS so far"
echo "=============================================================="
for exp in nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8 \
           nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8; do
    short="${exp#nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-}"
    for l in outputs/eval/$exp/global_step_200/*/test_eval.log; do
        hr1=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$l" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n 1p)
        hr5=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$l" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n 2p)
        printf "  %-12s %-22s HR@1=%-8s HR@5=%-8s\n" \
            "$short" "$(basename "$(dirname "$l")")" "${hr1:-pending}" "${hr5:-pending}"
    done
done

echo
echo "=============================================================="
echo "4. RETRIEVAL BEHAVIOR  (the axis that decides this A/B)"
echo "=============================================================="
preds=$(ls outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-{rthink-v7-n8,baseline-n8}/global_step_200/*/test_predictions.json 2>/dev/null)
if [ -n "$preds" ]; then
    # shellcheck disable=SC2086
    PYTHONPATH="$PROJECT_DIR" python3 \
        verl/utils/reward_score/reward_retrieval/audits/decode_behavior.py $preds
else
    echo "(no predictions yet)"
fi
