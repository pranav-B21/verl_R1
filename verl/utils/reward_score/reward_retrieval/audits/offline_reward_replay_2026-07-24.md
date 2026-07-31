# A0 offline reward replay — results (2026-07-24)

Ran `offline_reward_replay.py` inside the sglang container (Python 3.12, CPU,
`HF_HUB_OFFLINE`) on the step-200 eval predictions, with the **M3 training
hyperparameters** (τ=0.20, floor=0.25, W_RETQUAL=0.6, SCALE=0.10, CAP=0.08,
LEN_SOFT=600, LEN_W=0.0005, LEN_CAP=0.2). Reports
`E[r_total | retrieve] − E[r_total | abstain]` under the current (buggy)
`len_penalty` vs the Fix-1-patched one (excludes `<tool_response>` doc text).

## Command

```bash
module load tacc-apptainer
apptainer exec --bind /work:/work --bind /scratch:/scratch --pwd $PWD \
  sglang_25.10-py3-tls-fixed.sif bash -c '
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH"|tr ":" "\n"|grep -v cuda/compat|paste -sd ":" -)
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export HF_HOME=/scratch/11138/pranavbelligundu/huggingface HF_HUB_OFFLINE=1
    export RTHINK_RETQUAL_TAU=0.20 RTHINK_RETQUAL_FLOOR=0.25 RTHINK_W_RETQUAL=0.6 \
           RTHINK_W_COVGAIN=0.0 RTHINK_SCALE=0.10 RTHINK_CAP=0.08 RTHINK_HIT_K=10 \
           RTHINK_FORMAT_GATE=1 RTHINK_FORMAT_PENALTY=0.5 RTHINK_LEN_SOFT=600 \
           RTHINK_LEN_W=0.0005 RTHINK_LEN_CAP=0.2 RTHINK_RETRIEVAL_ONLY=0
    python3 verl/utils/reward_score/reward_retrieval/audits/offline_reward_replay.py \
        --preds <baseline@200 preds> <v7@200 preds> --data-source amazon --limit 350'
```

## Results

| Data (τ=0.20) | config | retr | abst | len_pen gap (buggy→patched) | **gap buggy** | **gap patched** |
|---|---|---|---|---|---|---|
| v7@200 (within-policy) | LS600 SC0.10 CAP0.08 | 50 | 950 | 0.0014 → ~0 | −0.0019 | **−0.0005** |
| baseline+v7 pooled (long regime) | LS600 SC0.10 CAP0.08 | 369 | 331 | 0.172 → 0.138 | −0.2312 | **−0.1969** |
| baseline+v7 pooled | LS1200 SC0.10 CAP0.12 | 350 | ~330 | — → 0.046 | −0.1466 | **−0.1050** |

Supporting means (pooled, LS600): retqual(retrieve)=+0.046, applied shaping
=+0.0028 (capped), best_sim=0.233, r_answer retrieve 0.0014 **<** abstain 0.0049.

## Verdict — the gate FAILS under Fix 1 alone; the diagnosis is refined

1. **Root cause confirmed AND broadened.** The length penalty does punish
   retrieving rollouts (retrieve−abstain `len_penalty` gap 0.172, total gap
   −0.23). But stripping the retriever's `<tool_response>` docs (Fix 1) removes
   only ~0.03 of that 0.17 gap. **The dominant anti-retrieval force is the length
   penalty on the model's OWN reasoning** — a retrieving rollout reasons twice
   (pre- and post-retrieval, ~900–1000 words) vs an abstaining rollout's single
   ~600-word block, and both blow past the 600-word `LEN_SOFT`, saturating near
   the 0.2 cap. Fix 1 is **necessary but insufficient.**
2. **Raising the word budget is what closes the length gap.** `LEN_SOFT` 600→1200
   collapses the patched `len_penalty` gap 0.138 → 0.046 and the total gap
   −0.197 → −0.105. So the length term must budget for multi-turn reasoning
   (raise/scale `LEN_SOFT` with turns, or exempt tool-turn rollouts), not just
   drop the doc text.
3. **`r_retqual` is positive at τ=0.20 (+0.046) but ~70× too weak** (applied
   +0.0028, capped at 0.08). Fix 2 (raise SCALE/CAP) is still needed for the
   residual — but CAP cannot exceed ~0.1 without the shaping dominating the
   outcome-tier reward, so length must be fixed structurally, not out-bid.
4. **The end-state (step-200) data is confounded:** the collapsed policy is terse
   (len_penalty ≈ 0), and the ~50 surviving retrievers are a biased residue
   (`r_answer` 0.0014 < abstain 0.0049 — harder prompts). A clean causal
   measurement needs the **pre-collapse ckpt-100/150 decodes (A2, short GPU
   decode)**, where retrieval was healthy and rollouts were long.

## Implication for the fix (needs a design call)

Fix 1 (already coded) stays — docs should never count. But the length-penalty
needs a **multi-turn-aware redesign**; candidate approaches:
- (a) **Exempt tool-turn rollouts** from `len_penalty` (zero it when n_turns>0) —
  decisive, but drops anti-rambling on retrieving rollouts.
- (b) **Scale `LEN_SOFT` with turns** (e.g. 600 + 600·n_turns) — preserves
  anti-rambling per reasoning block; keeps the doc-strip.
- (c) **Raise `LEN_SOFT` to ~1200–1500** flat + doc-strip — simplest; LS1200
  already cuts the len gap to 0.046.
Then a **modest Fix 2** (SCALE↑, CAP≤0.1) for the residual retqual signal.

## Update — turns-aware budget implemented + validated (INCR=400)

Marginal generated-word cost per credited turn (pooled 9 step-200 decodes, docs
stripped): abstain (0 turns) median **239** words (p90 402); +1 turn median
**999** (p90 1566); +2 median 1288; OLS slope ≈ **589 words/turn**; p90 marginal
≈ 1303, median 733. So a literal `INCR=p90=1303` → budget(k=1)=1903 > p90 length
→ **near-exempts retrieval** (the cliff). `INCR=400` puts budget(k=1)=1000 ≈ the
k=1 median, so the typical retrieving rollout sits at budget and only the
ramblier tail pays.

`orchestrator.py` now: (1) strips `<tool_response>` from the word count;
(2) budget = `LEN_SOFT + LEN_PER_TURN·min(credited_turns, LEN_TURN_CAP)` where a
credited turn is one that RETURNED docs (bare tool-call spam buys nothing).
Defaults `LEN_PER_TURN=400`, `LEN_TURN_CAP=3` (env-overridable). Replay with the
new code (pooled baseline+v7, τ=0.2, INCR=400):

| term | buggy (orig M3) | patched (v7b) |
|---|---|---|
| retrieve `len_penalty` | 0.173 | **0.063** |
| retrieve−abstain gap | −0.217 | **−0.107** |

Gap still <0 on this data, residual = ~0.06 length (baseline's rollouts are ~1000
words even at k=1 — INCR=400 budget is right at their median) + ~0.005 `r_answer`
(pooling baseline-retrievers vs v7-abstainers, a policy-mismatch artifact, not a
real effect). Both dissolve on **same-policy pre-collapse data (A2, ckpt 100/150
decodes)** — the clean gate. Fix-2 magnitude to be sized on that data.

