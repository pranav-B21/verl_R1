# Reasoning Reward — iteration v5 ("format-gated dense reward")

**Status:** implemented. v5 keeps v4's dense rank reward and fixes the two
failure modes that made v4's held-out curve **peak ~step 250 then decline below
its start**.

## Why v5 exists

v4 (`../v4/`) made the outcome reward dense — fixing v3's flat-zero gradient — and
trained stably. But the held-out eval (`TEST_OUTPUT.md`, full v4 @ step 250) and
the analysis (`REWARD_REASONING_ANALYSIS.md`, Part 3.3 / Part 4) found the dense
reward was being *gamed*, which is exactly the shape of the curve you see: up to
~step 250, then a decline.

| Symptom @ v4 step 250 | baseline | v3 | **v4** |
|---|---|---|---|
| multiple `<answer>` tags | 0% | 0% | **73.9%** |
| median response length (words) | 754 | 974 | **1203** |

**Root cause of the hack.** The base reward's multi-answer penalty (`match/4`, or
`-0.5`) lives on the float `match` in `reward_SPRec.similarity_match`. But v4
trains on `r_dense = f(rankId)`, computed from the **last** `<answer>` and never
reading `match`. So the only formatting discipline in the pipeline is bypassed —
spamming `<answer>` tags is free under the dense reward. After ~step 250 the policy
discovers this, inflating the temp-1 *train* reward while greedy held-out top-k
*declines*. Verbosity inflation is the same story with no length pressure.

## The change (three things; everything else is v4)

```
r_dense = ( 1 - log(rankId)/log(N) ) ** p          # p = RTHINK_DENSE_P, v5 default 2.0
r_dense = 0                if output is MALFORMED   # A. format gate (anti-hack)
penalty = format_penalty (if malformed) + length_penalty   # A + B, LongPAS-gated
R_total = r_dense + applied_shaping - penalty      # shaping = v3 terms (v4 coupling)
```

**A. Format-integrity gate (the core fix).** `r_dense` is the correctness proxy v4
actually trains on, so v5 only pays it on a *well-formed* answer — exactly one
`<answer>...</answer>`. Anything else (multi-answer spam **or** no answer) zeroes
`r_dense` and subtracts `RTHINK_FORMAT_PENALTY`. This carries the base
multi-answer penalty into the dense term and kills the loophole at its source.

**B. Length discipline.** A gentle, **capped** penalty for responses past a soft
word budget (`RTHINK_LEN_SOFT`), so the dense reward can't be farmed with rambling
`<think>`. Capped at `RTHINK_LEN_CAP` (default 0.2) so it can never overpower
correctness.

**C. HR-targeted dense default (`p = 2.0`).** HR cares only about the extreme top
of the rank distribution. Squaring the curve concentrates reward near the top
while staying **nonzero everywhere**, so it targets top-k *without* reintroducing
v4's dead gradient (which a hard rank floor would):

| target rank | v4 r_dense (p=1) | **v5 r_dense (p=2)** |
|---|---|---|
| 1     | 1.00 | 1.00 |
| 10    | 0.76 | 0.57 |
| 100   | 0.51 | 0.26 |
| 1000  | 0.27 | 0.07 |
| 5000  | 0.10 | 0.01 |

`RTHINK_RANK_FLOOR` stays available (default off) for the harder-targeting ablation.

**LongPAS (kept from v3/v4):** a genuinely correct *and well-formed* answer
(`r_answer >= 0.5`) is never penalised — positive shaping only, no format/length
penalty. Note `r_answer >= 0.5` is only reachable with exactly one answer at rank
<= 5, because the base reward divides a multi-answer `match` by 4 (so 1.0->0.25,
0.8->0.2, both < 0.5). The hack therefore never lands in the protected branch.

## What changed in the code

| File | Change |
|---|---|
| `reward_reasoning/v5/orchestrator.py` | new — v4 dense reward + format gate + length penalty + p=2 default. |
| `reward_reasoning/v5/__init__.py` | new — exports `compute_score`. |
| `reward_reasoning/__init__.py` | `RTHINK_MODE=v5` (alias `5`) dispatches here. |
| `run_in_container_rthink_v5.sh` | training launcher (defaults `RTHINK_MODE=v5`, full v5). |
| `sbatch_run_dual_gpu_rthink.sh` | launch with `RUN_SCRIPT=run_in_container_rthink_v5.sh`. |

`reward_SPRec.py` is **unchanged** — v5 reuses its `return_rank=True` path and
`count_answer_tags`, so baseline/v3/v4 are unaffected.

## Hyperparameters (env vars)

| Var | Default | Meaning |
|---|---|---|
| `RTHINK_DENSE_P`       | `2.0`    | sharpness exponent `p` (v5 raises it from v4's 1.0 to target top-k) |
| `RTHINK_RANK_FLOOR`    | `0`      | zero `r_dense` when `rankId > floor` (0 = off) |
| `RTHINK_DENSE_ONLY`    | `0`      | `1` = no process shaping (ablation); `0` = full v5 |
| `RTHINK_FORMAT_GATE`   | `1`      | enable the malformed-output gate (set `0` to reproduce v4 behaviour) |
| `RTHINK_FORMAT_PENALTY`| `0.5`    | penalty subtracted for malformed output |
| `RTHINK_LEN_SOFT`      | `600`    | word budget before the length penalty starts |
| `RTHINK_LEN_W`         | `0.0005` | penalty per word over the budget |
| `RTHINK_LEN_CAP`       | `0.2`    | maximum length penalty |
| `RTHINK_SCALE/_CAP/_W_*` | v3 defaults | process-shaping knobs (only when `DENSE_ONLY=0`) |

## Logging & how to judge it

Returns a dict; the reward manager logs every key:
`score` (= R_total), **`r_answer` (tiered, baseline-comparable north-star)**,
`r_dense`, `rank`, `r_think` (applied shaping), **`format_ok`** (fraction of
well-formed outputs — watch this climb toward 1.0), `format_penalty`,
`len_penalty`, and the four process components.

**Compare `r_answer` against the baseline's `reward/mean@1`, never `score`** —
`score` includes the dense term and is not comparable across runs. The new
`format_ok` metric is the v5 health check: if it does *not* approach 1.0, the gate
isn't biting and the hack survives.

## What success looks like (vs the v4 failure)

1. `format_ok` → ~1.0 within the first ~50 steps (multi-answer spam dies).
2. Median response length stops growing (no verbosity inflation).
3. Held-out `r_answer`/HR no longer peaks-then-declines around step 250 — the
   train↔val gap (F9) should *narrow* because the train reward can no longer be
   inflated by the hack.
4. Stretch goal (still the open F9 question): held-out HR@5 finally clears the
   baseline (0.007). If it still ties, the binding constraint is the embedding-rank
   proxy validity / temp-1-vs-greedy decoding gap, not the reward shaping — see
   `REWARD_REASONING_ANALYSIS.md` Part 4 next steps.

## Risks to watch

- **Gate too aggressive early.** A cold-start policy may be malformed often;
  `format_penalty=0.5` then dominates and could suppress the dense gradient before
  the policy learns the format. If `format_ok` stays stuck low *and* training
  stalls, lower `RTHINK_FORMAT_PENALTY` (e.g. 0.25) — the `r_dense=0` gate alone
  already removes the hack incentive.
- **Length penalty fighting legitimate reasoning.** It is capped at 0.2 and
  LongPAS-exempt for correct answers, but if it suppresses useful `<think>`, raise
  `RTHINK_LEN_SOFT` or lower `RTHINK_LEN_W`.
- **The proxy is still the proxy (F9).** v5 closes the *hacking* hole; it does not
  by itself prove the embedding-rank reward transfers to greedy held-out HR. That
  remains the open question to settle with matched-decoding eval.
