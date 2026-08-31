# RRCM Journal Extension — Reward Roadmap (v7)

> **STATUS 2026-08-30 — historical. This file is no longer the source of truth for strategy.**
> Its target ("a reward that beats RRCM's published outcome-only ranking reward") was tested
> through v8 and **not achieved**; the reward axis is closed by measurement, not by opinion
> (`REWARD_REASONING_ANALYSIS.md` Parts 10–15). Current strategy lives in
> `RRCM_NIPS_REWARD_REASONING_RETRIEVAL.md` (the paper section) and
> `RRCM_FROZEN_PROTOCOL.md` (the paper-faithful control). The "A2 pre-collapse decodes
> running" note below refers to July 2026 jobs that finished long ago.

**Single source of truth** for strategy, status, target, and protocol. The two design docs
(`retrieval_reward_design.md`, `reasoning_reward_design.md`) cover *implementation only* and defer to
this file for the frame — nothing strategic is duplicated there, so it can't drift again.

**Contribution:** a structured **retrieval + reasoning reward** that beats RRCM's published
outcome-only ranking reward, on the **identical RRCM system** (dual-memory corpus, retriever,
grounding, GRPO). Nothing about the corpus or retrieval substrate changes — the Table-1 comparison
only holds if the system underneath is identical. Last updated **2026-07-25** (M3 read at
step 200: retrieval **collapsed**; root cause found, fix implemented + offline-audited; A2
pre-collapse decodes running).

---

## STATUS (read this first)

### 2026-07-25 UPDATE — M3 read, collapsed, fix in progress (supersedes the 07-21 block below)

**M3 result (❌).** v7 read at step 200 vs baseline-n8: **retrieval policy collapsed to ~5%**
(baseline ~99%), HR a tie inside decode noise. Real collapse, not a dead retriever
(docs/ret ~3 on the rollouts that did retrieve). Full writeup: root
`REWARD_REASONING_ANALYSIS.md` **Part 8** + `TEST_OUTPUT.md`.

**Root cause (two independent collapse pressures, both now fixed).**
1. **Length penalty** — `len_penalty` counted the retriever's `<tool_response>` docs AND the
   model's own second (post-retrieval) reasoning block against a flat 600-word budget, so a
   retrieving rollout ate ~0.2 while `r_retqual` paid only ~+0.004 → retrieve ≈ −0.2 vs
   abstain ≈ 0. **Fix:** `orchestrator.py` now (a) strips `<tool_response>` from the word
   count and (b) grows the budget by `RTHINK_LEN_PER_TURN` per **doc-returning** turn (capped
   at `RTHINK_LEN_TURN_CAP=3`; spam-proof). Increment sized empirically (abstain 239 gen-words,
   +1 turn 999; p90 marginal 1303 would near-exempt, so INCR is set below it; placeholder 400,
   **pending A2**).
2. **Junk-retrieval penalty** — removed per **advisor Shijun Li (2026-07-25)**. `r_retqual` is
   now **ONE-SIDED**: below-τ retrievals score **0, never negative** (`retrieval.py::_retqual`;
   `RTHINK_RETQUAL_FLOOR` now **INERT**). Rationale: the term scores only the *collaborative*
   axis (cosine to GT next item); a below-τ retrieval is often legitimate item-*attribute*
   lookup, so penalizing it suppresses attribute retrieval. **This supersedes the "τ RE-CALIBRATED
   0.30→0.20 / floor as collapse guard" material below and in §4a** — τ is now only a boost
   threshold; the floor is gone by design.

**Audit-before-train.** New tool `audits/offline_reward_replay.py` re-scores logged rollouts
under old vs new reward. It confirmed the mechanism and caught that the doc-strip alone was
insufficient (retrieve `len_penalty` 0.173→0.138); the turns-aware budget takes it to 0.063.
Details: `audits/offline_reward_replay_2026-07-24.md`.

**Pending → v7b.** A2 = short GPU decodes of pre-collapse ckpts 100/150 (jobs 865934/865935) to
lock `LEN_PER_TURN`, re-run the gate with one-sided retqual, and size Fix 2 (retqual SCALE/CAP,
≤0.1). Then **v7b**: one-sided retqual + turns-aware length release + **more steps** +
**multi-node** training (Shijun: 200 steps is undertrained), plus the still-unrun
`RTHINK_RETRIEVAL_ONLY=1` parity ablation. **Judging protocol:** compare held-out at
convergence (not training curves); ≥3 decode seeds beating the spread; retrieval-side coverage
as the dev metric, HR@5 as the paper headline.

### Where we are right now (2026-07-21) — HISTORICAL; see the 07-25 update above

**M0 ✅ · M1 🔄 · M2 ✅ · M3 ❌ read (collapsed) → fix in progress · M4 ⬜ · M5 ⬜**

| | |
|---|---|
| **Comparison step** | **LOCKED at 200** for every arm. The baseline died at wall-time with a checkpoint at `global_step_200`; nothing is resumed to 300, so every arm is read at 200. |
| **M1 baseline** | `...-gpu-baseline-n8`, job `848159`, trained to step 220 (ckpt 200), outcome-only `reward_SPRec`, `n=8` verified. Eval decode for the HR@5 bar is job `854449`. **Open: whether the step-200 lock changes the 3-seed requirement — user/PI call, not assumed.** |
| **M2** | ✅ **DONE 2026-07-20.** `r_covgain` → `W=0` **permanent**. Top-k 3→20 carved out as its own single-variable experiment, explicitly **not** folded into M3. |
| **M3** | ❌ **READ 2026-07-24 — retrieval COLLAPSED** (see the 07-25 update above). Launched SLURM `854501` (`RTHINK_MODE=v7`, `r_retqual` only, `W_COVGAIN=0`, τ=0.20, `n=8`), hit wall at step 236, read at ckpt 200: retr% ~5% vs baseline ~99%. Fix implemented + offline-audited; v7b pending A2. |
| **Arm asymmetry (accepted)** | The baseline trained **before** the `tool_call_repair` JSON fix (landed 2026-07-20 18:55); M3 trains **with** it → ~2% of rollouts now retrieve where they previously got nothing. Accepted rather than burning a window on a re-baseline; the `RTHINK_RETRIEVAL_ONLY=1` ablation is the parity-matched control when it runs. **Disclose this in any writeup.** |

⚠️ **Operational: concurrent jobs race on the shared tool config.** Every launcher
`sed -i`s `retrieval_service_url` in the single shared
`examples/sglang_multiturn/config/tool_config/search_tool_config.yaml` and restores a backup on
exit, so two overlapping jobs repoint each other's retriever. **Fixed for
`sbatch_run_dual_gpu_rthink.sh` only** — it now copies the config to
`/scratch/.../verl/_tool_configs/search_tool_config.$SLURM_JOB_ID.yaml`, edits only that, and
exports `TOOL_CONFIG` (consumed by `run_in_container_rthink.sh`); both echo a `[tool-config]` line.
`sbatch_run_test_rthink.sh` and the `*_good.sh` launchers are **still racy** — check before running
any two jobs concurrently. (This bit for real: `854449` and `854501` overlapped.)

### Standing facts

| Fact | State |
|---|---|
| **Corpus** | FROZEN — PI-confirmed the ~4.6% GT-in-docs rate is correct (leakage otherwise). No corpus work. |
| **GRPO group size** | **DONE — `n=8` is set** (`run_in_container_rthink.sh`, overridable via `ROLLOUT_N`). Was `n=1` in the original code (a bug the PI confirmed and had already fixed to 8 in his own runs). Escalation ladder if under-performing: 8 → 12 → 16 — all clean at the current batch geometry, no edit needed. |
| **Batch geometry** | **DONE — `train_batch=56`, `ppo_mini_batch=56`, `n=8`** → 448/56 = 8 clean mini-batches. ⚠️ **This roadmap previously prescribed `ppo_mini_batch=64`, which CRASHES**: `verl/workers/config/actor.py:153` raises when `train_batch(56) < ppo_mini_batch(64)`. And the divisibility rule that matters is **unasserted** anywhere in verl (`dp_actor.py:388` splits with a non-strict chunker), so a bad combo trains a ragged final mini-batch with a mis-scaled loss and no error. The launcher now pre-flights all three constraints and fails loudly. |
| **v2–v6 history** | **DONE** -- Confounded — all ran at `n=1` (GRPO baseline disabled). "Reward shaping never beat baseline" is **not** valid evidence against shaping. PI acknowledged this bug was in the original code. |
| **Baseline** | The valid baseline is `n=8`. The local baseline was `n=1`; a later one was `n=5` (died at step 150); **neither is a valid comparator.** Re-run at `n=8` — done (`848159`, ckpt 200); the **3-seed** part is open pending a user/PI call on whether the step-200 lock changes it. `run_in_container_baseline.sh` now **refuses to run** (it silently trained at n=1, producing an invalid comparator that looked legitimate in WandB); use `USE_RTHINK=0` on `run_in_container_rthink.sh` for an identical substrate. |
| **Retrieval axis (`r_retqual`)** | **VALIDATED (lead).** Audit B: AUC 0.85, pearson +0.112, 15× quartile separation. PI endorsed the design ("good design… I would expect this reward design will work"). Implemented and **executed end-to-end offline** (M0 pass below), **never trained.** |
| **`r_retqual` calibration** | **τ RE-CALIBRATED 0.30 → 0.20** (= the measured `best_sim` median). The shipped τ=0.30 sat at the ~77th percentile and penalized 3 of every 4 retrievals (mean `retqual` −0.056), so **within a GRPO group an abstaining rollout (0.0) beat a retrieving one — a retrieval-collapse incentive in the default.** See §4a. |
| **`r_covgain` (multi-turn)** | **RETIRED — `W=0` PERMANENT (M2, 2026-07-20).** No longer merely "deferred": the M2 sweep measured later-turn **marginal coverage = 0.000** at both k=3 and k=20, on top of ~0.97 turns/rollout. The behavior it pays for does not occur and a wider turn budget does not create it. Left in the code, inert, so the ablation stays reproducible. PI's stated doubt about multi-turn is now empirically settled. |
| **Reasoning axis** | `r_select` justified by grounding forensics (54.5% train / 6% test) — **not built.** `r_infer` **FAILED** Audit B (+0.023) → redesign via co-occurrence weighting, or drop. |
| **GPU runs to date** | Two at `n=8`: the M1 baseline (`848159`, ckpt 200) and M3/v7 (`854501`, in flight). Everything before those was `n=1` and does not count. |

**Immediate critical path:** ~~set `n=8` + batch fix~~ **(done)** → ~~M2 offline~~ **(done)** →
~~launch P1/M3 (`r_retqual` only, τ=0.20)~~ **(launched `854501`)** → read both arms at **step 200**
→ decide on baseline seeds 2–3 → M4 reasoning axis.

> **Submit from a LOGIN node** — `sbatch` is not available on compute nodes.
> ```
> USE_RTHINK=0 EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 \
>   sbatch -o output_baseline_n8.log sbatch_run_dual_gpu_rthink.sh
> ```

### M0 execution pass (2026-07-16) — what running the code revealed

Four defects. **Three were invisible to code review** and surfaced only by executing the reward on
saved rollouts. All are fixed; each would have silently degraded or wasted the first `n=8` run.

| # | Defect | Why it mattered | Fix |
|---|---|---|---|
| 1 | **`v7/retrieval.py`'s `DOC` regex recovered only 67% of retrieved docs.** It required a literal `)\n`, but the result string is `.strip()`-ed, so the **last doc of every response** ends at `)` with no newline. Exactly one doc lost per response, in 100% of responses (6,108 headers − 4,102 recovered = 2,006 = the response count). | `r_retqual` takes a **max** over a turn's docs, so a third of the evidence was missing and `sim` could only be depressed. Worse: the Audit-B script parses at **100%** recall, so **the shipped term was not the term that scored +0.112 / AUC 0.85.** | Terminator → `\)(?:\n|\Z)`. Recall 100%; reward and audit parsers now agree exactly. Docs/turn 2.03 → **3.03**, which independently confirms effective top-k = 3. |
| 2 | **2.13% of `<tool_call>`s failed `json.loads`** (65/3,058), and 63 of 65 then received no `<tool_response>` at all — `sglang_rollout.py:960` caught `JSONDecodeError` and discarded the call, so the search never fired. Cause: apostrophe-leading titles (`'70s Gold`, `'68 Comeback`) colliding with the model's own quoting. | Those turns retrieved nothing at all. | `verl/utils/tool_call_repair.py`, invoked **only** from inside the existing `except` branch → structurally cannot regress the 97.87% that parse today. Recovers 55.4% (36/65): **2.13% → 0.95% residual**, with **0** regressions across a 947-call corpus. The residual 29 are garbled beyond safe repair (stray `?` before `]`, doubled `]]`). |
| 3 | **τ = 0.30 created a retrieval-collapse incentive** (§4a). | GRPO would actively push toward "never retrieve" — the mode the floor exists to prevent. | τ → 0.20 (measured median). |
| 4 | **The top-k lever is not in `retrieval_launch.sh`.** Its `--topk 1` is **inert**: `verl/tools/search_tool.py:178` sends `topk = config.get("topk", 3)` in the payload, and `retrieval_server.py:370-371` only falls back when the request omits it. **Effective top-k is 3**, set by `search_tool_config.yaml` — which declares no `topk:` key at all. | M2's "widen 3→20" would have edited a file with no effect and measured nothing. | Annotated in `retrieval_launch.sh`. **M2 must add `topk: 20` to `search_tool_config.yaml`.** |

**Also found — a stale installed verl** at `~/.local/lib/python3.12/site-packages/verl`, with no
`reward_reasoning` package. Training is unaffected (`python3 -m verl.trainer.main_ppo` with
cwd=project dir resolves the working tree), but any script run as `python3 foo.py` silently imports
the **stale** copy — it returned a bare `0.0` before this was caught. **Pin
`PYTHONPATH=/work/11138/pranavbelligundu/vista/verl_R1` in every offline script.**

**Standing lesson:** execute a new reward term offline against saved decodes before it ever sees GPU.
It is cheap, and it is where the bugs actually are.

---

## 0. The frozen system (never drift from this)

From the RRCM paper, **frozen** — improvements come from the reward only:

| Component | What it is (RRCM §) |
|---|---|
| **Dual-memory corpus** | Collaborative Memory (historical user-history docs) + Meta Memory (item metadata); §3.2. Amazon CDs&Vinyl = 76,287 collaborative records + item metadata. |
| **Retrieval interface** | single `Retrieve(q)` over both memories; multi-turn `<think>`→`<tool_call>`→`<tool_response>`→`<answer>`; §3.3. |
| **Grounding** | ŷ → SentenceTransformer embed → cosine to catalog → top-N list `L_cand`; §3.4.1. |
| **GRPO** | group-relative advantage, KL to ref; **group size 8** (§3.4.3 / PI-confirmed). |
| **Baseline reward (the bar)** | `R(τ) = Σ_{n∈{1,5,10,50,100}} wₙ·InTop@n(L_cand, i_gt) + λ·I_parse`, `w={0.5,0.3,0.1,0.08,0.02}`, `λ=1`; §3.4.2. |

**The bar** (RRCM Table 1, Amazon CDs&Vinyl): HR@5 0.0102, NDCG@5 0.0073, HR@10 0.0129, NDCG@10 0.0117.
The paper's own ablations (Table 2) show reasoning and CF retrieval both matter, yet the outcome-only
reward gives **zero explicit gradient to either** — it only scores the final rank. That gap is the thesis:

> RRCM learns retrieval + reasoning *implicitly* from an outcome-only objective. We show that
> **explicit, correctness-aligned process rewards** for retrieval and reasoning improve on it — and we
> identify *when* and *why* they help.

---

## 1. What the reward can and cannot move

```
test HR ≈ P(GT in retrieved docs) × P(model picks GT | GT in docs)
            └── COVERAGE ~5% ──┘      └── SELECTION ~6% test / ~54% train ──┘
            corpus-fixed, off-limits    reward-addressable — our lever
```

1. **Coverage (~5%) is a corpus property — off-limits.** No reward raises it.
2. **Selection has real headroom.** 54% on train vs 6% on test is a generalization gap, not a capability
   ceiling (greedy diagnostic: train-greedy HR@1 0.39 vs test 0.005 → not representation-bound).

**Why v7 can move what v2–v6 couldn't:** on the ~95% of test prompts where GT is unreachable, *every
outcome-reward rollout failed equally* → within a group, zero variance → zero GRPO gradient. `r_retqual`
has **non-zero variance on every prompt** — rollouts differ in how *close* their retrieved items land
even when none retrieves GT — so for the first time there is gradient on the unwinnable majority.
(This argument requires `n>1`; at the old `n=1` there was no group to vary. It is real only from the
`n=8` runs onward.)

**Realistic journal target:** a *consistent*, multi-seed HR/NDCG lift over outcome-only RRCM across all
three datasets, driven by better retrieval quality and selection. Even HR@5 0.0102→~0.012 replicated
with a clean reward story and mechanism analysis is a publishable extension.

---

## 2. Audits — run, verdicts recorded

Both audits are **done** (log-only, no GPU). They gate what gets built.

**Audit A — selection headroom.** On the winnable subset (GT in docs), are transferable features
(frequency / history co-occurrence) able to rank GT, vs the train-only positional cue?
- **Verdict:** transferable features beat the positional control in the ALL-items view (clustered CI
  excludes 0); the positional control never wins → whatever signal exists is *transferable*, not a
  train-only artifact (the failure mode that would have doomed the axis is ruled out). Held-out UNSEEN
  view can't be certified — only ~63 distinct winnable prompts exist (coverage starvation), so it's
  under-powered, not negative.
- **Consequence:** `r_select` is justified (transferable signal exists); certifying its held-out size is
  blocked by coverage, so judge it on selection-rate movement, not raw HR.

**Audit B — correctness correlation** (`correlation_audit.py`; named by function to avoid colliding with
the train-vs-test greedy diagnostic informally called "Audit B", which is separately done and decisive).

| term | vs InTop@10 | AUC | verdict |
|---|---|---|---|
| `r_retqual` | +0.112 | 0.85 | **PASS — lead term** (PI-endorsed design) |
| `best_sim` (raw) | +0.102 | 0.85 | PASS (shaping adds little over raw proximity — τ/floor are calibration knobs, not the signal) |
| `r_covgain` | −0.003 | 0.499 | FAIL-but-**deferred** (near-constant zero: the multi-turn behavior it rewards barely exists in the logs; wrong instrument for an exploration term). **PI also explicitly uncertain multi-turn helps** → deferral is PI-endorsed. |
| `r_infer` | +0.023 | 0.61 | **FAIL — real** (varies plenty, still doesn't predict; the v2 tripwire firing). Redesign via co-occurrence weighting or drop. |

Caveat on all held-out numbers: only ~25 InTop@10 positives even pooled — wide error bars; confirm on
the first real training curve.

---

## 3. The two axes — retrieval leads, reasoning is the conditional second act

Order set by the audit evidence (not the earlier draft's `r_infer`-first plan, which is superseded).
Both compose as **capped, LongPAS-gated shaping** on the frozen `InTop@n` outcome term. Math lives in
the design docs; this is the strategy.

### 3.1 Retrieval axis (lead — `retrieval_reward_design.md`)
- **`r_retqual`** — the PI's literal proposal: score each retrieval by similarity of retrieved items to
  GT; good query rewarded, junk penalized (floored, to avoid teaching "never retrieve"), no-retrieval
  neutral. **Validated (Audit B), PI-endorsed. First to train.**
- **`r_covgain`** — multi-turn bonus, pays only a later retrieval that beats the running best.
  **Deferred** (W=0) pending the Phase-1 offline test of whether multi-turn even raises coverage — the
  PI's stated uncertainty, made decidable.
- Later: `r_memtype` (which memory to retrieve), `r_when` (whether to retrieve).

### 3.2 Reasoning axis (conditional — `reasoning_reward_design.md`)
- **`r_select`** — credit for picking GT when it *is* in the docs (attacks the 6%→54% gap). Justified by
  grounding forensics; **not gated on Audit A/B** (its evidence is the forensics, which are settled).
  Sparse gradient (fires only on the ~5% winnable subset) → ride it *alongside* `r_retqual`, not alone.
- **`r_infer`** — the PI's "infer from similar items." Plain form **failed** Audit B. Redesign path:
  weight consistency by history co-occurrence (the transferable features Audit A found). Re-run Audit B
  on the variant; if it doesn't clear ~+0.10, drop it and rest the reasoning axis on `r_select`.
  *Note:* `r_infer`'s "provides within-group variance on all-miss prompts" rationale was untestable at
  `n=1` and only becomes meaningful once `n=8` runs exist.

---

## 4. Group size and the baseline

- **`n=8` is canonical** (PI). The prior `n=1` disabled GRPO's group-relative baseline (the `len==1`
  branch makes advantage = raw reward → baseline-free REINFORCE). This is why v2–v6 sat in the noise
  band and why their negative results don't count. **The PI confirmed this bug was in the original code**
  and that he fixed it to 8 in his own later experiments.
- **The baseline you must beat is `n=8`.** RRCM's Table 1 was produced by the PI's `n=8` runs, so your
  local `n=1` baseline was never the real baseline. Re-run the outcome-only RRCM reward at `n=8`, 3
  seeds — ideally reproducing the paper's ~0.0102 HR@5 — before any A/B.
- **Escalation ladder** (PI): start at 8; if improvement is below expectation, try 12 then 16.
- **Batch divisibility (pre-flight) — RESOLVED, and the earlier advice here was wrong.**
  `train_batch_size × n` must be divisible by `ppo_mini_batch_size`. The old 56/40 setup was tuned for
  `n=5` (280/40=7 clean); `n=8` gives 448, which 40 does **not** divide (448/40=11.2).
  - **Shipped fix: `ppo_mini_batch_size=56`** → 448/56 = **8** clean mini-batches, `train_batch` unchanged.
  - ⚠️ **The previously suggested `ppo_mini_batch=64` CRASHES**: `verl/workers/config/actor.py:153`
    raises `ValueError` when `train_batch_size (56) < ppo_mini_batch_size (64)`. Do not use it.
  - ⚠️ **"or the first step may crash" was also wrong — it does NOT crash.** Nothing in verl asserts
    this rule: `verl/workers/actor/dp_actor.py:388` splits with a non-strict chunker
    (`protocol.py:912`), so an indivisible combo **silently** trains a short final mini-batch whose
    loss is scaled by the wrong denominator (`dp_actor.py:400/418`). It is a quietly wrong gradient,
    not a crash — which is why `run_in_container_rthink.sh` now pre-flights it and exits non-zero.
  - The three real constraints, all enforced by that pre-flight:
    `train_batch × n % ppo_mini_batch == 0` (unasserted in verl) · `ppo_mini_batch % ppo_micro_batch_per_gpu == 0`
    (`fsdp_workers.py:257`) · `train_batch >= ppo_mini_batch` (`actor.py:153`).
    56/56/8 satisfies all three, and so do `ROLLOUT_N=12` and `16`.
- **`n=8` is GPU-safe.** Peak KV is n-independent (bounded by the sglang pool at util 0.65 and
  `ppo_micro_batch=8`, with ~3× idle headroom at n=1); only wall-clock rises (~higher per step). Still,
  watch the first ~15 steps live given the prior OOM history.

---

## 5. Milestones

| # | Milestone | Gate to advance |
|---|---|---|
| **M0** | ✅ **DONE** (2026-07-16, minus eval seeding). `n=8` + batch fix (`mini=56`) + launcher pre-flight; v7 wired into the launcher (`RTHINK_MODE=v7`, `W_COVGAIN=0`, all vars in the `--env` passthrough); JSON/apostrophe prevalence **measured (2.13%)** and repaired → 0.95%; DOC-parser bug fixed; τ calibrated. **PI-code prerequisite dropped by the user.** | ✅ v7 `compute_score` executed end-to-end offline (all 14 keys, shaping math, cap + LongPAS asserted). ⏳ **Remaining: the launch gate itself** — config launches without OOM (watch first ~15 steps). ⏸️ **Deferred: eval decode seeding** — needs a `seed` field on the `RolloutConfig` dataclass (shared with training); NOT required for the ≥3-seed mean±std protocol, which works unseeded (it only adds exact replayability). |
| **M1** | 🔄 **Re-baseline RRCM at `n=8`** (the real Table-1 baseline, reproduced). Job `848159` → ckpt **200**; eval decode `854449`. **The 3-seed requirement is OPEN** given the step-200 lock — do not spend windows on seeds 2–3 without a user/PI call. | stable target with seed spread. ⚠️ **"ideally reproduces the paper's ~0.0102" may not hold:** `reward_SPRec.py` pays tiers `1.0/0.8/0.5/0.1/0.001`, whereas §0's table describes the paper's reward as weighted `InTop@n` summing to `1.0/0.5/0.2/0.1/0.02`. Different shapes. The A/B stays internally valid (both arms share `reward_SPRec`), but a miss vs 0.0102 would be a **reward-definition mismatch, not a failed reproduction** — resolve before reading M1 as a negative. |
| **M2** | ✅ **DONE 2026-07-20.** Phase-1 offline: top-k 3→20 + **multi-query union-coverage** measurement. **Outcomes:** `r_covgain` → `W=0` permanent (later-turn marginal coverage 0.000 @3 and @20; ~0.97 turns/rollout); `r_retqual` headroom ~2× (model queries reach GT @20 = 0.111 vs tail5 heuristic 0.197, oracle GT-query @1 = 0.894); top-k 3→20 is material for coverage (0.039→0.111) but pays only ~0.7% HR@1, so it is **its own single-variable experiment, NOT folded into M3**. Results JSON in `reward_reasoning/diagnostics/`. | ~~union-coverage gain measured~~ **met** (this decides `r_covgain`'s fate and answers the PI's multi-turn doubt offline). **Reuse `reward_reasoning/diagnostics_scripts/coverage_sweep.py`** — it already replays queries against the same e5+FAISS index the online retriever serves and sweeps `--ks` (defaults already span 3 and 20). Two deltas: (a) it unions *within* a rollout's queries, but the multi-turn question needs a **cumulative union across turns** with marginal gain per added turn; (b) **top-k is widened in `search_tool_config.yaml` (add `topk: 20`), not `retrieval_launch.sh`** — see §M0 defect 4. |
| **M3** | 🔄 **P1: first v7 GPU run — LAUNCHED 2026-07-21, SLURM `854501`** (nodes c613-002 retriever / c635-042 training, 24h). `r_retqual` only (`W_COVGAIN=0`), `n=8`, **τ=0.20**, experiment `...-gpu-rthink-v7-n8`, log `output_rthink_v7_n8.log`. The `RETRIEVAL_ONLY=1` wiring ablation (must reproduce baseline) is **still unrun** and needs its own window. | **Read at step 200**, against `...-baseline-n8`. retrieval-rate healthy; test selection / HR moves beyond seed spread. **τ/floor calibration is now largely done offline (§4a)** — the remaining job is to confirm the *training-time* `best_sim` distribution matches the offline one, and to re-centre τ on the median if the policy shifts it as it learns. |
| **M4** | Reasoning axis: `r_select`; `r_infer` cooc-variant re-Audit-B'd (or dropped) | selection-rate ↑; each term Audit-B-clean before training |
| **M5** | Full sweep, 3 datasets; RQ2-style ablations + RQ3-style behavior analysis | consistent lift over outcome-only RRCM across datasets |

Code layout (shipped): retrieval reward in its own `reward_retrieval/v7/` package, dispatched from
`reward_reasoning/__init__.py` under `RTHINK_MODE=v7`; `reward_SPRec.py` and the `InTop@n` weights
untouched so the baseline stays reproducible.

---

## 6. Evaluation protocol (canonical — design docs cross-link here)

1. **All three datasets** (Goodreads, MovieLens, Amazon) — a single-dataset win won't survive review.
2. **≥3 decode seeds, report mean ± std.** One temp-1 decode swings HR@1 0.000↔0.005 — wider than every
   between-method gap. (This is itself a methodology contribution: single-decode eval is unreliable here.)
3. **Compare the `InTop@n`/`r_answer` term, never the shaped `score`** — the shaped score isn't
   baseline-comparable.
4. **Pre-decline checkpoint** (later checkpoints were worse in v4/v5 — peak-then-decline).
5. **RQ2-style ablations** (`w/o r_retqual`, `w/o r_select`, …) so each term is attributable.
6. **RQ3-style behavior analysis** — retrieval count/type, single→multi-query shift, copier→inferencer
   shift (hits-without-doc-support). The mechanism story is as valuable as the HR delta.
7. **Pre-register expected effects** before full runs, so a positive result is confirmatory.

---

## 7. Guardrails (carried from v1→v6; don't re-earn on the paper's clock)

- **`rollout.n = 8`** — verified/canonical (the single biggest fix; every shaping term was inert at n=1).
  Re-baseline at n=8; never compare against the old n=1 baseline.
- **Cap combined shaping under the 0.02 tier gap, or LongPAS** — a well-reasoned wrong pick must never
  outrank a correct one. With two axes, cap the *sum*.
- **Every new term passes Audit B before it trains** (the r_infer FAIL is why this rule exists).
  Exception: `r_select` (justified by forensics) and exploration terms whose behavior is absent from the
  logs (judge those on the behavior they induce, e.g. `r_covgain`).
- **`*_ONLY` ablation toggle per term; one variable per run.** Never touch the corpus or `InTop@n` weights.
- **Retrieval-collapse guard** — floor every penalty; monitor retrieval-rate (RRCM RQ3 shows count
  drifts down over training; don't accelerate it).
- **Anti-hacking gate per term** — retrieval-spam (`r_retqual`), count-parroting (`r_select`).

---

## 8. Risks & early-warning

- **The selection gap may be partly corpus-rooted** (train users' own histories are in the corpus, test
  users' aren't). Audit A ruled out the *positional-artifact* worst case, but held-out size is
  uncertifiable at ~5% coverage. If `r_select` doesn't move selection, the honest contribution re-scopes
  toward the retrieval axis (PI's priority anyway) + the eval-methodology + coverage×selection framework.
- **Multi-turn may not help** (PI's explicit doubt) — M2 answers this offline before any `r_covgain`
  training.
- **`r_infer` mush risk** — keep any redesign grounded in *retrieved* evidence, not global-embedding
  closeness (the v4 failure). Re-Audit-B before trusting it.
- **Decode noise dominates at the floor** — until a term lifts HR meaningfully off ~0.005, every
  comparison sits inside the ≥3-seed error bar; lead with Audit-B correlation and behavioral metrics,
  not raw HR@1.
- **Thin positives** — ~25 InTop@10 hits behind the AUC 0.85; the first training curve is the real
  confirmation, not the offline audit.