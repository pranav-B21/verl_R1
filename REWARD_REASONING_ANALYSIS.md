# Reasoning-Reward (R_think) Analysis — Why It Underperforms the No-Reasoning Baseline

**Scope:** `verl/utils/reward_score/reward_SPRec_reasoning.py` (Parts A/B/C),
`verl/utils/reward_score/reward_SPRec_rthink.py` (orchestrator), and the
`reward_SPRec.py` outcome reward it wraps.
**Evidence:** reward-debug lines parsed from
`nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink.log` (v1, β=1.0, 183 samples) and
`nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v2.log` (v2, β=0.5, 187 samples,
since overwritten by a new run), plus config dumps and the run scripts.

---

## 0. TL;DR

The reasoning reward, as currently integrated, does **not** behave as
`R_total = R_answer + λ·(α·info_gain − β·redundancy + γ·exploration)`. After the
two things that actually matter under GRPO — **per-rollout variance** and
**resume-safety** — are accounted for, it collapses to roughly:

> **"Add a near-constant negative offset to every rollout, whose only surviving
> intra-group signal is *how much the model restated the user's history* — a
> quantity uncorrelated with (and probably anti-correlated with) recommendation
> correctness."**

That is strictly worse than the baseline, where all-wrong groups contribute zero
gradient. The baseline ignores hopeless groups; R_think injects redundancy-noise
gradients into them. Every other component (info_gain, exploration) is
effectively inert. The empirical fingerprints below confirm this.

---

## 1. How the pipeline currently works

`default_compute_score` (reward_score/`__init__.py:104`) routes `amazon`/
`goodreads`/`movie` data sources to `reward_SPRec_rthink.compute_score` when
`USE_RTHINK=1`. That orchestrator:

1. Computes **R_answer** via `reward_SPRec.compute_score` — extract `<answer>`,
   embed the predicted title, rank the ground-truth item by cosine distance:
   rank-1 → 1.0, ≤5 → 0.8, ≤10 → 0.5, ≤100 → 0.1, ≤500 → 0.001, else 0
   (`reward_SPRec.py:91`). Multiple `<answer>` tags → ÷4 or −0.5.
2. Computes **info_gain / redundancy / exploration** in
   `reward_SPRec_reasoning.compute_reasoning_components`.
3. Forms `R_think = α·ig − β·red + γ·exp` (`reward_SPRec_rthink.py:73`).
4. **Asymmetric application** (`:84`):
   - `R_answer ≥ 0.5`: `R_total = R_answer + λ·max(0, R_think)`
   - `R_answer < 0.5`: `R_total = R_answer + λ·R_think`

The reward manager (`reward_manager/naive.py:88`) **does** populate
`extra_info["prompt_str"]`, so the prompt *is* available to the reasoning code
(the `solution_str` itself starts at `<think>` and contains no prompt).
Defaults in effect: `λ=0.3, α=0.5, β=0.5, γ=0.3` (v2); `β=1.0` (v1).
`adv_estimator=grpo`, `norm_adv_by_std_in_grpo=True`, `use_kl_in_reward=False`.

---

## 2. Empirical fingerprints (from the reward-debug lines)

| metric | v1 rthink (β=1.0) | v2 rthink (β=0.5) |
|---|---|---|
| `ig` mean (max) | 0.017 (0.114) | 0.017 (0.097) |
| `ig` > 0.1 frequency | 0.5 % | **0 %** |
| `red` mean | 0.877 | 0.629 |
| `red` = 1.0 frequency | 71 % | 8 % |
| `red` in [0.4, 0.8) | — | 79 % |
| `exp` mean | 0.273 | 0.274 |
| `R_think` mean | −0.787 | −0.224 |
| `R_think` sign | negative ≈100 % | negative ≈99 % (max +0.066) |
| `R_total` (wrong answers only) | — | mean −0.069, **std 0.033** |

Sanity check on v2 means: `0.5·0.017 − 0.5·0.629 + 0.3·0.274 = −0.225` ≈ observed
`R_think` mean −0.224. The CLAUDE.md "Tier-1 / CF-domain" fix worked (red=1.0
dropped 71 %→8 %), **but R_think is still negative ~99 % of the time and
info_gain is still dead.** The fix changed the magnitude of the penalty, not the
fact that the term is a one-sided penalty with no offsetting reward.

A representative think block (v1 log) shows exactly what `red` fires on:

```
<think>
Okay, let's see. The user has a list of music they've recently played:
"Can You Stand The Heat", "Seesaw", "Goin' Home", ...
I need to recommend a new music they might like.
First, I should analyze their existing preferences. ...
```
→ `ig=0.000  red=1.000`. This is **necessary grounding** (restating the user's
history as working memory), it is **near-identical across all rollouts** (learned
template), and after entity-stripping the *framing* ("the user has played … I
should analyze their preferences") still matches the prompt verbatim. So the same
text simultaneously (a) trips every redundancy tier and (b) saturates the
past-thinking-novelty axis that kills info_gain.

---

## 3. Root-cause findings (prioritized)

### F1 — `info_gain` is structurally dead → the α-reward is never paid (CRITICAL)
`ig` mean 0.017, **never exceeds 0.114, >0.1 in 0–0.5 % of samples.** Two
compounding causes:

- **`min(sentence_score, past_novelty)` is too aggressive** (`reasoning.py:290`).
  It requires novelty on *both* axes. `past_novelty` compares the current think
  block against a **global deque of the last 50–200 think blocks from *different*
  prompts** (`_PAST_THINK_EMBEDDINGS`, `:49`, `:190`). In recommendation, every
  think block shares the template "user listened to X → analyze → recommend Y",
  so cross-prompt cosine sim is high → `past_novelty ≈ 0` → `min(...) ≈ 0`. The
  template-collapse detector eats *all* signal because the buffer mixes prompts.
- **`sentence_level_novelty` vs prompt+search penalizes grounding** (`:159`).
  Legitimate reasoning restates the user's history; after entity stripping the
  structural framing still matches the prompt → low novelty.

**Consequence:** the "reward good reasoning" half of the design contributes ~0.
Only the `−β·redundancy` penalty is live. **R_think is net-negative by
construction**, not because the model reasons badly.

### F2 — `redundancy` penalizes necessary/structural reasoning, not copying (CRITICAL)
The penalty correlates with the model's natural, useful behavior (restating the
user's listening history and framing the task), **not** with reasoning quality.
Penalizing it teaches the model to *stop grounding its answer in the specific
history* — consistent with the observed accuracy drop and the `ORRatio` increase
(model retreats to safe high-frequency items when it stops reasoning about the
specific user). Tier 2 still uses `ref = prompt + search` (`:359`) and Tier 3
checks both, so even with the Tier-1 fix the template restatement keeps `red`
≈0.5–0.6 (v2 mean 0.629).

### F3 — R_think is a near-constant offset; under GRPO only its *variance* survives, and that variance is redundancy-noise (CRITICAL)
GRPO advantage = `(r_i − mean_group) / std_group` with
`norm_adv_by_std_in_grpo=True`. **Any reward component constant across a group
cancels.** What survives R_think is its intra-group *variance*. Since
`info_gain ≈ 0` (const) and `exploration ≈` const within a group (see F4), the
only surviving variance comes from **redundancy** (R_total std among wrong
answers = 0.033, entirely redundancy-driven). So post-normalization the entire
R_think machinery reduces to: *"down-weight whichever rollouts restated/structured
their think text most."* That is noise w.r.t. correctness and competes with the
real R_answer gradient.

**This is the core reason reasoning < no-reasoning.** In the baseline, an
all-wrong group has zero reward variance → zero advantage → zero gradient (the
group is harmlessly ignored). With R_think, that same group now has redundancy
variance → non-zero advantage → a gradient that optimizes a proxy unrelated to
correctness. R_think turns the ~majority of uninformative groups from "no-op"
into "actively misleading."

### F4 — `exploration_bonus` is invisible to GRPO (per-group constant) (HIGH)
`exp = EMA(sigmoid(global cum_avg)) · length_factor` (`:433`). The sigmoid/EMA
terms depend only on **global** cumulative reward → identical for all rollouts at
a given step → cancel in group-relative advantage. The *only* per-rollout
variation is `length_factor`, so the γ-term degenerates into a weak "prefer
~100-word think blocks" length reward. The intended train-progress-aware
exploration schedule (the whole point of Part C) has **zero effect** under GRPO.
Scalar-on-outcome rewards cannot reproduce AEPO/EPO token-/trajectory-entropy
regularization.

### F5 — module-global persistent state is not resume-safe and is mis-scoped (HIGH)
`_PAST_THINK_EMBEDDINGS`, `_STALENESS_NGRAMS`, `_CUM_REWARD_SUM`,
`_CUM_REWARD_COUNT`, `_EMA_BONUS` are module globals (`:48–57`).
- On **checkpoint resume** they reset to initial values: at step 250 the run
  restarts `cum_avg=0` → `exploration` re-inflates to "early-training high" exactly
  when it should be low; staleness vocabulary restarts empty. The decay/staleness
  schedules are therefore wrong after any resume (and these runs resume often).
- `past_novelty` is global/cross-prompt → it is a domain-template detector, not a
  per-prompt diversity detector (root of F1).
- Single `TaskRunner` today, so no cross-process fragmentation, but the design is
  brittle if reward ever runs in >1 worker.

### F6 — the asymmetric gate neuters the positive half and only ever penalizes (HIGH)
Because R_think is negative ~99 % of the time (F1–F3):
- `R_answer ≥ 0.5` branch: `max(0, R_think) = 0` → **correct answers get no
  reasoning bonus at all.** The LongPAS "preserve diverse reasoning on correct
  trajectories" intent is silently a no-op.
- `R_answer < 0.5` branch: full negative R_think applied → wrong answers get an
  extra penalty whose intra-group variance is redundancy-noise (F3).

Net asymmetric behavior = *"do nothing on correct, add redundancy-noise penalty
on wrong"* — a recipe to degrade.

### F7 — λ·R_think can swamp / reorder the R_answer partial-credit tiers (MEDIUM)
R_answer tiers in the partial-credit band are 0.1 (top-100), 0.001 (top-500), 0.
But `λ·R_think` reaches −0.13 (λ=0.3). So a genuinely-better top-100 answer
(0.1) can be pushed **below** a fully-wrong answer (0.0) purely on redundancy:
`0.1 − 0.13 = −0.03`. Within the `[0, 0.5)` band the reasoning penalty can
**reorder answer quality**, corrupting credit assignment. The hard 0.5 cutoff
also creates a cliff: rank-10 (0.5, full reward) vs rank-11 (≈0.1, penalized).

### F8 — debug logging truncates to `solution_str[:500]` (LOW, but blocks diagnosis)
The printed block is just the opening restatement; the **post-search synthesis**
(the part that *should* carry info gain) is never logged. β/Tier tuning has been
done partly blind.

### F9 — possible R_answer / true-HR misalignment (INVESTIGATE)
Printed **training** `R_answer` mean ≈ 0.31 with ~31 % at exactly 1.0, yet
**eval HR@1 = 0.002**. That gap suggests the embedding-rank reward is much easier
/ more gameable than true HR. Note `reward_SPRec.py:84-87`: if the target title
is **absent from `name2id`**, `target_id` defaults to **0**, and any prediction
whose nearest neighbor is item 0 scores rank-1 = 1.0 — a potential spurious
reward source. If R_answer is a loose proxy, shaping on top of it (and adding
noise via R_think) only accelerates drift from the real objective. **Audit the
`rankId` distribution and how often the target is missing from `name2id`.**

### F10 — `rollout.n = 1` resolved in the config (VERIFY)
The config dump shows `rollout_n: 1` and `rollout.n: 1`. GRPO normally needs
`n > 1` per prompt to form groups. If grouping instead relies on duplicated
prompts / shared `uid` in `train.parquet`, this is fine; if not, GRPO advantage
is near-degenerate for *both* baseline and rthink. This is a **shared** config so
it is not the differential cause of the regression, but it should be confirmed
before investing more in reward shaping. (Could not verify the parquet on the
login node — no pandas/pyarrow available.)

---

## 4. Why "reasoning" loses to "no reasoning" — the one-paragraph mechanism

With base correctness low, the *majority* of GRPO groups contain no correct
rollout. In the **baseline**, those groups have zero reward variance → zero
advantage → no gradient (correctly ignored). Adding R_think gives those same
groups **non-zero variance driven almost entirely by the redundancy penalty**
(F3), because the positive terms are dead (F1) or constant (F4). The model is
therefore trained, on the bulk of its data, to **reduce lexical/structural
overlap between its think block and the prompt** — i.e. to stop restating and
reasoning about the user's actual history (F2) — which removes useful working
memory and pushes it toward generic popular guesses (matching the reported
accuracy drop, higher `ORRatio`). The asymmetric gate (F6) ensures none of this
is offset by rewarding good reasoning on the correct rollouts.

---

## 5. Recommendations (prioritized)

**Fix the integration math first — it dominates everything.**

1. **Make R_think shape, not subtract (addresses F3, F6, F7).** Penalize
   redundancy only *relative to the group*: subtract the group/batch mean (or
   median) redundancy so only *above-average* copying is penalized and the
   constant offset disappears. Equivalently, z-score each R_think component across
   the group before combining. Cap `|λ·R_think|` well below the R_answer tier
   gaps (e.g. ≤ 0.05) so it can never reorder correctness tiers.

2. **Resurrect info_gain so the positive term pays out (addresses F1).**
   - Replace `min(sentence_score, past_novelty)` with a weighted blend, or drop
     `past_novelty` entirely.
   - If keeping past-novelty, scope it **per-prompt / per-GRPO-group** (compare a
     rollout's think only against the *other rollouts of the same prompt* — pass a
     group id via `extra_info`), not a global cross-prompt deque.
   - Score novelty only on the **synthesis** sentences (those *after* a
     `<info>`/`<tool_response>` block); exclude the necessary history-restatement
     from the denominator.

3. **Stop penalizing grounding in redundancy (addresses F2).** Apply redundancy
   only to the post-search portion of `<think>` (synthesis must not echo
   `<info>`); exempt the first/grounding think block, or reframe as *reward
   synthesis novelty vs `<info>`* instead of *penalize overlap with prompt*. Keep
   the Tier-1 prompt-only fix; reconsider Tier-2 using `prompt+search`.

4. **Drop or re-base exploration_bonus (addresses F4).** As a per-group constant
   it does nothing useful under GRPO. Either remove the γ-term, or drive it from a
   genuine **per-rollout** signal (e.g. token-level entropy of the think span, or
   per-rollout `1 − sim` to siblings in the same group), not global `cum_avg`.

5. **Make persistent state resume-safe and correctly scoped (addresses F5).**
   Persist/restore (or recompute from training step) `cum_avg`, `EMA`, and
   staleness across resumes, or tie them to `global_step` passed via `extra_info`;
   key the past-think buffer by prompt-group.

6. **Audit the base reward before more shaping (addresses F9, F10).** Log the
   `rankId` histogram and the fraction of targets missing from `name2id`; fix the
   `target_id = 0` fallback so absent targets don't score rank-1. Confirm how
   GRPO groups are formed given `rollout.n = 1` (check `uid`/`index` assignment
   and whether `train.parquet` repeats prompts).

7. **Log full think + per-tier breakdown (addresses F8).** Print the whole think
   block and `tier0/1/2/3`, `sentence_score`, `past_novelty`, `staleness` so
   tuning is not blind.

**Suggested validation sequence**
- Sanity ablation: `RTHINK_BETA=0, RTHINK_ALPHA=0, RTHINK_GAMMA=0` should
  reproduce the baseline exactly — confirms wiring and isolates each term.
- Then turn on **only** a group-relative redundancy term with capped magnitude
  (#1 + #3) and verify it does not regress accuracy before re-adding info_gain.
- Track in W&B: per-step mean/std of `ig`, `red`, `exp`, and the **fraction of
  groups with non-zero reward variance** (the quantity F3 is about).

---

## 6. File/line index

- Orchestrator + asymmetric gate: `reward_SPRec_rthink.py:73`, `:84`
- info_gain `min()` collapse: `reasoning.py:290`; global past buffer `:49`, `:190`
- redundancy tiers / ref=prompt+search: `reasoning.py:330`, `:359`; Tier-1 fix `:354`
- exploration global cum_avg: `reasoning.py:433`
- module-global state: `reasoning.py:48-57`
- truncated logging: `reasoning.py:531`, `rthink.py:98`
- R_answer tiers + `target_id=0` fallback: `reward_SPRec.py:84-102`
- prompt_str plumbing: `reward_manager/naive.py:88`
- reward routing: `reward_score/__init__.py:104`
- GRPO config: `run_in_container_rthink.sh:90`, resolved `rollout_n=1`,
  `norm_adv_by_std_in_grpo=True`
