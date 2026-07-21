# Reasoning Reward — iteration v6 ("HR-faithful, top-K outcome reward")

**Status:** implemented. v6 keeps everything v5 fixed (format gate, length cap, v3
shaping, LongPAS) and changes **one** thing: the *shape* of the outcome reward, so
that climbing it must move the metric we are actually judged on.

---

## The diagnosis v6 is built on (read this first)

v5 did its job: `format_ok → 1.0`, the multi-answer hack died, verbosity stopped
inflating, training was stable, train `r_answer → ~0.38`. **Held-out, v5 only
*ties* baseline — within decode noise.** The step-300 checkpoint re-decoded twice
gives HR@1 `0.000` then `0.005` and NDCG@5 `0.0024` then `0.0063` on the *same
frozen policy*; baseline is HR@5 `0.007`. So two things are true at once: v5 did
**not** beat baseline, and the eval is **so near its floor that decode variance
spans the entire gap between methods** (HR@5 ~0.005–0.007 vs random chance
~0.0004). The binding constraint was never reward *hacking*. Three facts locate
what it actually is:

1. **The reward and the eval metric are the same proxy.** `eval.py` computes
   held-out HR/NDCG by encoding the predicted title with the *same*
   `paraphrase-MiniLM-L3-v2`, taking `cdist` to the *same* `embeddings.pt`, and
   ranking the *same* target — identical to `reward_SPRec`. So the train↔val gap is
   **not** a metric mismatch; the reward is a faithful proxy of HR *in principle*.

2. **But v4/v5 rewarded the wrong part of the rank distribution.** HR@5 scores
   only `rank ≤ 5`. v4/v5's dense term `(1 − log(r)/log(N))**2` pays **0.26 at rank
   100** and **0.11 at rank 600** — the bulk of the "core reward" the policy climbed
   lived in the **mid-rank** region. Under GRPO the advantage is intra-group
   variance, so the gradient was dominated by "move the target from rank 9000 → rank
   600," i.e. landing the answer in a broad **semantic neighborhood** without ever
   committing to a top-K item. That is **semantic mush**: it lifts temp-1 train
   reward and does nothing for greedy HR@5.

3. **The residual gap is decoding + a near-floor, high-variance metric.** Train
   r_answer 0.38 (temp-1 mean) vs held-out ~0.005 is the shape of a **diffuse
   policy** — temp-1 sampling occasionally lands a hit while the typical decode is a
   bland hedge. The same-checkpoint re-decode swing (HR@1 0.000↔0.005) confirms the
   policy's per-decode hit is essentially a coin flip near the floor: the mass is
   not concentrated on a correct mode.

> The wandb "core reward" / `score` north-star is therefore **misleading**: it is
> dominated by the mid-rank dense term, so "best checkpoint by core reward" =
> "best mid-rank mush," which is exactly what does *not* transfer. Judge v6 on
> **`r_answer/mean@1`** (and `eval.py` HR@5), never `score`.

---

## What v6 changes (one substantive thing; everything else is v5)

Make the outcome reward a **faithful surrogate of the metric, at the resolution
of the metric.** Pay the exact `eval.py` NDCG term inside the top-K window; pay
only a weak, steep tail outside it.

```
r_outcome = 1 / log2(rankId + 1)                       if rankId <= K     # = eval.py's NDCG term
r_outcome = TAIL_W * (1 - log(rankId)/log(N)) ** TAIL_P  otherwise         # weak "get warmer"
r_outcome = 0                if output is MALFORMED                        # v5 gate (kept)
R_total   = r_outcome + clip(SCALE * R_think_raw, -CAP, +CAP) - penalty    # v5 plumbing (kept)
```

| target rank | v5 `r_dense` (p=2) | **v6 `r_outcome`** (K=10, tail_w=0.10, tail_p=4) |
|---|---|---|
| 1    | 1.00 | **1.00** |
| 5    | 0.69 | **0.39** (= NDCG@5 term) |
| 10   | 0.57 | **0.29** |
| 11   | 0.56 | **0.031**  ← top-K boundary: mush stops being paid |
| 100  | 0.26 | **0.0070** |
| 600  | 0.11 | **0.0011** |
| 3000 | 0.024| **0.00006** |

**Why this is the right move under GRPO.** The advantage keeps only the *intra-group
variance* of the reward. v6 concentrates that variance at `rank ≤ K` (where HR is
scored) instead of spreading it across the mush region, so the only way to raise
the reward is to push targets toward the top-K. The weak tail still gives an
all-far group a nonzero, monotonic "get warmer" gradient (so we don't reintroduce
v4's dead-zero groups), but it is ~100× smaller than v5's, so it can no longer be
*farmed*.

**The cost (intended).** The headline `score` will be **lower** than v5's, because
v5's number was inflated by mid-rank credit. That is the point — `score` was never
the objective. Watch `r_answer` and HR@5.

### Optional decisiveness bonus (`RTHINK_REAL_BONUS`, default 0 = off)

The v6.1 lever for the residual greedy gap (fact 3). When > 0, adds a small reward
if the predicted title is an **exact catalog item** (a key of `name2id`), pushing
the policy to commit to concrete, greedy-transferable items rather than
embedding-central phrases. Off by default so v6's headline is a clean
single-variable change from v5; turn it on (e.g. `0.1`) to ablate the diffuseness
hypothesis.

---

## How to run

```bash
# Full v6 (default): clean A/B vs v5 — differs ONLY in the outcome-reward shape.
RUN_SCRIPT=run_in_container_rthink_v6.sh sbatch sbatch_run_dual_gpu_rthink.sh

# Equivalent via the v5 script + mode override (no fresh experiment name):
RTHINK_MODE=v6 EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6 \
  sbatch sbatch_run_dual_gpu_rthink.sh

# Ablations
RTHINK_REAL_BONUS=0.1 EXPERIMENT_NAME=...-rthink-v6-real   RUN_SCRIPT=run_in_container_rthink_v6.sh sbatch ...
RTHINK_DENSE_ONLY=1   EXPERIMENT_NAME=...-rthink-v6-denseonly RUN_SCRIPT=run_in_container_rthink_v6.sh sbatch ...
RTHINK_HIT_K=5        EXPERIMENT_NAME=...-rthink-v6-k5      RUN_SCRIPT=run_in_container_rthink_v6.sh sbatch ...
```

## Hyperparameters (env vars)

| Var | Default | Meaning |
|---|---|---|
| `RTHINK_HIT_K`     | `10`   | top-K window scored as `1/log2(rank+1)` (mirrors eval HR@K) |
| `RTHINK_TAIL_W`    | `0.10` | weight of the weak tail-guidance term outside top-K |
| `RTHINK_TAIL_P`    | `4.0`  | tail steepness; larger ⇒ mid-rank reward more negligible |
| `RTHINK_REAL_BONUS`| `0.0`  | bonus if the answer is an exact catalog item (v6.1 lever; 0 = off) |
| `RTHINK_DENSE_ONLY`| `0`    | `1` = no process shaping (ablation) |
| `RTHINK_FORMAT_GATE` / `_FORMAT_PENALTY` | `1` / `0.5` | v5 anti-hack gate (kept) |
| `RTHINK_LEN_SOFT` / `_LEN_W` / `_LEN_CAP` | `600` / `0.0005` / `0.2` | v5 length cap (kept) |
| `RTHINK_SCALE/_CAP/_W_*` | v3 defaults | process-shaping knobs (only when `DENSE_ONLY=0`) |

## What success looks like

1. `format_ok` stays ~1.0 and length stays flat (v5 guards still hold).
2. `score` drops vs v5 — **expected**, the mush credit is gone.
3. **Held-out `r_answer/mean@1` clears baseline, averaged over ≥3 decode seeds.**
   Because one decode of one checkpoint swings HR@1 0.000↔0.005, a single eval
   number proves nothing — report mean ± std over seeds (and ideally an extra-top-K
   metric like HR@20 that lifts the signal off the floor). This is the only success
   criterion that matters; `score` is not it.

## The decisive diagnostic (run this regardless of the v6 result)

The one experiment that tells us whether we are **optimization-bound** (a reward
problem v6/v6.1 can fix) or **proxy/representation-bound** (no reward shape can fix
it): run `eval.py` at **greedy** on a sample of **train** prompts and compare to
test.

- **train-greedy HR ≫ test-greedy HR** → generalization gap; reward shaping is
  secondary, attack it with regularization (KL coef ↑, fewer steps / early stop on
  `r_answer`, more data).
- **train-greedy HR also ≈ 0** → the diffuse-policy / proxy-capacity ceiling. The
  `paraphrase-MiniLM-L3-v2` text embedding simply can't separate top-5 from
  top-500 for these titles. Then **no reward reshaping beats baseline**, and the
  real fix is the *representation*: CF/co-occurrence item embeddings, exact-title
  matching with fuzzy fallback, or a stronger encoder — out of scope for R_think,
  but the honest conclusion to escalate.

## Iteration history (one line each)

| v | Idea | Outcome |
|---|---|---|
| v1 | `α·info_gain − β·redundancy + γ·explore` | Tier-1/CF bug → `red=1.0` everywhere; dead gradient. |
| v2 | same, Tier-1 fixed, β=0.5 | positive terms dead; only redundancy varied, `corr≈+0.1` w/ correctness → lost to baseline. |
| v3 | evidence-grounded process terms (tool/ground/synth/rep) | stable, but process variance ⊥ correctness; didn't move HR. |
| v4 | dense rank reward `(1−log r/log N)^p` | revived dead groups, but unguarded → multi-answer hack + verbosity; peaks then declines. |
| v5 | v4 + format gate + length cap + p=2 | hacks dead, stable, train r_answer→0.38; **held-out still < baseline**. |
| **v6** | **NDCG@K-faithful reward (top-K credit, weak tail)** | **concentrates the gradient where HR is scored; removes mid-rank mush.** |

The through-line: v1→v3 fixed the *process* signal, v4→v5 fixed *sparsity* and
*hacking*, **v6 fixes the reward's *resolution*** — aligning it with the extreme
top-K the metric actually scores. If v6 + the decisiveness lever still ties
baseline, the diagnostic above says the ceiling is representational, not reward.

---

## Related Work & Citations (whole implementation, v1 → v6)

The literature each design choice rests on, grouped by the mathematical pillar it
grounds, with the version(s) it applies to. **[live]** = still reflected in the
shipped reward; **[historical]** = justified a component later removed (kept for
provenance). The same list, in per-version analysis context, is in
`../../../../../../REWARD_REASONING_ANALYSIS.md` Part 9.

### A. Foundations — task, outcome reward, optimizer (all versions)
- **Rec-R1** — *Bridging Generative LLMs and User-Centric Recommendation via RL*,
  arXiv [2503.24289](https://arxiv.org/abs/2503.24289). **[live]** The base
  framework everything here rewards.
- **Search-R1** — arXiv [2503.09516](https://arxiv.org/abs/2503.09516). **[live]**
  The agentic search→retrieve→reason→answer loop.
- **GRPO / DeepSeekMath** — Shao et al. 2024, arXiv
  [2402.03300](https://arxiv.org/abs/2402.03300). **[live]** The optimizer; its
  group-relative advantage is *why only intra-group reward variance matters* — the
  lens behind every postmortem.
- **SPRec** — arXiv [2412.09243](https://arxiv.org/abs/2412.09243). **[live]** Origin
  of the embedding-rank outcome reward in `reward_SPRec.py` (shared with eval.py).

### B. Process / reasoning rewards & shaping (v1→v3; LongPAS gate persists)
- **Potential-Based Reward Shaping** — Ng, Harada & Russell, ICML 1999
  ([semanticscholar](https://www.semanticscholar.org/paper/Policy-Invariance-Under-Reward-Transformations:-and-Ng-Harada/94066dc12fe31e96af7557838159bde598cb4f10)).
  **[live, caveat]** Theory for adding a shaping term. Our `R_answer + clip(shaping)`
  is **not** the invariant form `γΦ(s′)−Φ(s)`, so the `CAP < tier-gap` bound + LongPAS
  gate stand in for the no-bias guarantee. Making shaping potential-based is a citable
  upgrade.
- **LongPAS** — arXiv [2601.12465](https://arxiv.org/html/2601.12465). **[live]**
  Source of the **asymmetric application** (correct+well-formed ⇒ positive shaping
  only) used in v3–v6 — the `if r_answer >= 0.5:` branch in this orchestrator.
- **Rewarding Progress (PAVs)** — Setlur et al. 2024, arXiv
  [2410.08146](https://arxiv.org/abs/2410.08146). **[historical → conceptually live]**
  "Dense step-level progress > sparse outcome" — the principle v4's dense reward
  operationalizes.
- **Synthetic Semantic Information Gain Reward** — arXiv
  [2602.00845](https://arxiv.org/pdf/2602.00845). **[reference]** A modern working
  `info_gain` reward, if the v1/v2 process direction is reopened.
- **EPO** ([2509.22576](https://arxiv.org/abs/2509.22576)), **AEPO**
  ([2510.14545](https://arxiv.org/abs/2510.14545)), **RM-R1**
  ([2505.02387](https://arxiv.org/abs/2505.02387)), **BERTScore**
  ([1904.09675](https://arxiv.org/abs/1904.09675)). **[historical]** Justified the
  removed v1/v2 terms; **AEPO's length penalty re-surfaces in v5/v6**, the rest are
  retired.

### C. Why dense + the GRPO zero-variance trap (v4; v6 keeps a weak tail)
- **Advantage Collapse in GRPO** — arXiv
  [2605.21125](https://arxiv.org/html/2605.21125). **[live]** Formalizes the
  all-wrong-group → zero-advantage → vanishing-gradient failure (our ~87% dead
  groups) — why v4 went dense and v6 keeps a *non-zero* tail, not a hard rank floor.
- **Scaf-GRPO** — arXiv [2510.19807](https://arxiv.org/abs/2510.19807). **[reference]**
- **RL to Rank Using Coarse-grained Rewards** — arXiv
  [2208.07563](https://arxiv.org/html/2208.07563v2). **[live]** Backs the
  v3(coarse tiers) → v4/v6(fine-grained) move.

### D. Reward hacking / proxy over-optimization (v5; v6 philosophy)
- **Scaling Laws for Reward Model Overoptimization** — Gao, Schulman & Hilton,
  ICML 2023, arXiv [2210.10760](https://arxiv.org/abs/2210.10760). **[live]** The
  canonical "optimize a proxy too hard ⇒ true performance peaks then declines"
  (Goodhart) — the v4 curve and the v6 mid-rank-mush diagnosis.
- **Reward Shaping to Mitigate Reward Hacking in RLHF** — arXiv
  [2502.18770](https://arxiv.org/html/2502.18770v3). **[live]** Backs the v5 format gate.

### E. Top-K ranking-metric optimization — the core of v6
- **Breaking the Top-K Barrier / SoftmaxLoss@K** — Yang et al., **KDD 2025**, arXiv
  [2508.05673](https://arxiv.org/abs/2508.05673). **[live, closest prior work]**
  Optimizes **NDCG@K** for recommenders; names v6's exact issues (NDCG@K is
  *discontinuous* + *top-K truncation* is hard) and derives a **smooth upper bound**
  → the upgrade path for v6's hard cliff at `K` (**v6.1**).
- Differentiable-NDCG family (justifies "reward the NDCG discount, concentrated on
  top-K"): **LambdaLoss** (Wang et al., CIKM 2018,
  [tdcommons](https://www.tdcommons.org/cgi/viewcontent.cgi?article=2281&context=dpubs_series)),
  **NeuralNDCG** ([researchgate](https://www.researchgate.net/publication/349363732_NeuralNDCG_Direct_Optimisation_of_a_Ranking_Metric_via_Differentiable_Relaxation_of_Sorting)),
  **PiRank** ([openreview](https://openreview.net/pdf?id=dL8p6rLFTS3)),
  **NDCG-surrogate stochastic optimization** (Qiu et al., ICML 2022,
  [PMLR](https://proceedings.mlr.press/v162/qiu22a/qiu22a.pdf)),
  **Learning to Rank by Optimizing NDCG** (Valizadegan et al. 2009), **SoftRank**
  (Taylor et al. 2008), **ApproxNDCG** (Qin et al. 2010). **[reference]**
- Rank metric **as an RL reward** (LTR→GRPO bridge): **Top-K Off-Policy Correction
  for REINFORCE** (Chen et al., WSDM 2019, arXiv
  [1812.02353](https://arxiv.org/pdf/1812.02353)); **Rank-GRPO** (arXiv
  [2510.20150](https://arxiv.org/pdf/2510.20150)) — GRPO + ranking reward for LLM
  recommenders, our exact integration. **[live]**

### F. The F9 train(temp-1) ↔ greedy(val) decoding gap
- **Pass@k Training** — arXiv [2508.10751](https://arxiv.org/pdf/2508.10751).
  **[live]** Temperature trades pass@1 (≈greedy) for pass@k → "judge on ≥3 seeds."
- **Learnable Temperature Policy** — arXiv
  [2602.13035](https://arxiv.org/html/2602.13035). **[reference]** A non-reward lever
  for the F9 gap.
