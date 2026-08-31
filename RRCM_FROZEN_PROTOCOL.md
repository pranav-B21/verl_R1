# Frozen-corpus RRCM control and decision-entropy extension

## Research question and end goal

This is not a v9 handcrafted reward. The immediate objective is to establish a
paper-faithful RRCM control on the immutable repaired corpus. The only extension
then tests whether preventing an early retrieve-versus-answer policy collapse
increases ground-truth retrieval coverage and, ultimately, held-out recommendation
accuracy.

The final success criterion is better test HR@5. More retrieval, more metadata
use, or higher decision entropy without better HR@5 is a mechanistic result, not
a successful recommender.

## Immutable substrate

Every training and evaluation job must pass `scripts/verify_rrcm_frozen_inputs.py`
against `data/amazon_data/rrcm_frozen_manifest.json`. The guard pins:

- `corpora_v2.jsonl`: 85,302 documents
- `e5_Flat_v2.index`: the matched 85,302-row index
- `train.parquet`: 4,096 training prompts
- `val_rrcm_512_seed43.parquet`: 512 validation prompts, sampled from the existing
  validation JSON after excluding exact train/test duplicates
- `test.parquet`: 1,000 test prompts, used only after checkpoint selection is locked

There is no corpus regeneration, new metadata, index change, reranker, memory-type
router, or forced CF-to-META transition in this protocol.

## Matched control

The control uses the setup described in `RRCM_NIPS.md`:

| Component | Locked value |
|---|---|
| Model | Qwen3-1.7B |
| Algorithm | GRPO, 8 rollouts per prompt, temperature 1.0 |
| Batch geometry | 56 prompts × 8 rollouts = 448 trajectories |
| Tool | unified natural-language memory, top-1 result, one query per call |
| Turn cap | 5 assistant turns |
| Reward | published cumulative InTop@{1,5,10,50,100} weights plus `-1` malformed-answer penalty |
| Optimizer | LR 1e-6, warmup ratio 0.285, PPO clip 0.2, KL loss coefficient 0.001 |
| Checkpoints | every 50 steps, through step 1,300 |
| Validation | the frozen 512-prompt validation set; greedy, one sample |
| Seeds | 41, 42, 43 |

The reward accepts exactly one `<answer>"title"</answer>` and contains no
retrieval, successor, metadata, grounding, or reasoning bonus.

## Single-variable extension

The extension uses the exact control configuration and reward. Its only changed
variable is binary entropy regularization at a recognized post-retrieval action:

1. The first `<tool_call>` is not regularized.
2. After that retrieval, the first token that distinguishes `<tool_call>` from
   `<answer>` receives binary entropy regularization.
3. Entropy is normalized over only those two action tokens. Tokens outside the
   two actions cannot satisfy the regularizer, so malformed output is not rewarded.
4. The coefficient is 0.001 through step 300, decays linearly, and is zero at and
   after step 500.

This intervention can preserve another-retrieval versus answer-now exploration.
It does not dictate a query, memory type, item, or answer.

## Staged execution and gates

### Stage 0: preflight

Run:

```bash
cd /work/11138/pranavbelligundu/vista/verl_R1
python scripts/verify_rrcm_frozen_inputs.py \
  --data-dir data/amazon_data \
  --manifest data/amazon_data/rrcm_frozen_manifest.json \
  --require-validation
```

The entropy arm is invalid if any of these occur: an input hash mismatch,
effective top-k other than 1, more than one submitted query per call, an empty
decision mask throughout training, or a coefficient that does not follow the
300-to-500 schedule.

### Stage 1: matched seed-41 pilots through step 500

Submit the control and extension with the same code path:

```bash
TOTAL_STEPS=500 SEED=41 DECISION_ENTROPY_ENABLED=0 \
  sbatch sbatch_run_rrcm_faithful.sh

TOTAL_STEPS=500 SEED=41 DECISION_ENTROPY_ENABLED=1 \
  sbatch sbatch_run_rrcm_faithful.sh
```

Decode validation—not test—at steps 300 and 500. Use a distinct validation tag:

```bash
DECODE_PARQUET="$PWD/data/amazon_data/val_rrcm_512_seed43.parquet" \
DECODE_TAG=valgreedy EVAL_REPEATS=3 \
RTHINK_RUNS="rrcm-paper-frozen-control-seed41:300 \
rrcm-paper-frozen-control-seed41:500 \
rrcm-paper-frozen-decision-entropy-seed41:300 \
rrcm-paper-frozen-decision-entropy-seed41:500" \
  sbatch sbatch_run_test_rrcm_faithful.sh
```

Continue beyond the pilot only if:

- the intervention moves its intended mechanism by step 300: binary action
  entropy rises by at least 0.05 nats or second-query rate rises by at least
  5 percentage points versus control; and
- by step 500 it produces either at least +0.5 percentage points of retrieval
  coverage or at least +0.002 validation HR@5; and
- valid-answer rate is not more than 5 percentage points below control.

If the mechanism moves but coverage and HR do not, this is the same dissociation
as v8 and the extension stops. Do not tune the coefficient on test results.

### Stage 2: three-seed completion

If Stage 1 passes, resume seed 41 and launch seeds 42 and 43 through step 1,300:

```bash
for seed in 41 42 43; do
  TOTAL_STEPS=1300 SEED="$seed" DECISION_ENTROPY_ENABLED=0 \
    sbatch sbatch_run_rrcm_faithful.sh
  TOTAL_STEPS=1300 SEED="$seed" DECISION_ENTROPY_ENABLED=1 \
    sbatch sbatch_run_rrcm_faithful.sh
done
```

The seed-41 jobs resume automatically from their step-500 experiment directories.

## Validation-only checkpoint selection

For each arm and seed, evaluate the locked grid 500, 700, 900, 1100, and 1300 on
the validation parquet. Select by mean validation HR@5, then mean NDCG@5, then the
earlier step. The selector refuses to inspect untagged test decodes:

```bash
python scripts/select_rrcm_checkpoint.py \
  --eval-root outputs/eval/rrcm-paper-frozen-control-seed41 \
  --tag-prefix valgreedy \
  --output outputs/eval/rrcm-paper-frozen-control-seed41/selection.json
```

Repeat for all six arm/seed runs and lock the resulting JSON records before any
test decode.

## Final test and interpretation

Evaluate each locked checkpoint on `test.parquet` with three greedy decodes and
the same five-turn/top-1/one-query tool configuration. The primary endpoint is
the paired, within-prompt extension-minus-control HR@5 difference, pooled with a
hierarchical bootstrap over prompts and seeds. HR@1 and NDCG@5 are secondary;
coverage, second-query rate, and binary action entropy are mechanistic endpoints.

Call the extension successful only if the HR@5 point improvement is at least
0.002 and its 95% interval excludes zero. If behavior or coverage improves while
HR remains null, report the negative result and stop this intervention family.

