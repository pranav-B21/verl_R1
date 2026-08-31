# SPDX-License-Identifier: Apache-2.0
"""Does reading the ITEM-METADATA memory actually pay off in ranking quality?

Context. RRCM is a dual-memory system -- collaborative history memory and item
metadata memory -- and `decode_behavior.py`'s META% column shows the policy
*extinguishes* the metadata memory over training in every arm, including the
unshaped outcome-only baseline (42.7% of rollouts at step 200 -> 0.07% at step
500). The paper's own ablation (Table 2, Amazon CDs & Vinyl) prices that memory
at HR@5 0.0064 w/o META vs 0.0102 full, and our converged baseline sits at
0.0080. Before building a reward that keeps the metadata memory alive, measure
whether using it is associated with better ranking AT ALL on the decodes we
already have. If metadata-using rollouts rank *worse*, the hypothesis dies here
for a couple of CPU-minutes rather than a two-node training run.

    HR@k = P(rank of GT among catalog items, by distance to the embedded answer, < k)

reproduced exactly as `eval.py` computes it: same extraction
(`extract_last_quoted_answer_span`), same encoder
(paraphrase-MiniLM-L3-v2), same `torch.cdist(p=2)`, same `argsort().argsort()`,
same `name2id` target lookup, and prompts whose target is missing from `name2id`
are dropped (eval.py `continue`s on them).

READ THE CONFOUND BEFORE READING THE NUMBERS. The policy CHOOSES when to consult
metadata, so this is observational, not causal. Two designs are reported:

  * UNPAIRED -- all META-using rollouts vs all others. Confounded by prompt
    difficulty: if the model looks up metadata precisely when the history is
    ambiguous, the META group is drawn from harder prompts and will look worse
    for reasons that have nothing to do with metadata.

  * PAIRED (the one to read) -- restricted to prompts where, across the pooled
    decodes of one checkpoint, at least one rollout used metadata and at least
    one did not. The per-prompt difference of means cancels prompt difficulty
    entirely, since it is the SAME prompt with the SAME target under both
    conditions. A residual within-prompt confound survives -- the model may
    reach for metadata on the rollouts that were already going badly -- so read
    a positive result as an upper bound and a negative one as a real warning.
    Only the A/B in Phase 2 is causal.

Usage:
    python3 memtype_payoff.py --label baseline@200 preds1.json preds2.json preds3.json
    python3 memtype_payoff.py --json out.json --label v7b@200 <preds...>
"""

import argparse
import json
import math
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_behavior import _memtype_counts  # noqa: E402  (validated doc/memtype parser)

DATA_ROOT = os.environ.get("REC_DATA_ROOT", "./data")
CATALOG = os.path.join(DATA_ROOT, "amazon_data/CDs_and_Vinyl")


def extract_last_quoted_answer_span(prediction):
    """Verbatim copy of eval.py:82-106 -- keep byte-identical to stay comparable."""
    if not prediction:
        return ""
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", prediction, flags=re.IGNORECASE | re.DOTALL)
    search_space = answer_blocks[-1] if answer_blocks else prediction
    quoted = re.findall(r'"([^"]+)"', search_space)
    if quoted:
        return quoted[-1].strip()
    if answer_blocks:
        quoted_full = re.findall(r'"([^"]+)"', prediction)
        if quoted_full:
            return quoted_full[-1].strip()
    for line in prediction.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def tiered_match(rank_1based):
    """`reward_SPRec.py:159-170` payout -- what the BASELINE reward actually paid."""
    if rank_1based == 1:
        return 1.0
    if rank_1based <= 5:
        return 0.8
    if rank_1based <= 10:
        return 0.5
    if rank_1based <= 100:
        return 0.1
    if rank_1based <= 500:
        return 0.001
    return 0.0


def load_rollouts(paths):
    """[(prompt_id, used_meta, predicted_title, gt_title)] over all pooled decodes."""
    out = []
    for p in paths:
        for rec in json.load(open(p, encoding="utf-8")):
            text = "".join(rec.get("predict") or [])
            _, n_meta, _ = _memtype_counts(text)
            name = extract_last_quoted_answer_span(text) or "NAN"
            out.append((rec["input"], bool(n_meta), name, rec.get("output", "").strip().strip('"')))
    return out


def rank_all(titles, device):
    """1-based rank of every catalog item for each distinct predicted title."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device=str(device))
    embeddings = torch.tensor(torch.load(os.path.join(CATALOG, "embeddings.pt")), device=device)
    name2id = json.load(open(os.path.join(CATALOG, "name2id.json"), encoding="utf-8"))

    uniq = sorted(set(titles))
    vecs = torch.tensor(model.encode(uniq, batch_size=256, show_progress_bar=False), device=device)
    # eval.py: dist = cdist(p=2); rank = argsort().argsort() -> rank[j] = 0-based rank of item j
    ranks = torch.cdist(vecs, embeddings, p=2).argsort(dim=-1).argsort(dim=-1)
    return {t: ranks[i] for i, t in enumerate(uniq)}, name2id, embeddings.shape[0]


def boot_ci(vals, rng, n_boot=10000):
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


METRICS = ("hr@1", "hr@5", "hr@10", "ndcg@5", "reward")


def score(rank0):
    """eval.py metric contributions from a 0-based GT rank, + the baseline reward."""
    ndcg5 = (1 / math.log(rank0 + 2)) / (1 / math.log(2)) if rank0 < 5 else 0.0
    return {
        "hr@1": float(rank0 < 1),
        "hr@5": float(rank0 < 5),
        "hr@10": float(rank0 < 10),
        "ndcg@5": ndcg5,
        "reward": tiered_match(rank0 + 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", help="pooled test_predictions.json for ONE checkpoint")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--json", help="write the result dict here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rollouts = load_rollouts(args.preds)
    ranks, name2id, n_items = rank_all([r[2] for r in rollouts], torch.device("cpu"))

    # Per-rollout metrics, dropping targets absent from name2id exactly as eval.py does.
    per_prompt = {}
    n_dropped = 0
    for pid, used_meta, title, gt in rollouts:
        if gt not in name2id:
            n_dropped += 1
            continue
        rank0 = int(ranks[title][name2id[gt]].item())
        per_prompt.setdefault(pid, {True: [], False: []})[used_meta].append(score(rank0))

    def agg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    meta_rows = [r for v in per_prompt.values() for r in v[True]]
    plain_rows = [r for v in per_prompt.values() for r in v[False]]
    discordant = {p: v for p, v in per_prompt.items() if v[True] and v[False]}

    res = {
        "label": args.label,
        "files": args.preds,
        "n_rollouts": len(rollouts),
        "n_dropped_target_not_in_catalog": n_dropped,
        "n_catalog_items": n_items,
        "unpaired": {"n_meta": len(meta_rows), "n_plain": len(plain_rows)},
        "paired": {"n_prompts_discordant": len(discordant), "n_prompts_total": len(per_prompt)},
    }

    print(f"\n{'=' * 78}\nM-B  metadata-memory payoff   [{args.label}]\n{'=' * 78}")
    print(f"pooled rollouts={len(rollouts)}  dropped (target not in name2id)={n_dropped}  "
          f"catalog={n_items}")
    print(f"META-using rollouts={len(meta_rows)}  other={len(plain_rows)}")
    print(f"prompts with BOTH conditions present (the paired sample)={len(discordant)}"
          f" of {len(per_prompt)}\n")

    print("UNPAIRED  (confounded by prompt difficulty -- reference only)")
    print(f"  {'metric':8s} {'META':>10s} {'no-META':>10s} {'diff':>10s}")
    for k in METRICS:
        a, b = agg(meta_rows, k), agg(plain_rows, k)
        res["unpaired"][k] = {"meta": a, "plain": b, "diff": a - b}
        print(f"  {k:8s} {a:10.5f} {b:10.5f} {a - b:+10.5f}")

    print("\nPAIRED within prompt  (cancels prompt difficulty -- READ THIS ONE)")
    print(f"  {'metric':8s} {'mean diff':>11s} {'bootstrap 95% CI':>26s}")
    for k in METRICS:
        diffs = [agg(v[True], k) - agg(v[False], k) for v in discordant.values()]
        lo, hi = boot_ci(diffs, rng)
        m = sum(diffs) / len(diffs) if diffs else float("nan")
        res["paired"][k] = {"mean_diff": m, "ci95": [lo, hi], "n": len(diffs)}
        flag = "" if (lo <= 0 <= hi) else "  <- CI excludes 0"
        print(f"  {k:8s} {m:+11.5f}   [{lo:+9.5f}, {hi:+9.5f}]{flag}")

    print("\nCONFOUND: the policy chose when to read metadata. Pairing removes prompt")
    print("difficulty but not within-prompt selection (it may consult metadata on the")
    print("rollouts that were already going badly). Read a positive result as an upper")
    print("bound, a negative one as a real warning. Only the Phase-2 A/B is causal.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
