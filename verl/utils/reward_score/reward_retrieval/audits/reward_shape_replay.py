# SPDX-License-Identifier: Apache-2.0
"""M-C: does our outcome reward's SHAPE differ from RRCM's in a way GRPO can feel?

Our baseline (`reward_SPRec.py:159-174`) pays a tiered step function that is NOT the
reward RRCM trains on (`RRCM_NIPS.md:203-206, 253`). `ROADMAP_v7.md:264` flagged this and
it was never fixed, so every A/B in this project ran against a non-paper outcome reward:

    GT rank | paper  Sum_n w_n*InTop@n | reward_SPRec.py
    ------- | ---------------------- | ---------------
          1 | 1.0                    | 1.0
        <=5 | 0.5                    | 0.8
       <=10 | 0.2                    | 0.5
       <=50 | 0.1                    | --          (our band is <=100)
      <=100 | 0.02                   | 0.1
      <=500 | 0                      | 0.001
    parse   | lambda*I_parse = -1    | /4, floor -0.5

Why the shape can matter more than the values. GRPO does not use the reward; it uses the
reward's WITHIN-GROUP spread -- advantage = (r - group_mean)/group_std over the n=8
rollouts of one prompt. A group whose rollouts all land in the same tier has zero variance
and contributes NO gradient at all, whatever the tier pays. Our reward gives one flat 0.1
plateau across ranks 10-100 where the paper gives two tiers, and pays 0.001 out to rank
500 where the paper pays nothing. Most rollouts live in exactly that region.

So the question is not "are the numbers different" (obviously) but:

  1. DEAD GROUPS   -- what share of prompt-groups have zero reward variance, i.e. no
                      gradient, under each reward?
  2. DISCORDANCE   -- within a group, how often do the two rewards ORDER a pair of
                      rollouts differently? The decisive sub-case is pairs our reward
                      TIES that the paper's reward SEPARATES: those are gradient the
                      current reward is throwing away.

If both rewards produce near-identical group structure, the shape difference is cosmetic
and a reward-shape arm is not worth GPU. If our reward ties materially more pairs, the
outcome term is a live lever -- the only component v2-v8 never varied.

CAVEAT. Groups here are pooled DECODES of the same held-out prompt, not true training
rollout groups: temperature 1.0 matches training, but these are held-out prompts at
max_response_length 3072 vs training's 2048, and the policy is frozen. This measures the
reward functions' resolving power on a realistic rank distribution, which is the question;
it is not a simulation of a training step.

Usage:
    python3 reward_shape_replay.py --label baseline@200 <pooled test_predictions.json...>
"""

import argparse
import json
import os
import random
import re
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memtype_payoff import CATALOG, extract_last_quoted_answer_span, tiered_match  # noqa: E402

# RRCM_NIPS.md:203-206 + :253 -- N = {1,5,10,50,100}, w = {0.5,0.3,0.1,0.08,0.02}, lambda = 1.
# InTop@n is cumulative, so a rank-1 answer collects every w_n.
PAPER_CUTOFFS = ((1, 0.5), (5, 0.3), (10, 0.1), (50, 0.08), (100, 0.02))
PAPER_LAMBDA = 1.0

ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.S)


def paper_reward(rank_1based, parse_ok):
    """Sum_{n} w_n * InTop@n(L_cand, i_gt)  +  lambda * I_parse."""
    acc = sum(w for n, w in PAPER_CUTOFFS if rank_1based is not None and rank_1based <= n)
    return acc + (0.0 if parse_ok else -PAPER_LAMBDA)


def current_reward(rank_1based, n_open, n_close, has_title):
    """`reward_SPRec.similarity_match` payout, reproduced exactly (lines 159-176)."""
    if not has_title:
        return 0.0
    match = tiered_match(rank_1based)
    if n_open > 1 or n_close > 1:
        match = match / 4
        if match == 0:
            match = -0.5
    return match


def spread(vals):
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def boot_ci(vals, rng, n_boot=10000):
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", help="pooled test_predictions.json for ONE checkpoint")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-key", choices=("paper", "paper_acc"), default="paper",
                    help="which paper variant the pairwise section compares against "
                         "('paper_acc' drops lambda*I_parse to isolate the rank tiers)")
    args = ap.parse_args()
    global PAIR_KEY
    PAIR_KEY = args.pair_key

    rng = random.Random(args.seed)

    rollouts = []
    for p in args.preds:
        for rec in json.load(open(p, encoding="utf-8")):
            text = "".join(rec.get("predict") or [])
            blocks = ANSWER_BLOCK.findall(text)
            title = extract_last_quoted_answer_span(text)
            # "the required <answer> format ... using two double quotes"
            parse_ok = len(blocks) == 1 and bool(re.search(r'"[^"]+"', blocks[0]))
            rollouts.append({
                "pid": rec["input"],
                "gt": rec.get("output", "").strip().strip('"'),
                "title": title or "NAN",
                "n_open": text.count("<answer>"),
                "n_close": text.count("</answer>"),
                "has_title": bool(title),
                "parse_ok": parse_ok,
            })

    # Rank the GT for every distinct predicted title -- same numerics as eval.py.
    from sentence_transformers import SentenceTransformer

    device = torch.device("cpu")
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
    embeddings = torch.tensor(torch.load(os.path.join(CATALOG, "embeddings.pt")), device=device)
    name2id = json.load(open(os.path.join(CATALOG, "name2id.json"), encoding="utf-8"))
    uniq = sorted({r["title"] for r in rollouts})
    vecs = torch.tensor(model.encode(uniq, batch_size=256, show_progress_bar=False), device=device)
    ranks = torch.cdist(vecs, embeddings, p=2).argsort(dim=-1).argsort(dim=-1)
    idx = {t: i for i, t in enumerate(uniq)}

    groups, n_dropped = {}, 0
    for r in rollouts:
        if r["gt"] not in name2id:
            n_dropped += 1
            continue
        rank1 = int(ranks[idx[r["title"]]][name2id[r["gt"]]].item()) + 1
        groups.setdefault(r["pid"], []).append({
            "cur": current_reward(rank1, r["n_open"], r["n_close"], r["has_title"]),
            "paper": paper_reward(rank1, r["parse_ok"]),
            # DECOMPOSITION. The paper's lambda*I_parse = -1 dwarfs every accuracy tier
            # (max 1.0), so a group that merely mixes parseable and unparseable rollouts
            # gets variance ~1.0 regardless of ranking. Score the accuracy term ALONE to
            # separate "the rank tiers resolve more" from "a format penalty resolves
            # more" -- the latter is v5's format gate, already built and already tested.
            "paper_acc": paper_reward(rank1, True),
            "rank": rank1,
        })
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    dead = {"cur": 0, "paper": 0, "paper_acc": 0}
    stds = {"cur": [], "paper": [], "paper_acc": []}
    tie_cur_sep_paper = tie_paper_sep_cur = flipped = both_tied = concordant = n_pairs = 0

    for rows in groups.values():
        for k in ("cur", "paper", "paper_acc"):
            s = spread([x[k] for x in rows])
            stds[k].append(s)
            if s == 0.0:
                dead[k] += 1
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                dc = (a["cur"] > b["cur"]) - (a["cur"] < b["cur"])
                dp = (a[PAIR_KEY] > b[PAIR_KEY]) - (a[PAIR_KEY] < b[PAIR_KEY])
                n_pairs += 1
                if dc == 0 and dp == 0:
                    both_tied += 1
                elif dc == 0:
                    tie_cur_sep_paper += 1
                elif dp == 0:
                    tie_paper_sep_cur += 1
                elif dc != dp:
                    flipped += 1
                else:
                    concordant += 1

    ng = len(groups)
    rank_dist = [x["rank"] for rows in groups.values() for x in rows]
    bands = {"1": 0, "2-5": 0, "6-10": 0, "11-50": 0, "51-100": 0, "101-500": 0, ">500": 0}
    for r in rank_dist:
        k = ("1" if r == 1 else "2-5" if r <= 5 else "6-10" if r <= 10 else "11-50" if r <= 50
             else "51-100" if r <= 100 else "101-500" if r <= 500 else ">500")
        bands[k] += 1

    print(f"\n{'=' * 78}\nM-C  outcome-reward shape replay   [{args.label}]\n{'=' * 78}")
    print(f"groups (prompts with >1 pooled decode)={ng}   rollouts={len(rank_dist)}   "
          f"dropped (target not in name2id)={n_dropped}")

    print("\nGT-rank distribution (where the gradient actually has to come from):")
    for k, v in bands.items():
        print(f"  rank {k:8s} {v:6d}  ({v / len(rank_dist):6.1%})")

    print("\n1. DEAD GROUPS  (zero within-group reward variance => no GRPO gradient)")
    for k, name in (("cur", "reward_SPRec (ours)"), ("paper", "RRCM full (+lambda*I_parse)"),
                    ("paper_acc", "RRCM accuracy term ONLY")):
        print(f"  {name:24s} {dead[k]:5d}/{ng}  ({dead[k] / ng:6.1%})   "
              f"mean within-group std = {statistics.mean(stds[k]):.4f}")

    print(f"\n2. PAIRWISE ORDERING  vs '{PAIR_KEY}'  (within group, over all rollout pairs)")
    print(f"  total pairs                          {n_pairs}")
    print(f"  tied under BOTH                      {both_tied:7d}  ({both_tied / n_pairs:6.1%})")
    print(f"  concordant (same order)              {concordant:7d}  ({concordant / n_pairs:6.1%})")
    print(f"  OURS ties, PAPER separates           {tie_cur_sep_paper:7d}  "
          f"({tie_cur_sep_paper / n_pairs:6.1%})   <- gradient we discard")
    print(f"  PAPER ties, OURS separates           {tie_paper_sep_cur:7d}  "
          f"({tie_paper_sep_cur / n_pairs:6.1%})")
    print(f"  strictly FLIPPED order               {flipped:7d}  ({flipped / n_pairs:6.1%})")

    live = [g for g in groups.values() if spread([x["cur"] for x in g]) == 0.0
            and spread([x[PAIR_KEY] for x in g]) > 0.0]
    print(f"\n  groups DEAD under ours but LIVE under '{PAIR_KEY}': {len(live)}/{ng} "
          f"({len(live) / ng:.1%})")

    res = {
        "label": args.label, "pair_key": PAIR_KEY, "files": args.preds, "n_groups": ng, "n_rollouts": len(rank_dist),
        "rank_bands": bands, "dead_groups": dead, "mean_group_std":
            {k: statistics.mean(v) for k, v in stds.items()},
        "pairs": {"total": n_pairs, "both_tied": both_tied, "concordant": concordant,
                  "ours_ties_paper_separates": tie_cur_sep_paper,
                  "paper_ties_ours_separates": tie_paper_sep_cur, "flipped": flipped},
        "dead_ours_live_paper": len(live),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
