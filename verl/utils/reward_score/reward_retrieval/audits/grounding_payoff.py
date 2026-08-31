# SPDX-License-Identifier: Apache-2.0
"""Does GROUNDING the answer in a retrieved doc pay -- when the GT is not in the docs?

The unexamined assumption. Every arm drives grounding UP: v3-v6 pay `RTHINK_W_GROUND`,
v8 pays `r_select`'s `is_grounded` (and shipped at W_GROUND=0.40), and both arms converge
to 99-100% grounded. Nobody has asked whether that is good, because the funnel
`HR = coverage x P(pick GT | covered)` silently assumes the answer comes from the docs.

It only assumes that because the POLICY makes it true. Coverage is ~3.2%, so on ~96.8% of
prompts, grounding to a retrieved item GUARANTEES the answer is not the GT. In that
majority the only thing that can still score is embedding proximity -- `eval.py` ranks the
catalog by L2 distance to the embedded answer STRING, so HR@5 does not require picking the
GT, only emitting something whose embedding lands near it. Whether another user's specific
album is closer to this user's next item than a title the model would generate from the
history alone is an empirical question, and it has never been measured.

The one data point that exists leans against grounding: v7's collapsed arm answered at 5%
retrieval, almost entirely from parametric memory, and its HR@5 (0.0043) was not worse than
the 99%-grounded baseline's (0.0030) -- Part 8.6. That was confounded by being a different
policy. This script removes the confound by comparing grounded and ungrounded rollouts of
the SAME checkpoint, stratified by coverage, and paired within prompt.

This is the reward-addressable version of roadmap P3 `r_when` (never built): not "select
better" but "decide when to ground at all". It is the only remaining lever that does not
route through coverage or selection, both of which are closed by measurement.

Stratification, and the cell that matters:

    covered   + grounded    -- the winnable set; selection is at chance here (settled)
    covered   + ungrounded  -- threw away a retrieved GT
    UNCOVERED + grounded    -- 96.8% of prompts. Guaranteed miss on exact match;
                               scores only via embedding proximity.  <-- THE CELL
    UNCOVERED + ungrounded  -- parametric fallback.                   <-- vs THIS ONE

If uncovered+ungrounded ranks better than uncovered+grounded, a reward that grounds
CONDITIONALLY beats one that grounds always, without selection ever beating chance.

Usage:
    python3 grounding_payoff.py --label baseline@200 <pooled test_predictions.json...>
"""

import argparse
import json
import os
import random
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_behavior import _norm, retrieved_titles  # noqa: E402
from memtype_payoff import CATALOG, METRICS, extract_last_quoted_answer_span, score  # noqa: E402


def boot_ci(vals, rng, n_boot=10000):
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rollouts = []
    for p in args.preds:
        for rec in json.load(open(p, encoding="utf-8")):
            text = "".join(rec.get("predict") or [])
            title = extract_last_quoted_answer_span(text) or "NAN"
            docs = retrieved_titles(text)
            gt = _norm(rec.get("output", ""))
            rollouts.append({
                "pid": rec["input"],
                "gt_raw": rec.get("output", "").strip().strip('"'),
                "title": title,
                # grounded = the emitted title is EXACTLY one of the retrieved items
                "grounded": _norm(title) in docs,
                "covered": bool(gt) and gt in docs,
            })

    from sentence_transformers import SentenceTransformer

    dev = torch.device("cpu")
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
    emb = torch.tensor(torch.load(os.path.join(CATALOG, "embeddings.pt")), device=dev)
    name2id = json.load(open(os.path.join(CATALOG, "name2id.json"), encoding="utf-8"))
    uniq = sorted({r["title"] for r in rollouts})
    vecs = torch.tensor(model.encode(uniq, batch_size=256, show_progress_bar=False), device=dev)
    ranks = torch.cdist(vecs, emb, p=2).argsort(dim=-1).argsort(dim=-1)
    idx = {t: i for i, t in enumerate(uniq)}

    cells, per_prompt, dropped = {}, {}, 0
    for r in rollouts:
        if r["gt_raw"] not in name2id:
            dropped += 1
            continue
        rank0 = int(ranks[idx[r["title"]]][name2id[r["gt_raw"]]].item())
        s = score(rank0)
        s["rank"] = rank0 + 1
        cells.setdefault((r["covered"], r["grounded"]), []).append(s)
        if not r["covered"]:  # the uncovered majority is the comparison of interest
            per_prompt.setdefault(r["pid"], {True: [], False: []})[r["grounded"]].append(s)

    def agg(rows, k):
        return sum(x[k] for x in rows) / len(rows) if rows else float("nan")

    n_tot = sum(len(v) for v in cells.values())
    print(f"\n{'=' * 78}\ngrounding payoff   [{args.label}]\n{'=' * 78}")
    print(f"scored rollouts={n_tot}  dropped (target not in name2id)={dropped}\n")

    print(f"{'stratum':28s} {'n':>6s} {'HR@1':>8s} {'HR@5':>8s} {'HR@10':>8s} {'NDCG@5':>8s} {'medRank':>9s}")
    for covered in (True, False):
        for grounded in (True, False):
            rows = cells.get((covered, grounded), [])
            if not rows:
                continue
            name = f"{'covered' if covered else 'UNCOVERED':10s} + {'grounded' if grounded else 'ungrounded'}"
            print(f"{name:28s} {len(rows):6d} {agg(rows,'hr@1'):8.5f} {agg(rows,'hr@5'):8.5f} "
                  f"{agg(rows,'hr@10'):8.5f} {agg(rows,'ndcg@5'):8.5f} "
                  f"{statistics.median([x['rank'] for x in rows]):9.0f}")

    disc = {p: v for p, v in per_prompt.items() if v[True] and v[False]}
    print(f"\nUNCOVERED prompts, PAIRED within prompt (ungrounded - grounded), n={len(disc)}")
    print(f"  {'metric':8s} {'mean diff':>11s} {'bootstrap 95% CI':>26s}")
    res = {"label": args.label, "files": args.preds, "n_scored": n_tot,
           "cells": {f"cov={c},grnd={g}": {"n": len(v), **{k: agg(v, k) for k in METRICS}}
                     for (c, g), v in cells.items()},
           "paired_uncovered": {"n_prompts": len(disc)}}
    for k in METRICS:
        diffs = [agg(v[False], k) - agg(v[True], k) for v in disc.values()]
        lo, hi = boot_ci(diffs, rng)
        m = sum(diffs) / len(diffs) if diffs else float("nan")
        res["paired_uncovered"][k] = {"mean_diff": m, "ci95": [lo, hi]}
        flag = "" if (lo <= 0 <= hi) else "  <- CI excludes 0"
        print(f"  {k:8s} {m:+11.5f}   [{lo:+9.5f}, {hi:+9.5f}]{flag}")

    print("\nPositive = UNGROUNDED (parametric) answers rank better on prompts whose GT the")
    print("retriever did not return. That is the 96.8% majority, so it dominates HR.")
    print("CONFOUND: the policy chose when to ground. Pairing removes prompt difficulty,")
    print("not within-prompt selection. Causal test = an A/B on a conditional-grounding reward.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
