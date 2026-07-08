# SPDX-License-Identifier: Apache-2.0
"""Do train-user behaviors encode the transitions test users need?

Pre-corpus-build check for the CF-continuation restructuring: before building
"users who played X next played Y" docs from train histories, measure the
fraction of test (history -> target) pairs whose target actually appears as a
continuation of some item the test user played, in TRAIN data only. This is
the hard ceiling for any train-derived CF-transition corpus, computed with
pure set arithmetic (no embeddings, no index, CPU-only, seconds).

Sweeps two axes of the doc-builder design:
  - window w: target must appear within w steps AFTER the anchor item in a
    train sequence (w=1 -> strict adjacent transition; w=inf -> same-sequence
    co-occurrence, i.e. arbitrarily long windows)
  - anchor set: which test-history items may serve as the anchor (full
    history vs last-1/3/5 items -- the recency-shaped queries the retriever
    will actually see)

Train sequences are reconstructed from the cumulative-prefix structure of
CDs_and_Vinyl_train.json (an example is subsumed if the next example's input
equals its input + output); each maximal sequence is one user's full history.

Usage (any python3, run from verl_R1 root):
  python verl/utils/reward_score/reward_reasoning/diagnostics_scripts/transition_coverage.py
"""

import argparse
import json
import re
from collections import defaultdict

QUOTED = re.compile(r'"([^"]+)"')


def norm(t: str) -> str:
    return t.strip().strip('"').strip().lower()


def parse_titles(s: str) -> list[str]:
    return [norm(t) for t in QUOTED.findall(s)]


def load_train_sequences(path: str) -> list[list[str]]:
    data = json.load(open(path))
    seqs = []
    prev = None  # full sequence (input titles + output) of previous example
    for ex in data:
        cur_in = parse_titles(ex["input"])
        cur_full = cur_in + [norm(ex["output"])]
        if prev is not None and cur_in != prev:
            seqs.append(prev)  # prev was maximal -> one user's full history
        prev = cur_full
    if prev is not None:
        seqs.append(prev)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/amazon_data/CDs_and_Vinyl_train.json")
    ap.add_argument(
        "--test-preds",
        default="outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6/"
        "global_step_300/greedy_test_2026-07-02/predictions.json",
        help="predictions.json whose input/output fields give the same 1000 "
        "test samples used by all prior diagnostics",
    )
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--tails", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--out", default=None, help="optional json output path")
    args = ap.parse_args()

    print("reconstructing train sequences...")
    seqs = load_train_sequences(args.train)
    n_items = sum(len(s) for s in seqs)
    print(f"  {len(seqs)} train users, {n_items} item plays, "
          f"avg len {n_items / len(seqs):.1f}, max len {max(map(len, seqs))}")

    # item -> occurrences, and corpus-size stats for the aggregated doc builder
    occ = defaultdict(list)  # title -> [(seq_idx, pos)]
    adj_transitions = set()
    for si, s in enumerate(seqs):
        for p, t in enumerate(s):
            occ[t].append((si, p))
            if p >= 1:
                adj_transitions.add((s[p - 1], t))
    anchors_with_out = {a for a, _ in adj_transitions}
    print(f"  {len(occ)} distinct items, {len(adj_transitions)} distinct "
          f"adjacent transitions, {len(anchors_with_out)} distinct anchor items")

    preds = json.load(open(args.test_preds))
    samples = [(parse_titles(p["input"]), norm(p["output"])) for p in preds]
    print(f"  {len(samples)} test samples")

    windows = args.windows + [None]  # None = inf (same-sequence co-occurrence)
    anchor_sets = {"full": None} | {f"tail-{k}": k for k in args.tails}

    results = {}
    n = len(samples)
    pop_floor = sum(
        1 for _, tgt in samples if any(p >= 1 for _, p in occ.get(tgt, []))
    )
    results["target_is_any_train_continuation"] = pop_floor / n
    print(f"\npopularity floor (target follows *something* in train): "
          f"{pop_floor / n:.3f}")

    print(f"\n{'anchor set':<10}" + "".join(
        f"w={'inf' if w is None else w:<6}" for w in windows))
    for aname, k in anchor_sets.items():
        row = {}
        for w in windows:
            hit = 0
            for hist, tgt in samples:
                anchors = set(hist if k is None else hist[-k:])
                for si, p in occ.get(tgt, []):
                    if p < 1:
                        continue
                    lo = 0 if w is None else max(0, p - w)
                    if any(seqs[si][q] in anchors for q in range(lo, p)):
                        hit += 1
                        break
            row["inf" if w is None else w] = hit / n
        results[aname] = row
        print(f"{aname:<10}" + "".join(f"{row['inf' if w is None else w]:<8.3f}"
                                       for w in windows))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
