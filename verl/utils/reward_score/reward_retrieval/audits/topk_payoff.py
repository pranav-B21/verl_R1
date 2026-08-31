# SPDX-License-Identifier: Apache-2.0
"""Does widening retrieval top-k actually buy HR? Controlled, single-checkpoint.

`coverage_sweep.py` answers half the question: replay a frozen checkpoint's own
queries at increasing k and coverage rises steeply (k=3 -> 10 nearly doubles it).
Read alone that argues for widening. It is the wrong half.

    HR@1  =  coverage  x  P(pick GT | GT in the candidate set)

Widening k raises coverage AND enlarges the candidate set the policy must pick
from. The selector in both arms sits AT chance (= 1/|items|, measured in
decode_selection.py), so the two effects move in opposite directions and the
product is what matters. This script measures BOTH terms off the same frozen
queries and reports the product, which is the only number that answers "should
we spend an arm on top-k widening".

|items| is counted the way decode_selection.py counts it: unique quoted item
titles appearing in the retrieved docs (deduplicated across docs, so overlap
between docs is credited correctly and the count saturates sublinearly in k).

Runs in the `retriever` conda env on one GPU, ~2 min. No policy, no rollout:
the queries are replayed verbatim from saved decodes, so the ONLY thing that
varies between conditions is k.

Usage:
    conda activate retriever
    python topk_payoff.py --preds <test_predictions.json> --label <name> [...]
"""

import argparse
import importlib.util
import json
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SWEEP = os.path.join(_HERE, "..", "..", "reward_reasoning", "diagnostics_scripts",
                      "coverage_sweep.py")

spec = importlib.util.spec_from_file_location("coverage_sweep", _SWEEP)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

QUOTED = re.compile(r'"([^"]+)"')
META_TITLE = re.compile(r"^title: (.*?), price: ", re.I)


def doc_titles(content_lower):
    """Unique item titles a doc exposes -- the same surface decode_selection.py picks from."""
    t = {x.strip() for x in QUOTED.findall(content_lower)}
    m = META_TITLE.match(content_lower)
    if m:
        t.add(m.group(1).strip())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", required=True)
    ap.add_argument("--corpus", default="data/amazon_data/corpora.jsonl")
    ap.add_argument("--index", default="data/amazon_data/e5_Flat.index")
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10, 20, 50])
    ap.add_argument("--out", default="topk_payoff.json")
    args = ap.parse_args()
    kmax = max(args.ks)

    import faiss

    print("loading corpus...")
    contents_lower, _ = cs.load_corpus(args.corpus)
    titles_per_doc = [doc_titles(c) for c in contents_lower]
    print(f"  {len(contents_lower)} docs")
    index = faiss.read_index(args.index)
    encoder = cs.E5Encoder(args.model)

    results = {}
    for preds_path, label in zip(args.preds, args.label):
        preds = json.load(open(preds_path))
        gts, qlists = [], []
        for p in preds:
            gts.append(cs.norm_title(p["output"]))
            rollout = p["predict"][0] if isinstance(p["predict"], list) else p["predict"]
            qlists.append(cs.extract_queries(rollout))

        flat = [q for qs in qlists for q in qs]
        print(f"\n[{label}] encoding {len(flat)} queries...")
        emb = encoder.encode(flat)
        _, ids = index.search(emb, kmax)

        # map flat query rows back to their sample
        owner, cur = [], 0
        for qs in qlists:
            owner.append(list(range(cur, cur + len(qs))))
            cur += len(qs)

        entry = {}
        for k in args.ks:
            cov = 0
            n_items = []
            for i, rows in enumerate(owner):
                docs = []
                for r in rows:
                    docs.extend(int(x) for x in ids[r][:k] if x >= 0)
                seen, uniq = set(), []
                for d in docs:  # union across this sample's queries, order preserved
                    if d not in seen:
                        seen.add(d)
                        uniq.append(d)
                items = set()
                for d in uniq:
                    items |= titles_per_doc[d]
                n_items.append(len(items))
                if gts[i] and gts[i] in items:
                    cov += 1
            c = cov / len(preds)
            it = float(np.mean(n_items))
            chance = 1.0 / it if it else 0.0
            entry[k] = dict(coverage=c, items_mean=it, chance=chance, hr_at_chance=c * chance)
            print(f"  k={k:3d}  coverage={c:.4f}  |items|={it:6.1f}  "
                  f"chance={chance:.4f}  HR@1 if selector at chance={c * chance:.5f}")
        results[label] = entry

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\nHR@1-at-chance is the decisive column. Widening k pays only if the")
    print("selector can be held ABOVE chance; at chance the coverage gain and the")
    print("candidate-set dilution cancel, and the product is flat or falls.")


if __name__ == "__main__":
    main()
