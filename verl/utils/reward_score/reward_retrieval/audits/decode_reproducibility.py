# SPDX-License-Identifier: Apache-2.0
"""How reproducible is a decode regime? Measured at the METRIC, not at the text.

Why this exists (2026-08-08). `test_in_container_greedy_sweep.sh` ships a
determinism probe, and on its first ever run it fired: two `DECODE_TEMPERATURE=0`
decodes of one frozen checkpoint (baseline-n8 @ step 200) reproduced only 35.1%
of rollouts byte-for-byte. sglang's continuous batching makes logits depend on
batch composition, so near-tie tokens flip argmax and one flip cascades over a
3072-token response; `temperature=0, top_k=1` cannot control that. Divergence
starts before any retrieval in 293 of the differing prompts and in rollouts that
make no tool call at all, so it is the inference stack, not the retriever.

That number is real and it retires the claim (CLAUDE.md, and this script's own
first-draft header) that greedy is "the only regime without a decode-noise term".

**But byte-identity is the wrong bar.** 95% of rollouts land past GT rank 500, so
almost all text divergence is invisible to HR@k. What a comparison needs is the
reproducibility of the METRIC. Measured on the same pair with this script:

    regime      identical text   identical answer   identical GT rank   HR@5 spread
    greedy           35.1%            62.0%              61.7%            0.00102
    temp 1.0          0.0%             1.5%               1.6%            0.00407

HR@1 was reproduced exactly under greedy and not under temperature 1.0. Greedy
buys ~4x on the metric and ~41x on answer-level reproducibility. Consequences:

  * A SINGLE greedy decode is a better measurement than a single sampled one.
  * `paired_arm_test.py` pairs the two ARMS within prompt, so a cross-arm greedy
    comparison is valid without within-arm repeats. Repeats buy the residual
    estimate, not the validity.
  * Do not report greedy as noise-free. Quote the residual (~1 prompt/1000 on
    HR@5 at this checkpoint) next to the table.

Metrics reproduce `eval.py` exactly via `memtype_payoff.score`.

Usage:
    python3 decode_reproducibility.py A/test_predictions.json B/test_predictions.json
    python3 decode_reproducibility.py --json out.json --label "greedy baseline@200" A.json B.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memtype_payoff import extract_last_quoted_answer_span, score  # noqa: E402

DATA_ROOT = os.environ.get("REC_DATA_ROOT", "./data")
CATALOG = os.path.join(DATA_ROOT, "amazon_data/CDs_and_Vinyl")


def _titles(records):
    return [extract_last_quoted_answer_span("".join(r.get("predict") or [])) or "NAN" for r in records]


def compare(path_a, path_b, label):
    da, db = json.load(open(path_a, encoding="utf-8")), json.load(open(path_b, encoding="utf-8"))
    if len(da) != len(db):
        raise SystemExit(f"decode length mismatch: {len(da)} vs {len(db)}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
    emb = torch.tensor(torch.load(os.path.join(CATALOG, "embeddings.pt")))
    name2id = json.load(open(os.path.join(CATALOG, "name2id.json"), encoding="utf-8"))

    ta, tb = _titles(da), _titles(db)
    uniq = sorted(set(ta) | set(tb))
    vecs = torch.tensor(model.encode(uniq, batch_size=256, show_progress_bar=False))
    ranks = torch.cdist(vecs, emb, p=2).argsort(-1).argsort(-1)
    idx = {t: i for i, t in enumerate(uniq)}

    n_all = len(da)
    text_same = sum(1 for x, y in zip(da, db)
                    if "".join(x.get("predict") or []) == "".join(y.get("predict") or []))
    ans_same = sum(1 for x, y in zip(ta, tb) if x == y)

    rows_a, rows_b, rank_same, n, hr5_disagree = [], [], 0, 0, 0
    for rec, x, y in zip(da, ta, tb):
        gt = rec.get("output", "").strip().strip('"')
        if gt not in name2id:  # eval.py drops these
            continue
        n += 1
        ra, rb = int(ranks[idx[x]][name2id[gt]]), int(ranks[idx[y]][name2id[gt]])
        rank_same += ra == rb
        sa, sb = score(ra), score(rb)
        rows_a.append(sa)
        rows_b.append(sb)
        hr5_disagree += sa["hr@5"] != sb["hr@5"]

    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    res = {
        "label": label, "a": path_a, "b": path_b, "n_prompts": n_all, "n_scored": n,
        "identical_text": text_same / n_all,
        "identical_answer": ans_same / n_all,
        "identical_gt_rank": rank_same / n,
        "hr5_disagree_prompts": hr5_disagree,
        "metrics": {k: {"a": mean(rows_a, k), "b": mean(rows_b, k),
                        "abs_diff": abs(mean(rows_a, k) - mean(rows_b, k))}
                    for k in ("hr@1", "hr@5", "hr@10", "ndcg@5")},
    }

    print(f"\n{'=' * 74}\ndecode reproducibility   [{label}]\n{'=' * 74}")
    print(f"prompts={n_all}  scored (target in catalog)={n}")
    print(f"  identical ROLLOUT TEXT      {text_same:5d}/{n_all} = {text_same / n_all:6.1%}")
    print(f"  identical FINAL ANSWER      {ans_same:5d}/{n_all} = {ans_same / n_all:6.1%}")
    print(f"  identical GT RANK           {rank_same:5d}/{n} = {rank_same / n:6.1%}")
    print(f"  prompts where HR@5 differs  {hr5_disagree:5d}/{n} = {hr5_disagree / n:6.2%}"
          "   <- the only divergence the metric can see")
    print(f"\n  {'metric':8s} {'A':>9s} {'B':>9s} {'|diff|':>9s}")
    for k in ("hr@1", "hr@5", "hr@10", "ndcg@5"):
        m = res["metrics"][k]
        print(f"  {k:8s} {m['a']:9.5f} {m['b']:9.5f} {m['abs_diff']:9.5f}")
    print("\nRead the metric rows, not the text row. Byte-identity is not the bar:")
    print("95% of rollouts sit past GT rank 500, where any amount of text divergence")
    print("scores the same. Reference (baseline-n8@200): greedy text 35.1%/answer 62.0%")
    print("/HR@5 spread 0.00102  vs  temp-1.0 text 0.0%/answer 1.5%/HR@5 spread 0.00407.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--json")
    args = ap.parse_args()
    res = compare(args.a, args.b, args.label)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
