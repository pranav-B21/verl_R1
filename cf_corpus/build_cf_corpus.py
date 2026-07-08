# SPDX-License-Identifier: Apache-2.0
"""Build a user-independent CF-continuation corpus from train-user histories.

Motivation (see cf_corpus/README.md and the coverage diagnostics): the system
is retrieval-bound, not reward-bound. Test targets EXIST in the corpus
(coverage@inf 98.1%) and are selectable when surfaced (train-side selection
~50%), but nothing in the current corpus connects a *history* to its
*continuation* in a way a test user can reach: history-shaped queries top out
at ~26% coverage@100 against a 100% GT-as-query oracle. The fix is to
restructure train behaviour into aggregated, user-independent transition docs:

    Users who played "Kind of Blue" next played: "Bitches Brew" (41), ...

This script does steps 1-3 of the pipeline (extract -> aggregate -> render)
plus the step-2/step-5 leakage/transition check (pure set arithmetic, no
embeddings). Encoding + FAISS indexing is build_index.py; the embedding
coverage sweep is diagnostics_scripts/coverage_sweep.py.

Design decisions baked in here (justified in README.md):
  * TRAIN USERS ONLY. Test sequences never enter the builder -- the whole point
    is that test targets become reachable via *other* users' behaviour.
  * Aggregate, don't dump. One doc per anchor key with continuation *counts*;
    this bounds doc length, gives the selector a frequency cue, and avoids the
    raw-co-occurrence popularity explosion.
  * Single-item anchor by default (--anchor-len 1): maximises coverage and
    matches a recency-shaped query (one recent title -> one doc). Longer
    anchors are more precise but sparser; sweep them at step 6.
  * Windowed continuations (--window): only items within w steps AFTER the
    anchor count, so a moderate window keeps the sequential-CF signal instead
    of collapsing to global popularity. Train sequences are short (avg ~10,
    max 16) so w>=10 ~= same-sequence co-occurrence.
  * Min-support + top-M pruning keeps docs short (k=20 must fit the 2048-token
    budget) and cuts singleton noise; the realized-coverage report shows what
    the pruning costs so it is a measured knob, not a guess.

Title canonicalization: all titles are matched through ONE normaliser and, when
possible, mapped to their exact name2id spelling, so the strings the model must
copy (and the exact-match reward scorer) agree with the corpus verbatim.

Usage (any python3, from verl_R1 root -- CPU, seconds):
  python cf_corpus/build_cf_corpus.py \
      --anchor-len 1 --window 10 --min-support 2 --max-cont 15 \
      --out data/amazon_data/cf/corpora_cf.jsonl \
      --report data/amazon_data/cf/build_report.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

QUOTED = re.compile(r'"([^"]+)"')


def norm(t: str) -> str:
    return t.strip().strip('"').strip().lower()


def parse_titles(s: str) -> list[str]:
    return [t.strip() for t in QUOTED.findall(s)]


def load_canonical(name2id_path: str) -> dict[str, str]:
    """norm(title) -> exact name2id spelling (the string the model must copy)."""
    if not name2id_path or not os.path.exists(name2id_path):
        return {}
    name2id = json.load(open(name2id_path))
    canon = {}
    for name in name2id:
        canon.setdefault(norm(name), name)
    return canon


def load_train_sequences(path: str, canon: dict[str, str]) -> list[list[str]]:
    """Reconstruct each train user's full chronological sequence.

    CDs_and_Vinyl_train.json stores cumulative prefixes: consecutive examples of
    one user grow by one item (input_{i+1} == input_i + output_i). A maximal
    sequence ends where the next example's input stops matching. Identical logic
    to diagnostics_scripts/transition_coverage.py so the realized-coverage
    numbers here are directly comparable to the pre-build ceiling.

    Titles are canonicalized to their name2id spelling when known, else kept
    verbatim (first-seen casing). We compare on normalized form to detect
    sequence boundaries.
    """
    data = json.load(open(path))
    seqs = []
    prev_norm = None

    def canonize(raw_titles):
        return [canon.get(norm(t), t) for t in raw_titles]

    prev_seq = None
    for ex in data:
        cur_raw = parse_titles(ex["input"])
        # Mirror transition_coverage.py exactly for boundary detection: the
        # output title is the whole `output` string with quotes/whitespace
        # stripped (not re-extracted via the quote regex), so sequence
        # reconstruction -- and thus the realized ceiling -- is reproducible.
        out_norm = norm(ex["output"])
        out_canon = canon.get(out_norm) or ex["output"].strip().strip('"').strip()
        cur_norm = [norm(t) for t in cur_raw]
        full_raw = canonize(cur_raw) + [out_canon]
        full_norm = cur_norm + [out_norm]
        if prev_norm is not None and cur_norm != prev_norm:
            seqs.append(prev_seq)
        prev_seq, prev_norm = full_raw, full_norm
    if prev_seq is not None:
        seqs.append(prev_seq)
    return seqs


def build_transitions(seqs, anchor_len, window):
    """anchor-key (tuple of `anchor_len` consecutive titles) -> Counter{cont: n}.

    For each position p that has a full anchor ending at p, count every item in
    positions p+1 .. p+window as a continuation (window=None -> to end of seq).
    Continuations that are part of the anchor itself are skipped (no self loops).
    """
    agg = defaultdict(Counter)
    for s in seqs:
        n = len(s)
        for p in range(anchor_len - 1, n):
            key = tuple(s[p - anchor_len + 1 : p + 1])
            key_set = set(key)
            hi = n if window is None else min(n, p + 1 + window)
            for q in range(p + 1, hi):
                cont = s[q]
                if cont in key_set:
                    continue
                agg[key][cont] += 1
    return agg


def render_doc(anchor_key, continuations):
    """continuations: list of (title, count) already pruned & sorted."""
    anchors = ", ".join(f'"{a}"' for a in anchor_key)
    conts = ", ".join(f'"{t}" ({c})' for t, c in continuations)
    return f"Users who played {anchors} next played: {conts}"


def prune(counter, min_support, max_cont):
    items = [(t, c) for t, c in counter.items() if c >= min_support]
    # sort by count desc, then title for determinism
    items.sort(key=lambda x: (-x[1], x[0].lower()))
    return items[:max_cont]


def realized_coverage(docs_by_anchor, samples, tails):
    """Step-2/step-5 leakage check on the *pruned, rendered* corpus.

    For each test (history, target) and each tail size k, does `target` appear as
    a continuation in any built doc whose anchor is (contained in) the test
    user's last-k played items? Single-item anchors match a last-k item
    directly; multi-item anchors match iff every anchor title is among the
    tail-k items. This is the corpus's intrinsic ceiling *after* pruning --
    independent of embeddings/retrieval -- so compare against
    transition_coverage_results.json to read off the pruning cost.
    """
    # index: for single-anchor docs, item -> set(continuations); general form
    # keyed by frozenset of anchor titles.
    cont_by_anchorset = {}
    for anchor_key, conts in docs_by_anchor.items():
        cont_by_anchorset[frozenset(norm(a) for a in anchor_key)] = {norm(t) for t, _ in conts}

    n = len(samples)
    out = {}
    for k in tails:
        hit = 0
        for hist_norm, tgt in samples:
            tail = set(hist_norm[-k:]) if k else set(hist_norm)
            found = False
            for aset, conts in cont_by_anchorset.items():
                if aset <= tail and tgt in conts:
                    found = True
                    break
            if found:
                hit += 1
        out[k] = hit / n if n else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/amazon_data/CDs_and_Vinyl_train.json")
    ap.add_argument("--name2id", default="data/amazon_data/CDs_and_Vinyl/name2id.json")
    ap.add_argument("--anchor-len", type=int, default=1,
                    help="titles per anchor key (1 = single-item, best coverage)")
    ap.add_argument("--window", type=int, default=10,
                    help="count continuations within this many steps after the "
                         "anchor; 0 or negative = to end of sequence (co-occur)")
    ap.add_argument("--min-support", type=int, default=2,
                    help="drop continuations seen fewer than this many times")
    ap.add_argument("--max-cont", type=int, default=15,
                    help="keep at most this many (top-count) continuations per doc")
    ap.add_argument("--out", default="data/amazon_data/cf/corpora_cf.jsonl")
    ap.add_argument("--start-id", type=int, default=0,
                    help="first doc id (set to len(base corpus) when appending)")
    ap.add_argument("--test-preds",
                    default="outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6/"
                            "global_step_300/greedy_test_2026-07-02/predictions.json",
                    help="test samples for the realized-coverage check")
    ap.add_argument("--tails", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--report", default=None, help="optional json report path")
    args = ap.parse_args()

    window = None if args.window is None or args.window <= 0 else args.window

    print("loading canonical titles...")
    canon = load_canonical(args.name2id)
    print(f"  {len(canon)} normalized->canonical title mappings")

    print("reconstructing train sequences (train users ONLY)...")
    seqs = load_train_sequences(args.train, canon)
    n_plays = sum(len(s) for s in seqs)
    print(f"  {len(seqs)} users, {n_plays} plays, avg {n_plays/len(seqs):.1f}, "
          f"max {max(map(len, seqs))}")

    print(f"aggregating transitions (anchor-len={args.anchor_len}, "
          f"window={'inf' if window is None else window})...")
    agg = build_transitions(seqs, args.anchor_len, window)
    print(f"  {len(agg)} distinct anchor keys before pruning")

    print(f"pruning (min-support={args.min_support}, max-cont={args.max_cont}) "
          "and rendering...")
    docs_by_anchor = {}
    for key, counter in agg.items():
        pruned = prune(counter, args.min_support, args.max_cont)
        if pruned:
            docs_by_anchor[key] = pruned

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc_lens = []
    with open(args.out, "w") as f:
        for i, (key, conts) in enumerate(sorted(docs_by_anchor.items())):
            contents = render_doc(key, conts)
            doc_lens.append(len(contents))
            f.write(json.dumps({"id": str(args.start_id + i), "contents": contents}) + "\n")
    n_docs = len(docs_by_anchor)
    import numpy as np
    doc_lens = np.array(doc_lens)
    # ~1 token per 4 chars is a safe upper bound for these short docs
    approx_tok = doc_lens / 4
    print(f"  wrote {n_docs} CF docs -> {args.out}")
    print(f"  doc chars: mean {doc_lens.mean():.0f}  p50 {np.percentile(doc_lens,50):.0f}  "
          f"p95 {np.percentile(doc_lens,95):.0f}  max {doc_lens.max()}")
    print(f"  approx tokens: mean {approx_tok.mean():.0f}  p95 {np.percentile(approx_tok,95):.0f}  "
          f"(k=20 docs ~= {20*approx_tok.mean():.0f} tok, budget 2048)")

    report = {
        "config": {
            "anchor_len": args.anchor_len,
            "window": "inf" if window is None else window,
            "min_support": args.min_support,
            "max_cont": args.max_cont,
        },
        "train_users": len(seqs),
        "n_docs": n_docs,
        "anchor_keys_before_prune": len(agg),
        "doc_chars": {"mean": float(doc_lens.mean()),
                      "p95": float(np.percentile(doc_lens, 95)),
                      "max": int(doc_lens.max())},
    }

    # step-2/step-5 leakage + realized transition coverage
    if args.test_preds and os.path.exists(args.test_preds):
        preds = json.load(open(args.test_preds))
        samples = [([norm(t) for t in parse_titles(p["input"])],
                    norm(p["output"])) for p in preds]
        print(f"\nrealized transition coverage on {len(samples)} test samples "
              "(target appears as a 'next played' in a tail-k-reachable doc):")
        cov = realized_coverage(docs_by_anchor, samples, args.tails)
        for k in args.tails:
            print(f"  tail-{k}: {cov[k]:.3f}")
        report["realized_transition_coverage"] = {str(k): cov[k] for k in args.tails}
        print("\n  ^ compare to transition_coverage_results.json (raw, unpruned) "
              "to read the pruning cost; this is the corpus ceiling BEFORE any "
              "embedding/retrieval loss.")
    else:
        print(f"\n[skip realized-coverage: test-preds not found at {args.test_preds}]")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        json.dump(report, open(args.report, "w"), indent=2)
        print(f"\nwrote report -> {args.report}")


if __name__ == "__main__":
    main()
