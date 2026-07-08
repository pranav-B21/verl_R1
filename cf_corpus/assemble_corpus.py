# SPDX-License-Identifier: Apache-2.0
"""Assemble a servable corpus from doc-set parts and renumber ids (step 4).

The retriever (retrieval_server.py: load_docs -> corpus[int(idx)]) maps a FAISS
row to the corpus JSONL by *line order*, so the assembled file's line order must
equal the index's vector order and ids must be 0..N-1 in that order. This script
concatenates parts in the given order and rewrites ids accordingly, so the
FAISS index built from the SAME file (build_index.py) stays aligned.

Also provides `--split-base`: slice the existing corpora.jsonl into its metadata
docs (the item-representation docs, kept as oracle-reachable item reps) and its
history docs (the per-user "A user played ..." docs), so you can A/B whether the
old history docs help or crowd out the new CF docs for history-shaped queries.

Usage (from verl_R1 root):
  # one-time: split the existing corpus into meta / history parts
  python cf_corpus/assemble_corpus.py --split-base \
      --base data/amazon_data/corpora.jsonl --out-dir data/amazon_data/cf

  # assemble the recommended primary corpus: CF docs + item metadata
  python cf_corpus/assemble_corpus.py \
      --inputs data/amazon_data/cf/corpora_cf.jsonl \
               data/amazon_data/cf/corpora_meta.jsonl \
      --out data/amazon_data/cf/corpora_cf_meta.jsonl
"""

import argparse
import json
import os

HISTORY_PREFIX = "A user played the following musics"


def split_base(base_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "corpora_meta.jsonl")
    hist_path = os.path.join(out_dir, "corpora_history.jsonl")
    n_meta = n_hist = 0
    with open(base_path) as f, open(meta_path, "w") as fm, open(hist_path, "w") as fh:
        for line in f:
            c = json.loads(line)["contents"]
            if c.startswith(HISTORY_PREFIX):
                fh.write(line if line.endswith("\n") else line + "\n")
                n_hist += 1
            else:
                fm.write(line if line.endswith("\n") else line + "\n")
                n_meta += 1
    print(f"split {base_path}: {n_hist} history -> {hist_path}, "
          f"{n_meta} metadata -> {meta_path}")
    print("  (ids NOT renumbered in the splits; renumber when assembling)")


def assemble(inputs: list[str], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    counts = []
    with open(out_path, "w") as fo:
        for part in inputs:
            c0 = n
            with open(part) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["id"] = str(n)
                    fo.write(json.dumps(rec) + "\n")
                    n += 1
            counts.append((part, n - c0))
    print(f"wrote {n} docs -> {out_path}")
    for part, c in counts:
        print(f"  {c:>7} from {part}")
    print("Next: build_index.py on this exact file (line order == vector order).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-base", action="store_true")
    ap.add_argument("--base", default="data/amazon_data/corpora.jsonl")
    ap.add_argument("--out-dir", default="data/amazon_data/cf")
    ap.add_argument("--inputs", nargs="+", help="doc-set parts, in serving order")
    ap.add_argument("--out", default="data/amazon_data/cf/corpora_cf_meta.jsonl")
    args = ap.parse_args()

    if args.split_base:
        split_base(args.base, args.out_dir)
    if args.inputs:
        assemble(args.inputs, args.out)
    if not args.split_base and not args.inputs:
        ap.error("nothing to do: pass --split-base and/or --inputs")


if __name__ == "__main__":
    main()
