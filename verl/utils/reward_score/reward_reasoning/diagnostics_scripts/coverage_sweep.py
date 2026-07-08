# SPDX-License-Identifier: Apache-2.0
"""Offline GT-coverage@k sweep against the amazon retrieval corpus.

Replays queries against the same e5 + FAISS Flat index the online retriever
serves, sweeping topk, and measures whether the ground-truth title appears
(quote-delimited) in the retrieved docs. No training or GPU rollout needed.

Query variants per test sample:
  - model:   queries extracted from <tool_call> JSON in saved predictions.json
             rollouts (union of retrieved docs when a rollout made >1 query)
  - history: the raw user-history string from the prompt (mirrors corpus doc
             phrasing; upper bound for query-policy improvements)
  - gt:      the ground-truth title itself as the query (oracle reachability
             probe: can ANY single e5 query surface a GT-bearing doc?)
  - tailK_items:  each of the last K played titles as its OWN query, results
             unioned (matches single-anchor CF-continuation docs + a multi-turn
             policy issuing one query per recent item). This is the headline
             variant for the restructured CF corpus.
  - tailK_concat: the last K played titles joined into ONE query (matches a
             single query that names a short recent-history window).

Also reports coverage@inf: does the GT title exist anywhere in the corpus at
all (as a quoted item in any doc). Distinguishes corpus expansion (absent)
from corpus restructuring (present but unreachable).

Usage (retriever conda env):
  python coverage_sweep.py \
    --preds outputs/eval/<exp>/global_step_300/greedy_test_2026-07-02/predictions.json \
    --label v6-test \
    --out coverage_sweep_results.json
"""

import argparse
import json
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

QUOTED = re.compile(r'"([^"]+)"')
META_TITLE = re.compile(r"^title: (.*?), price: ", re.I)
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
HISTORY = re.compile(
    r"The user has played the following musics before:\s*(.*?),\s*please write", re.S
)


def norm_title(t: str) -> str:
    return t.strip().strip('"').strip().lower()


def extract_queries(rollout: str) -> list[str]:
    queries = []
    for chunk in TOOL_CALL.findall(rollout):
        chunk = chunk.strip()
        try:
            args = json.loads(chunk)["arguments"]["query_list"]
            queries.extend(q for q in args if isinstance(q, str) and q.strip())
        except (json.JSONDecodeError, KeyError, TypeError):
            # ~1% of rollouts have malformed JSON (unescaped quotes); salvage
            m = re.search(r'"query_list"\s*:\s*\[\s*"(.*)"\s*\]', chunk, re.S)
            if m:
                queries.append(m.group(1))
    return queries


def load_corpus(path: str):
    contents_lower = []
    all_titles = set()
    with open(path) as f:
        for line in f:
            c = json.loads(line)["contents"]
            cl = c.lower()
            contents_lower.append(cl)
            all_titles.update(t.strip().lower() for t in QUOTED.findall(c))
            m = META_TITLE.match(cl)
            if m:
                all_titles.add(m.group(1).strip())
    return contents_lower, all_titles


class E5Encoder:
    """Matches retrieval_server.py: 'query: ' prefix, mean pool, L2 norm, fp16."""

    def __init__(self, model_path="intfloat/e5-base-v2", max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, queries: list[str], batch_size=128) -> np.ndarray:
        out = []
        for i in range(0, len(queries), batch_size):
            batch = [f"query: {q}" for q in queries[i : i + batch_size]]
            inputs = self.tokenizer(
                batch, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            output = self.model(**inputs, return_dict=True)
            last, mask = output.last_hidden_state, inputs["attention_mask"]
            emb = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32, order="C")


def gt_in_docs(gt: str, doc_ids, contents_lower) -> dict:
    """Smallest rank (1-indexed) at which a GT-bearing doc appears, or None.

    Matches both corpus doc types: quoted title inside a user-history doc,
    and the title field of an item-metadata doc.
    """
    quoted = f'"{gt}"'
    meta = f"title: {gt}, price: "
    for rank, di in enumerate(doc_ids, 1):
        c = contents_lower[di]
        if quoted in c or c.startswith(meta):
            return rank
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", required=True, help="one label per preds file")
    ap.add_argument("--corpus", default="data/amazon_data/corpora.jsonl")
    ap.add_argument("--index", default="data/amazon_data/e5_Flat.index")
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    ap.add_argument("--tails", type=int, nargs="+", default=[3, 5],
                    help="recency window sizes for the history-tail query variants")
    ap.add_argument("--out", default="coverage_sweep_results.json")
    args = ap.parse_args()
    assert len(args.preds) == len(args.label)
    kmax = max(args.ks)

    import faiss

    print("loading corpus...")
    contents_lower, all_titles = load_corpus(args.corpus)
    print(f"  {len(contents_lower)} docs, {len(all_titles)} unique quoted titles")
    print("loading index...")
    index = faiss.read_index(args.index)
    assert index.ntotal == len(contents_lower), (index.ntotal, len(contents_lower))
    encoder = E5Encoder(args.model)

    results = {}
    for preds_path, label in zip(args.preds, args.label):
        preds = json.load(open(preds_path))
        n = len(preds)
        gts, model_qs, hist_qs, hist_titles, parse_fail = [], [], [], [], 0
        for p in preds:
            gts.append(norm_title(p["output"]))
            rollout = p["predict"][0] if isinstance(p["predict"], list) else p["predict"]
            qs = extract_queries(rollout)
            if not qs:
                parse_fail += 1
            model_qs.append(qs)
            m = HISTORY.search(p["input"])
            hist_qs.append(m.group(1).strip() if m else "")
            hist_titles.append([t.strip() for t in QUOTED.findall(m.group(1))] if m else [])

        cov_inf = [gt in all_titles for gt in gts]
        print(f"\n[{label}] n={n}  no-query rollouts={parse_fail}  "
              f"coverage@inf={np.mean(cov_inf):.3f}")

        variants = {
            "model": model_qs,
            "history": [[h] if h else [] for h in hist_qs],
            "gt": [[gt] for gt in gts],
        }
        for k in args.tails:
            # one query per recent item (results unioned across the sample's queries)
            variants[f"tail{k}_items"] = [ts[-k:] for ts in hist_titles]
            # the last-k titles joined into a single query, quoted like the corpus
            variants[f"tail{k}_concat"] = [
                ['. '.join(f'"{t}"' for t in ts[-k:])] if ts else [] for ts in hist_titles
            ]
        entry = {"n": n, "no_query_rollouts": parse_fail,
                 "coverage_inf": float(np.mean(cov_inf)), "variants": {}}

        for vname, qlists in variants.items():
            flat, owner = [], []
            for i, qs in enumerate(qlists):
                for q in qs:
                    flat.append(q)
                    owner.append(i)
            if not flat:
                continue
            print(f"  encoding {len(flat)} '{vname}' queries...")
            emb = encoder.encode(flat)
            _, ids = index.search(emb, kmax)

            # best (smallest) GT rank across a sample's queries, per k
            best_rank = [None] * n
            per_query_docs = [[] for _ in range(n)]
            for row, i in enumerate(owner):
                per_query_docs[i].append(ids[row])
            for i in range(n):
                for doc_ids in per_query_docs[i]:
                    r = gt_in_docs(gts[i], doc_ids, contents_lower)
                    if r is not None and (best_rank[i] is None or r < best_rank[i]):
                        best_rank[i] = r

            cov = {k: float(np.mean([br is not None and br <= k for br in best_rank]))
                   for k in args.ks}
            entry["variants"][vname] = {
                "coverage_at_k": cov,
                "n_queries": len(flat),
                "best_rank_present": sorted(br for br in best_rank if br is not None),
            }
            print(f"    {vname}: " + "  ".join(f"@{k}={cov[k]:.3f}" for k in args.ks))

        results[label] = entry

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
