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


def extract_queries_by_turn(rollout: str) -> list[list[str]]:
    """Queries grouped by <tool_call> block (== one assistant turn's search).

    Preserves turn boundaries so cross-turn marginal coverage can be measured;
    extract_queries() flattens this. A turn with only malformed/empty queries is
    dropped (it retrieved nothing), so len(...) == number of *effective* turns.
    """
    turns = []
    for chunk in TOOL_CALL.findall(rollout):
        chunk = chunk.strip()
        try:
            args = json.loads(chunk)["arguments"]["query_list"]
            qs = [q for q in args if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            # ~1% of rollouts have malformed JSON (unescaped quotes); salvage
            m = re.search(r'"query_list"\s*:\s*\[\s*"(.*)"\s*\]', chunk, re.S)
            qs = [m.group(1)] if m else []
        if qs:
            turns.append(qs)
    return turns


def extract_queries(rollout: str) -> list[str]:
    return [q for turn in extract_queries_by_turn(rollout) for q in turn]


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
        model_qs_by_turn = []
        for p in preds:
            gts.append(norm_title(p["output"]))
            rollout = p["predict"][0] if isinstance(p["predict"], list) else p["predict"]
            turns = extract_queries_by_turn(rollout)
            model_qs_by_turn.append(turns)
            qs = [q for turn in turns for q in turn]
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

        # ---- cross-turn cumulative coverage (the r_covgain premise) ----
        # Existing 'model' variant unions docs across ALL of a rollout's queries
        # regardless of turn. r_covgain instead credits a *later* turn beating the
        # running best, so the question is: after retrieving with turn 1's queries,
        # how much extra GT-reachability do turns 2,3,... add? We compute, per
        # sample, the earliest turn whose cumulative retrieval places GT within
        # top-k, then report cumulative coverage after each turn and its marginal.
        max_turns = max((len(t) for t in model_qs_by_turn), default=0)
        turn_hist = [len(t) for t in model_qs_by_turn]
        entry["turn_distribution"] = {
            "mean_turns": float(np.mean(turn_hist)) if turn_hist else 0.0,
            "max_turns": max_turns,
            "counts": {str(t): int(np.sum(np.array(turn_hist) == t)) for t in range(max_turns + 1)},
        }
        print(f"  turns/rollout: mean={entry['turn_distribution']['mean_turns']:.2f} "
              f"max={max_turns}  dist={entry['turn_distribution']['counts']}")

        if max_turns >= 1:
            flat, owner, turn_idx = [], [], []
            for i, turns in enumerate(model_qs_by_turn):
                for ti, qs in enumerate(turns):
                    for q in qs:
                        flat.append(q)
                        owner.append(i)
                        turn_idx.append(ti)
            emb = encoder.encode(flat)
            _, ids = index.search(emb, kmax)
            # docs retrieved by each (sample, turn)
            docs_by_turn = [[[] for _ in range(max_turns)] for _ in range(n)]
            for row, (i, ti) in enumerate(zip(owner, turn_idx)):
                docs_by_turn[i][ti].append(ids[row])
            # earliest cumulative turn (0-indexed) at which GT lands within top-k
            cross = {}
            for k in args.ks:
                earliest = [None] * n
                for i in range(n):
                    best = None
                    for ti in range(len(model_qs_by_turn[i])):
                        for doc_ids in docs_by_turn[i][ti]:
                            r = gt_in_docs(gts[i], doc_ids[:k], contents_lower)
                            if r is not None and (best is None or r < best):
                                best = r
                        if best is not None:
                            earliest[i] = ti  # first turn making GT reachable@k
                            break
                # cumulative coverage after t turns; marginal = new hits at turn t
                cum = [float(np.mean([e is not None and e <= t for e in earliest]))
                       for t in range(max_turns)]
                marginal = [cum[0]] + [cum[t] - cum[t - 1] for t in range(1, max_turns)]
                cross[str(k)] = {
                    "cumulative_by_turn": cum,
                    "marginal_by_turn": marginal,
                    "reach_at_turn1": cum[0],
                    "extra_from_later_turns": (cum[-1] - cum[0]) if max_turns > 1 else 0.0,
                }
            entry["cross_turn_coverage"] = cross
            for k in (3, 20):
                if str(k) in cross:
                    c = cross[str(k)]
                    print(f"    cross-turn@{k}: turn1={c['reach_at_turn1']:.3f}  "
                          f"cumulative={['%.3f' % x for x in c['cumulative_by_turn']]}  "
                          f"later-turn gain={c['extra_from_later_turns']:.3f}")

        results[label] = entry

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
