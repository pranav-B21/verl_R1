# SPDX-License-Identifier: Apache-2.0
"""Correlation audit -- roadmap ROADMAP_v7.md sec 2 "Audit B" (correctness-correlation).

THE v2 TRIPWIRE. For every candidate reward TERM, compute it on logged rollouts
and correlate it with the outcome (`InTop@n` hit). A term that does not clearly,
positively correlate with correctness is NOISE and must be discarded BEFORE it
costs a GPU-week -- exactly the mistake v2 made (its redundancy penalty
correlated with correctness at only +0.10 and made the model worse).

NAMING: this implements the roadmap's "Audit B". It is DISTINCT from the
train-vs-test greedy diagnostic in ``reward_reasoning/diagnostics/`` (which some
notes informally call "Audit B"); that one is already run and decisive. To avoid
the bare-letter clash this file is named by function, `correlation_audit.py`.

Two candidate terms are scored here (per the scoped ask 2026-07-14):
  (a) r_infer   -- cosine(answer, centroid of retrieved CF-neighbour items).
                   The reasoning axis's lead term (ROADMAP_v7.md sec 3.1). Uses
                   NO ground truth -- it scores grounding in retrieved evidence.
  (b) r_retqual -- v7's shipped retrieval-quality term: per-turn cosine of the
                   best retrieved doc to GT, tau/floor-shaped, mean over turns
                   (ROADMAP_v7.md sec 4.1). The retrieval axis's gate. Also
                   reports the raw un-shaped `best_sim` = max_doc cos(doc, GT).

CORPUS: never touched. Reads only saved rollout logs (predictions.json) -- the
text the model already produced, including the docs the retriever already
returned. The GROUNDING catalog (embeddings.pt / name2id.json) is used only to
compute the `InTop@n` correctness LABEL, exactly as reward_SPRec.py / eval.py do
-- that is the frozen grounding, not the retrieval corpus. No GPU required
(single-thread CPU; the embedding model is a 3-layer MiniLM).

Usage (run in an env with torch+transformers, e.g. the `retriever` conda env;
the login node needs single-thread limits or it OOMs on thread-local storage):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /work/.../anaconda/envs/retriever/bin/python correlation_audit.py \\
        --preds <one or more test_predictions.json> --data-source amazon \\
        --label v6-test --limit 3000
"""

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict

# ---- keep the login node from exhausting thread-local storage on torch ------ #
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

# reuse the exact parsing the selection audit already validated against the logs
from audit_a_selection import (  # noqa: E402
    ANSWER, HISTORY, QUOTED, TOOL_RESPONSE, DOC_SPLIT, norm,
)

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
# reward_SPRec.py extracts the LAST <answer> then its first quoted span; match it
# so our correctness label equals the outcome reward's, not a variant.
FIRST_QUOTE = re.compile(r'"([^"]*)')


# --------------------------- embedding (mean-pool MiniLM) -------------------- #
class Embedder:
    """paraphrase-MiniLM-L3-v2 via transformers, mean-pooled -- byte-equivalent
    to SentenceTransformer('...paraphrase-MiniLM-L3-v2') (that ST model is just
    Transformer + mean Pooling, no dense/normalize), so `raw` matches the
    un-normalised embeddings.pt catalog (row-norm ~5.75) for cdist ranking, and
    `normalize=True` gives cosine space for the similarity terms."""

    def __init__(self, name="sentence-transformers/paraphrase-MiniLM-L3-v2"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.mdl = AutoModel.from_pretrained(name).eval()
        self._cache = {}

    @torch.no_grad()
    def _encode(self, texts):
        b = self.tok(texts, padding=True, truncation=True, max_length=256,
                     return_tensors="pt")
        out = self.mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return (out * m).sum(1) / m.sum(1).clamp(min=1e-9)  # raw mean-pool

    def embed(self, texts, normalize=False, batch=64):
        """Cached, batched. Returns [len(texts), 384]."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            vecs = self._encode(chunk)
            for t, v in zip(chunk, vecs):
                self._cache[t] = v
        out = torch.stack([self._cache[t] for t in texts]) if texts else torch.empty(0, 384)
        return F.normalize(out, dim=1) if normalize else out


# --------------------------------- parsing ---------------------------------- #
def parse_rollout(rec):
    """One logged rollout -> the fields every candidate term needs."""
    roll = rec["predict"][0] if isinstance(rec["predict"], list) else rec["predict"]

    # predicted title: LAST <answer>, first quoted span (reward_SPRec convention)
    ans = ANSWER.findall(roll)
    pred = ""
    if ans:
        q = FIRST_QUOTE.search(ans[-1])
        pred = norm(q.group(1)) if q else norm(ans[-1])

    m = HISTORY.search(rec["input"])
    history = {norm(t) for t in QUOTED.findall(m.group(1))} if m else set()

    # per-turn retrieved docs, in rollout order (for r_retqual)
    calls, resps = TOOL_CALL.findall(roll), TOOL_RESPONSE.findall(roll)
    turns_docs = []
    for resp in resps[:min(len(calls), len(resps))]:
        body = resp.strip()
        try:
            result = json.loads(body).get("result", "")
        except (json.JSONDecodeError, AttributeError):
            result = body
        if not isinstance(result, str):
            turns_docs.append([])
            continue
        docs = []
        for p in DOC_SPLIT.split(result)[1:]:
            p = re.sub(r"\n?-{2,}\s*$", "", p).strip()
            if p:
                docs.append(p)
        turns_docs.append(docs)

    # CF-neighbour item set E: quoted titles across all retrieved docs, minus the
    # user's own history (those are the query, not neighbours) -- for r_infer
    E = set()
    for docs in turns_docs:
        for d in docs:
            for it in QUOTED.findall(d):
                it = norm(it)
                if it and it not in history:
                    E.add(it)

    return {
        "pred": pred,
        "gt": norm(rec["output"]),
        "gt_raw": rec["output"].strip().strip('"'),
        "turns_docs": turns_docs,
        "E": sorted(E),
    }


# ------------------------------- the terms ---------------------------------- #
def r_infer(sample, emb):
    """cos(answer, centroid of retrieved CF-neighbour items). None if undefined.
    Grounded strictly in the RETRIEVED items (not a global-catalog embedding of
    GT) -- the v4-mush guard from ROADMAP_v7.md sec 3.1."""
    if not sample["pred"] or not sample["E"]:
        return None
    item_vecs = emb.embed(sample["E"], normalize=True)
    centroid = F.normalize(item_vecs.mean(0, keepdim=True), dim=1)
    pv = emb.embed([sample["pred"]], normalize=True)
    return float((pv @ centroid.T).item())


def _per_turn_best_sims(sample, emb):
    """Per-turn cosine of the best retrieved doc to GT, in rollout order, only for
    turns that retrieved >=1 doc. [] if no docs / no GT. Shared by r_retqual and
    r_covgain so both score exactly what `v7/retrieval.py` would."""
    gt = sample["gt_raw"]
    if not gt:
        return []
    gv = emb.embed([gt], normalize=True)
    per_turn = []
    for docs in sample["turns_docs"]:
        if not docs:
            continue
        dv = emb.embed(docs, normalize=True)
        per_turn.append(float((dv @ gv.T).squeeze(1).clamp(-1, 1).max().item()))
    return per_turn


def r_retqual(sample, emb, tau=0.30, floor=0.25):
    """v7's shipped term (replicated): per-turn best-doc cosine to GT, tau/floor
    two-sided shaping, mean over productive turns. Returns (shaped, raw_best_sim).
    None if no docs were retrieved."""
    per_turn = _per_turn_best_sims(sample, emb)
    if not per_turn:
        return None, None

    def shape(s):
        if s >= tau:
            return (s - tau) / max(1e-6, 1.0 - tau)
        return -floor * (tau - s) / max(1e-6, tau)

    return sum(shape(s) for s in per_turn) / len(per_turn), max(per_turn)


def r_covgain(sample, emb):
    """v7's shipped `covgain_agg` (replicated exactly from
    v7/retrieval.py::compute_retrieval_components): sum of positive best-sim
    improvements a LATER turn makes over the running-best of earlier turns. Targets
    the 99% single-query collapse without rewarding retrieval-spam. 0.0 with <=1
    productive turn; None if nothing was retrieved (so the audit drops it, matching
    r_retqual's undefined-domain)."""
    per_turn = _per_turn_best_sims(sample, emb)
    if not per_turn:
        return None
    covgain, running_best = 0.0, None
    for s in per_turn:
        if running_best is not None:
            covgain += max(0.0, s - running_best)
        running_best = s if running_best is None else max(running_best, s)
    return float(covgain)


# --------------------- correctness label: InTop@n via catalog --------------- #
def load_catalog(data_source):
    if "amazon" in data_source:
        base = "data/amazon_data/CDs_and_Vinyl"
    elif "goodreads" in data_source:
        base = "data/goodreads_data/Goodreads"
    else:
        raise ValueError(f"no catalog for data_source={data_source}")
    emb = torch.as_tensor(torch.load(f"{base}/embeddings.pt", map_location="cpu",
                                     weights_only=False)).float()
    name2id = json.load(open(f"{base}/name2id.json"))
    return emb, name2id


def intop_ranks(preds, emb, catalog, name2id, gts_raw, chunk=128):
    """rankId (1-based) of GT among the 13k catalog items under L2 distance from
    the predicted title's embedding -- exactly reward_SPRec.similarity_match.
    None when GT is not in name2id (F9 case) or the answer was empty."""
    ranks = []
    pv_all = emb.embed(preds, normalize=False) if preds else torch.empty(0, 384)
    for i in range(0, len(preds), chunk):
        pv = pv_all[i:i + chunk]
        d = torch.cdist(pv, catalog, p=2)          # [chunk, N]
        order = d.argsort(dim=1)                    # ascending distance
        # position of each row's GT id
        for j in range(pv.shape[0]):
            gt = gts_raw[i + j]
            if not preds[i + j] or gt not in name2id:
                ranks.append(None)
                continue
            tid = name2id[gt]
            pos = (order[j] == tid).nonzero(as_tuple=False)
            ranks.append(int(pos.item()) + 1 if pos.numel() else len(name2id) + 1)
    return ranks


# ------------------------------ statistics ---------------------------------- #
def _pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def _spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(ranks(x), ranks(y))


def _auc(scores, labels):
    """P(term(hit) > term(miss)); ties count 0.5. = normalised Mann-Whitney U."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    sum_pos = sum(rank[i] for i in range(len(scores)) if labels[i])
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _quartile_lift(scores, labels):
    """hit-rate in the top quartile of the term vs the bottom quartile."""
    idx = sorted(range(len(scores)), key=lambda i: scores[i])
    q = max(1, len(idx) // 4)
    bot = [labels[i] for i in idx[:q]]
    top = [labels[i] for i in idx[-q:]]
    return (sum(top) / len(top), sum(bot) / len(bot))


def report(name, scores, labels, hit_name):
    """scores/labels are aligned; drop samples where the term is undefined."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None and l is not None]
    if len(pairs) < 10:
        print(f"  {name:<16} vs {hit_name:<10}  n<10, skipped")
        return {}
    xs = [p[0] for p in pairs]
    ys = [1.0 if p[1] else 0.0 for p in pairs]
    n_pos = int(sum(ys))
    r, rho, auc = _pearson(xs, ys), _spearman(xs, ys), _auc(xs, [bool(y) for y in ys])
    top, bot = _quartile_lift(xs, [bool(y) for y in ys])
    verdict = "PASS" if (not math.isnan(r) and r > 0.10) else ("weak" if (not math.isnan(r) and r > 0.03) else "FAIL")
    print(f"  {name:<16} vs {hit_name:<10} n={len(pairs):>4} pos={n_pos:>3} "
          f"| pearson={r:+.3f} spearman={rho:+.3f} AUC={auc:.3f} "
          f"| Q4/Q1 hit={top:.3f}/{bot:.3f}  [{verdict}]")
    return {"n": len(pairs), "n_pos": n_pos, "pearson": round(r, 4),
            "spearman": round(rho, 4), "auc": round(auc, 4),
            "q4_hit": round(top, 4), "q1_hit": round(bot, 4), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, nargs="+")
    ap.add_argument("--data-source", default="amazon")
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else subsample")
    ap.add_argument("--out", default="correlation_audit_results.jsonl")
    args = ap.parse_args()
    t0 = time.time()

    recs = []
    for p in args.preds:
        recs.extend(json.load(open(p)))
    if args.limit and len(recs) > args.limit:
        recs = recs[:args.limit]
    print(f"[correlation audit: {args.label}] {len(recs)} rollouts from "
          f"{len(args.preds)} file(s)")

    emb = Embedder()
    catalog, name2id = load_catalog(args.data_source)
    print(f"  catalog {tuple(catalog.shape)}  (row-norm {catalog.norm(dim=1).mean():.2f})")

    samples = [parse_rollout(r) for r in recs]

    # candidate terms (per rollout)
    ri = [r_infer(s, emb) for s in samples]
    rq_pairs = [r_retqual(s, emb) for s in samples]
    rq = [p[0] for p in rq_pairs]
    best_sim = [p[1] for p in rq_pairs]
    rcg = [r_covgain(s, emb) for s in samples]

    # correctness labels
    ranks = intop_ranks([s["pred"] for s in samples], emb, catalog, name2id,
                        [s["gt_raw"] for s in samples])
    hr1 = [(1 if s["pred"] and s["pred"] == s["gt"] else 0) for s in samples]  # exact
    in1 = [(1 if (rk is not None and rk <= 1) else (None if rk is None else 0)) for rk in ranks]
    in10 = [(1 if (rk is not None and rk <= 10) else (None if rk is None else 0)) for rk in ranks]

    base_rate = sum(1 for x in in10 if x) / max(1, sum(1 for x in in10 if x is not None))
    print(f"  base rates: exactHR@1={sum(hr1)/len(hr1):.4f}  "
          f"InTop@1={sum(1 for x in in1 if x)/max(1,sum(1 for x in in1 if x is not None)):.4f}  "
          f"InTop@10={base_rate:.4f}  (embed+rank in {time.time()-t0:.0f}s)")

    out = {"label": args.label, "preds": args.preds, "n": len(samples),
           "results": {}}
    print("\n  term                 vs label       n     pos    correlations                         quartile-lift  verdict")
    for tname, tvals in (("r_infer", ri), ("r_retqual(shaped)", rq),
                         ("best_sim(raw)", best_sim), ("r_covgain", rcg)):
        for hname, hvals in (("exactHR@1", hr1), ("InTop@1", in1), ("InTop@10", in10)):
            key = f"{tname}|{hname}"
            out["results"][key] = report(tname, tvals, hvals, hname)

    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"\n  appended -> {args.out}   (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
