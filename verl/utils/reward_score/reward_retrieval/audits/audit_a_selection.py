# SPDX-License-Identifier: Apache-2.0
"""Audit A — selection-headroom audit (ROADMAP_v7.md sec 2).

THE DECISIVE GATE. Question: on the cases where the ground-truth item IS
present in the retrieved documents (the "winnable set"), is GT distinguishable
from the other candidate items in those docs by a TRANSFERABLE feature
(frequency / co-occurrence with the user's history / CF-adjacency) -- as
opposed to only by a train-only positional cue the model currently overfits?

    some transferable feature ranks GT high  -> selection is reward-addressable
                                                -> green-light r_select / r_infer
    GT indistinguishable from distractors    -> selection ceiling ~ 1/#candidates
                                                -> no reward fixes it on this corpus
                                                -> re-scope with the PI

CORPUS: never touched. This reads ONLY saved rollout logs (predictions.json),
i.e. text the model already produced, including the documents the retriever
already returned into the rollout. No corpora.jsonl, no e5_Flat.index, no
embedding model, no GPU, no training. Pure counting -- runs in stock python3.

Usage:
    python3 audit_a_selection.py \
        --preds outputs/eval/<exp>/global_step_300/test_predictions.json \
        --label v6-test
    # compare train vs test to see whether the winning feature is the one the
    # model actually uses, or a train-only cue:
    python3 audit_a_selection.py --preds <train preds> --label v6-train
"""

import argparse
import json
import random
import re
from collections import defaultdict

QUOTED = re.compile(r'"([^"]+)"')
TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
HISTORY = re.compile(
    r"The user has played the following musics before:\s*(.*?),\s*please write", re.S
)
# Documents inside a tool_response, as built by
# verl/tools/utils/search_r1_like_utils.py::_passages2string
DOC_SPLIT = re.compile(r"Doc\s+\d+\s+\(Title:\s*", re.S)


def norm(t):
    return re.sub(r"\s+", " ", t.strip().strip('"').strip()).lower()


def docs_in_rollout(rollout):
    """Recover each retrieved document's raw text, in rollout order.

    Split on the "Doc N (Title: " headers rather than trying to regex a
    balanced closing paren -- album titles themselves contain parentheses
    (e.g. 'Greatest Hits (Deluxe Edition)'), so a non-greedy '\\)' capture
    truncates the title. Splitting is robust to that.
    """
    docs = []
    for resp in TOOL_RESPONSE.findall(rollout):
        body = resp.strip()
        try:
            result = json.loads(body).get("result", "")
        except (json.JSONDecodeError, AttributeError):
            result = body  # tolerate malformed JSON, as coverage_sweep.py does
        if not isinstance(result, str):
            continue
        parts = DOC_SPLIT.split(result)[1:]  # drop preamble before "Doc 1 ("
        for p in parts:
            p = re.sub(r"\n?-{2,}\s*$", "", p).strip()
            if p:
                docs.append(p)
    return docs


def load(path):
    """Yield one parsed sample per saved rollout."""
    for rec in json.load(open(path)):
        rollout = rec["predict"][0] if isinstance(rec["predict"], list) else rec["predict"]
        m = HISTORY.search(rec["input"])
        history = [norm(t) for t in QUOTED.findall(m.group(1))] if m else []
        a = ANSWER.findall(rollout)
        # the model's committed pick (first <answer>; matches how eval grounds it)
        pred = norm(QUOTED.findall(a[0])[0]) if (a and QUOTED.findall(a[0])) else (
            norm(a[0]) if a else "")
        docs = docs_in_rollout(rollout)
        yield {
            "gt": norm(rec["output"]),
            "history": history,
            "pred": pred,
            "docs": docs,
            "doc_items": [[norm(t) for t in QUOTED.findall(d)] for d in docs],
            # prompt identity, so pooling multiple decodes of the SAME test set
            # can be bootstrapped by CLUSTER (prompt), not by rollout -- 8 decodes
            # of one prompt are not 8 independent samples.
            "pid": rec["input"],
        }


def features(sample, exclude_history):
    """Transferable-feature scores for every candidate item in the retrieved docs.

    All features are computable at TEST time from the rollout alone -- no
    train-only information, no corpus access. `pos_first` is the deliberate
    control: it is the positional cue we suspect the model actually overfits.

    `exclude_history` drops items the user has already played. This is the
    realistic candidate set -- a recommender never re-recommends a seen item --
    and it matters enormously here: the user's own history items trivially
    maximize `freq` and `cooc_hist` (they co-occur with the history by
    definition), so leaving them in floods the top of every ranking with
    already-played items. On TRAIN this is severe, because the retriever
    surfaces the user's OWN history document.
    """
    doc_items, history = sample["doc_items"], sample["history"]
    hist_set, tail_set = set(history), set(history[-3:])

    feats = defaultdict(lambda: defaultdict(float))
    for di, items in enumerate(doc_items):
        for pos, c in enumerate(items):
            if exclude_history and c in hist_set:
                continue
            f = feats[c]
            f["freq"] += 1.0
            # co-occurrence: how much of this user's history sits in a doc that
            # also contains c (the "users like you also played c" signal)
            f["cooc_hist"] += len(set(items) & hist_set)
            f["cooc_tail3"] += len(set(items) & tail_set)
            # CF-adjacency: c directly FOLLOWS a history item in this doc's
            # play sequence -- "users who played X next played c"
            if pos > 0 and items[pos - 1] in hist_set:
                f["adjacency"] += 1.0
            # positional cue (control): earlier appearance = higher score
            f.setdefault("pos_first", -(di * 100.0 + pos))
    return feats


FEATURES = ["freq", "cooc_hist", "cooc_tail3", "adjacency", "pos_first", "random"]
# the genuinely transferable (test-computable, non-positional) features -- the
# ones whose CI-lift-over-chance decides whether selection is reward-addressable
TRANSFERABLE = ["freq", "cooc_hist", "cooc_tail3", "adjacency"]


def rank_of(gt, feats, name, rng):
    """Expected rank of GT among candidates under `name`, honest about ties.

    Ties are broken in expectation (a feature that gives every candidate the
    same score must not be credited with 'ranking GT first'). Returns
    (expected_rank, p_hit_at_1, n_candidates).
    """
    cands = list(feats)
    n = len(cands)
    if n == 0 or gt not in feats:
        return None
    if name == "random":
        scores = {c: rng.random() for c in cands}
    else:
        scores = {c: feats[c].get(name, 0.0) for c in cands}
    g = scores[gt]
    better = sum(1 for c in cands if scores[c] > g)
    tied = sum(1 for c in cands if scores[c] == g)  # includes GT
    exp_rank = better + (tied + 1) / 2.0
    p_at_1 = (1.0 / tied) if better == 0 else 0.0
    return exp_rank, p_at_1, n


def boot_ci(vals, rng, n_boot=5000):
    """Percentile bootstrap 95% CI of the mean -- the winnable set is small
    (~25-30 per test checkpoint), so a point estimate alone is not decidable."""
    if not vals:
        return (float("nan"), float("nan"))
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def paired_boot_ci(diffs, rng, n_boot=5000):
    """95% CI of the MEAN PAIRED DIFFERENCE (a feature's per-sample hit@1 minus
    the same sample's uniform-random floor 1/#candidates). This is the decidable
    form of Audit A: if the CI excludes 0, the feature separates GT from the
    other candidates BEYOND CHANCE on the winnable set -- i.e. selection is
    reward-addressable. Paired (not two-sample) because both quantities are
    computed on the very same winnable rollouts."""
    if not diffs:
        return (float("nan"), float("nan"), float("nan"))
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (sum(diffs) / n,
            means[int(0.025 * n_boot)],
            means[int(0.975 * n_boot)])


def _clusters(pids):
    """Map cluster key (prompt) -> list of sample indices sharing it."""
    g = defaultdict(list)
    for i, p in enumerate(pids):
        g[p].append(i)
    return list(g.values())


def cluster_boot_ci(vals, pids, rng, n_boot=5000):
    """95% CI of the mean under a CLUSTER bootstrap: resample whole prompts
    (with all their pooled decodes) rather than individual rollouts, so
    pooling N decodes of the same test set does not fake N-fold independence.
    Reduces to the plain bootstrap when every prompt is unique."""
    groups = _clusters(pids)
    if not vals:
        return (float("nan"), float("nan"))
    k = len(groups)
    means = []
    for _ in range(n_boot):
        idx = [i for _ in range(k) for i in groups[rng.randrange(k)]]
        means.append(sum(vals[i] for i in idx) / len(idx))
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def cluster_paired_boot_ci(diffs, pids, rng, n_boot=5000):
    """Cluster-bootstrap 95% CI of the mean paired difference (feature hit@1
    minus per-sample chance floor). CI excludes 0 => feature beats chance even
    after accounting for prompt-level correlation across pooled decodes."""
    groups = _clusters(pids)
    if not diffs:
        return (float("nan"), float("nan"), float("nan"))
    k = len(groups)
    means = []
    for _ in range(n_boot):
        idx = [i for _ in range(k) for i in groups[rng.randrange(k)]]
        means.append(sum(diffs[i] for i in idx) / len(idx))
    means.sort()
    return (sum(diffs) / len(diffs),
            means[int(0.025 * n_boot)],
            means[int(0.975 * n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, nargs="+",
                    help="one or more predictions.json; multiple are POOLED "
                         "(the winnable set per checkpoint is too small to decide on)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="audit_a_results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    samples = []
    for p in args.preds:
        samples.extend(load(p))
    n_total = len(samples)

    retrieved = [s for s in samples if s["docs"]]
    winnable = [s for s in retrieved if s["gt"] in {c for d in s["doc_items"] for c in d}]

    # --- the two factors of test HR = coverage x selection --------------- #
    coverage = len(winnable) / n_total if n_total else 0.0
    sel_hits = sum(1 for s in winnable if s["pred"] == s["gt"])
    selection = sel_hits / len(winnable) if winnable else 0.0
    hr1 = sum(1 for s in samples if s["pred"] == s["gt"]) / n_total if n_total else 0.0
    # does the model ever answer correctly WITHOUT doc support? (copier check)
    off_doc = sum(
        1 for s in samples
        if s["pred"] == s["gt"] and s["gt"] not in {c for d in s["doc_items"] for c in d}
    )

    # GT is sometimes a title the user already played (Amazon title collisions:
    # "Greatest Hits" etc). Those samples are destroyed by the unseen filter, so
    # size the effect explicitly rather than hiding it.
    gt_in_hist = sum(1 for s in winnable if s["gt"] in set(s["history"]))

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    res = {
        "label": args.label,
        "preds": args.preds,
        "n_samples": n_total,
        "n_with_retrieval": len(retrieved),
        "coverage_gt_in_docs": round(coverage, 4),
        "selection_p_hit_given_gt_in_docs": round(selection, 4),
        "hr@1": round(hr1, 4),
        "hits_without_doc_support": off_doc,
        "n_winnable": len(winnable),
        "gt_is_a_replayed_title": gt_in_hist,
        "views": {},
    }

    print(f"\n{'='*78}\nAUDIT A — selection headroom   [{args.label}]\n{'='*78}")
    print(f"samples={n_total}  with_retrieval={len(retrieved)}")
    print(f"  coverage  P(GT in retrieved docs)      = {coverage:6.1%}   <- corpus-fixed, off-limits")
    print(f"  selection P(model picks GT | in docs)  = {selection:6.1%}   <- the reward's lever")
    print(f"  => HR@1 = {hr1:.4f}   (hits without doc support: {off_doc})")
    print(f"\nwinnable set: n={len(winnable)}   (GT is an already-played title in {gt_in_hist})")

    # --- Audit A proper: can any transferable feature rank GT on the winnable set? --- #
    # The decision is per-feature: does its hit@1 beat the uniform-random floor
    # (1/#candidates) by a paired-bootstrap 95% CI that excludes 0? `pos_first`
    # is the positional control (train-only cue); `random` is the sanity floor.
    for view, excl in (("ALL items in docs", False), ("UNSEEN items only", True)):
        p1 = {f: [] for f in FEATURES}   # sample-aligned per-feature hit@1
        ranks = {f: [] for f in FEATURES}
        floors, n_cands, pids, n_scored = [], [], [], 0
        for s in winnable:
            feats = features(s, exclude_history=excl)
            if s["gt"] not in feats:
                continue  # unseen filter removed GT (it was a replayed title)
            n_scored += 1
            ncand = len(feats)
            n_cands.append(ncand)
            floors.append(1.0 / ncand)  # expected P(GT ranked 1st) under chance
            pids.append(s["pid"])
            for f in FEATURES:
                r = rank_of(s["gt"], feats, f, rng)
                p1[f].append(r[1] if r else 0.0)
                ranks[f].append(r[0] if r else float(ncand))

        n_clusters = len(set(pids))
        feat_out = {}
        for f in FEATURES:
            lo, hi = cluster_boot_ci(p1[f], pids, rng)
            dmean, dlo, dhi = cluster_paired_boot_ci(
                [a - b for a, b in zip(p1[f], floors)], pids, rng)
            feat_out[f] = {
                "mean_expected_rank": round(mean(ranks[f]), 3),
                "hit@1": round(mean(p1[f]), 4),
                "hit@1_ci95": [round(lo, 4), round(hi, 4)],
                "mrr": round(mean([1.0 / r for r in ranks[f]]), 4),
                "lift_vs_chance": round(dmean, 4),
                "lift_vs_chance_ci95": [round(dlo, 4), round(dhi, 4)],
                "beats_chance": bool(dlo > 0),
            }
        winners = [f for f in TRANSFERABLE if feat_out[f]["beats_chance"]]
        v = {
            "n_scored": n_scored,
            "n_prompt_clusters": n_clusters,
            "mean_n_candidates": round(mean(n_cands), 2),
            "uniform_floor_hit@1": round(mean(floors), 4) if floors else float("nan"),
            "transferable_beating_chance": winners,
            "features": feat_out,
        }
        res["views"][view] = v

        print(f"\n  --- candidate set: {view} --- "
              f"(n={n_scored} rollouts / {n_clusters} distinct prompts, "
              f"mean #candidates={mean(n_cands):.1f}, "
              f"uniform floor hit@1={v['uniform_floor_hit@1']:.1%})")
        print(f"  {'feature':<12} {'exp.rank':>8} {'hit@1':>7} {'hit@1 95%CI':>15} "
              f"{'lift>chance 95%CI':>20} sig")
        for f in FEATURES:
            d = feat_out[f]
            tag = ""
            if f == "pos_first":
                tag = "  <- positional control"
            if f == "random":
                tag = "  <- floor"
            ci, lci = d["hit@1_ci95"], d["lift_vs_chance_ci95"]
            sig = "***" if d["beats_chance"] else ""
            print(f"  {f:<12} {d['mean_expected_rank']:>8.2f} {d['hit@1']:>7.1%} "
                  f"[{ci[0]:>5.1%},{ci[1]:>5.1%}] "
                  f"[{lci[0]:>+6.3f},{lci[1]:>+6.3f}] {sig:>3}{tag}")
        print(f"  => transferable features with 95%-CI lift over chance: "
              f"{winners or 'NONE'}")

    with open(args.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"\nappended -> {args.out}")


if __name__ == "__main__":
    main()
