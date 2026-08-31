# SPDX-License-Identifier: Apache-2.0
"""Probe (b) -- CAN the policy discriminate the ground truth among its candidates?

`decode_selection.py` answers this with one binary event: among rollouts where
the GT was actually retrieved, did the model emit it? That statistic is honest
but nearly powerless -- ~35 winnable cases per 1000-prompt decode, ~6% of which
land, so the arms tie at p=0.71 and we learn nothing about *ability*.

This probe replaces the binary with a graded one and buys back roughly 30x the
sample size.

    THE STATISTIC. For a rollout that answered with one of its retrieved items,
    take the candidate set it chose from, embed every candidate and the GT with
    the SAME encoder eval.py scores with (data/amazon_data/<ds>/embeddings.pt),
    and ask where the model's pick sits in the candidate similarity-to-GT
    ordering:

        pct = ( #{c : sim(c,GT) < sim(pick,GT)} + 0.5 ) / |candidates|

    Under the null "the policy picks uniformly among its candidates" this is
    exactly Uniform-distributed with E[pct] = 0.5 for EVERY candidate-set size,
    so no size correction is needed. E[pct] > 0.5 means the policy systematically
    reaches toward the GT even when it does not land on it -- latent
    discriminative signal that a selection reward could amplify. E[pct] = 0.5
    means it is picking at random and there is nothing to amplify.

    WHY IT IS 30x BETTER POWERED. The binary test needs GT to be *in* the
    candidate set (~3.5% of rollouts). The graded test only needs GT to be
    *known*, which it always is -- so it runs on every grounded rollout. That is
    the difference between ~35 and ~700 usable events per decode.

Also reported, because both are cheap and both are alternative explanations for
below-chance selection:
  * the binary P(pick GT | covered) against its own 1/|candidates| chance bar
  * POSITION BIAS -- where the pick sits in *retrieval order*. If the policy just
    takes the first item of the first document, selection is a formatting
    artifact, not a capability ceiling, and the fix is a prompt change.
  * a RANDOM-PICK CONTROL that re-runs the whole pipeline choosing uniformly.
    It must come out at 0.5; if it does not, the instrument is broken.

Repeats of one checkpoint are correlated (same prompt, 3 decodes), so the test
aggregates within a prompt FIRST and then treats prompts as the independent
unit.

Usage (needs torch for embeddings.pt -- use the retriever env):

    /work/11138/pranavbelligundu/vista/anaconda/envs/retriever/bin/python \\
        selection_capability_probe.py \\
        --dataset-dir data/amazon_data/CDs_and_Vinyl \\
        --preds outputs/eval/<arm>/global_step_500/decode*/test_predictions.json \\
        --json capability_probe.json
"""

import argparse
import importlib.util
import json
import math
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load("decode_behavior", os.path.join(_HERE, "decode_behavior.py"))
selection = _load("v8_selection", os.path.join(os.path.dirname(_HERE), "v8", "selection.py"))
dsel = _load("decode_selection", os.path.join(_HERE, "decode_selection.py"))


def _ordered_items(seqs):
    """Retrieved items in first-appearance (retrieval) order, deduped.

    decode_selection.py uses a set; position bias needs the order, so rebuild it
    here rather than changing the shared helper.
    """
    out, seen = [], set()
    for s in seqs:
        for i in s:
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


def _mid_rank_pct(sims, pick_idx):
    """(#strictly-below + 0.5) / m -- Uniform with mean 0.5 under random picking."""
    m = len(sims)
    sp = sims[pick_idx]
    below = sum(1 for i, s in enumerate(sims) if s < sp)
    ties = sum(1 for i, s in enumerate(sims) if s == sp)
    # split the tie block evenly so exact duplicates cannot bias the statistic
    return (below + ties / 2.0) / m


def _tstat(xs, mu=0.5):
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return dict(n=n, mean=mean, sd=0.0, t=None, p=None)
    se = math.sqrt(var / n)
    t = (mean - mu) / se
    # two-sided normal approximation (n is in the hundreds/thousands here)
    p = math.erfc(abs(t) / math.sqrt(2))
    return dict(n=n, mean=mean, sd=math.sqrt(var), se=se, t=t, p=p,
                ci95=(mean - 1.96 * se, mean + 1.96 * se))


def analyze(pred_paths, name2id, emb, rng):
    import torch

    gt_pct, rand_pct, first_pct = {}, {}, {}   # prompt -> list of percentiles
    pos_of_pick, n_cand_hist = [], []
    n_seen = n_grounded = n_usable = 0
    n_cov = n_cov_ans = n_pick_gt = 0
    sum_items = 0

    for path in sorted(pred_paths):
        for pid, rec in enumerate(json.load(open(path, encoding="utf-8"))):
            n_seen += 1
            text = "".join(rec.get("predict") or [])
            seqs = selection.doc_sequences(db._retrieved_spans(text))
            cands = _ordered_items(seqs)
            gt = selection.norm_title(rec.get("output", ""))
            pick = dsel._pred_title(text)
            sum_items += len(cands)

            if gt and gt in set(cands):
                n_cov += 1
                if pick:
                    n_cov_ans += 1
                    if pick == gt:
                        n_pick_gt += 1
            if not pick or pick not in cands:
                continue
            n_grounded += 1

            # the graded test needs the GT and >=2 candidates embeddable
            if gt not in name2id:
                continue
            rows = [name2id[c] for c in cands if c in name2id]
            keep = [c for c in cands if c in name2id]
            if len(rows) < 2 or pick not in keep:
                continue

            g = emb[name2id[gt]]
            M = emb[rows]
            sims = torch.nn.functional.cosine_similarity(M, g.unsqueeze(0), dim=1).tolist()
            pi = keep.index(pick)

            n_usable += 1
            gt_pct.setdefault(pid, []).append(_mid_rank_pct(sims, pi))
            rand_pct.setdefault(pid, []).append(_mid_rank_pct(sims, rng.randrange(len(keep))))
            # Does RETRIEVAL ORDER itself carry GT signal? If the first-returned
            # candidate scores > 0.5 then the policy's primacy bias is pointing at
            # something real and re-ranking/prompt work is a live fix; if it sits
            # at 0.5 the bias is inert and there is no ordering to exploit.
            first_pct.setdefault(pid, []).append(_mid_rank_pct(sims, 0))
            pos_of_pick.append(cands.index(pick) / max(1, len(cands) - 1) if len(cands) > 1 else 0.0)
            n_cand_hist.append(len(keep))

    # collapse repeats within a prompt, then treat prompts as independent
    per_prompt = [sum(v) / len(v) for v in gt_pct.values()]
    per_prompt_rand = [sum(v) / len(v) for v in rand_pct.values()]
    per_prompt_first = [sum(v) / len(v) for v in first_pct.values()]

    items_mean = sum_items / n_seen if n_seen else 0.0
    return dict(
        n_rollouts=n_seen, n_grounded=n_grounded, n_usable_graded=n_usable,
        n_prompts=len(per_prompt),
        graded=_tstat(per_prompt),
        random_control=_tstat(per_prompt_rand),
        first_candidate=_tstat(per_prompt_first),
        position_of_pick=_tstat(pos_of_pick) if pos_of_pick else None,
        mean_candidates=sum(n_cand_hist) / len(n_cand_hist) if n_cand_hist else 0.0,
        binary=dict(n_cov=n_cov, n_cov_ans=n_cov_ans, n_pick_gt=n_pick_gt,
                    pick_gt=(n_pick_gt / n_cov_ans) if n_cov_ans else 0.0,
                    chance=(1.0 / items_mean) if items_mean else 0.0,
                    lift=((n_pick_gt / n_cov_ans) * items_mean) if n_cov_ans and items_mean else 0.0),
        first_position_share=(sum(1 for p in pos_of_pick if p == 0.0) / len(pos_of_pick))
        if pos_of_pick else 0.0,
    )


def calibrate(pred_paths, name2id, emb, rng, betas):
    """How much percentile shift would the 1.55x top-k break-even bar need?

    A bare "mean pct = 0.505, p = 0.8" says the effect is not distinguishable
    from zero but not whether zero-vs-detectable would even matter. So replay
    the REAL candidate sets under a soft preference model -- pick candidate i
    with probability proportional to exp(beta * sim(i, GT)) -- and sweep beta
    from 0 (uniform, the null) upward, reporting both the percentile statistic
    this probe measures and the selection lift the roadmap gates on. That maps
    the observed percentile onto the decision scale.

    beta is an ORACLE knob: it uses similarity to the GT, which the policy
    cannot see. It is a yardstick for effect size, not a proposed method.
    """
    import torch

    sets = []
    for path in sorted(pred_paths):
        for rec in json.load(open(path, encoding="utf-8")):
            text = "".join(rec.get("predict") or [])
            cands = _ordered_items(selection.doc_sequences(db._retrieved_spans(text)))
            gt = selection.norm_title(rec.get("output", ""))
            if gt not in name2id:
                continue
            keep = [c for c in cands if c in name2id]
            if len(keep) < 2:
                continue
            sims = torch.nn.functional.cosine_similarity(
                emb[[name2id[c] for c in keep]], emb[name2id[gt]].unsqueeze(0), dim=1)
            sets.append((sims, keep.index(gt) if gt in keep else -1))
    if not sets:
        return []

    # One multinomial draw per candidate set leaves the lift estimate ~1 binomial
    # draw wide over the ~100 covered sets (beta=0.5 -> 1.95x, beta=0.8 -> 1.46x
    # is pure sampling noise). Average the EXACT per-set probabilities instead of
    # sampling the hit, and average the percentile over R draws.
    R = 32
    rows = []
    for beta in betas:
        pcts, p_hit, cov, chance = [], 0.0, 0, 0.0
        for sims, gi in sets:
            p = torch.softmax(beta * sims, dim=0)
            sl = sims.tolist()
            for j in torch.multinomial(p, R, replacement=True).tolist():
                pcts.append(_mid_rank_pct(sl, j))
            if gi >= 0:
                cov += 1
                chance += 1.0 / len(sims)
                p_hit += float(p[gi])          # exact E[hit], no sampling noise
        rows.append(dict(beta=beta, mean_pct=sum(pcts) / len(pcts),
                         n_covered=cov, pick_gt=p_hit / cov if cov else 0.0,
                         chance=chance / cov if cov else 0.0,
                         lift=(p_hit / chance) if chance else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true",
                    help="sweep an oracle preference strength to map percentile -> lift")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--group-by-step", action="store_true",
                    help="report one row per global_step_* directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    import torch
    name2id = json.load(open(os.path.join(args.dataset_dir, "name2id.json"), encoding="utf-8"))
    name2id = {selection.norm_title(k): v for k, v in name2id.items()}
    emb = torch.load(os.path.join(args.dataset_dir, "embeddings.pt"), map_location="cpu")
    emb = emb.float()

    groups = {}
    if args.group_by_step:
        for p in args.preds:
            groups.setdefault(os.path.dirname(os.path.dirname(os.path.abspath(p))), []).append(p)
    else:
        groups["ALL"] = args.preds

    rows = []
    for key, paths in sorted(groups.items()):
        rng = random.Random(args.seed)
        r = analyze(paths, name2id, emb, rng)
        r["group"] = os.path.relpath(key) if key != "ALL" else "ALL"
        rows.append(r)

    print("=" * 100)
    print("PROBE (b) -- can the policy discriminate the GT among its own candidates?")
    print("=" * 100)
    print("H0: picks uniformly among candidates  ->  mean percentile = 0.500\n")
    hdr = (f"  {'group':<44} {'Nprompt':>7} {'graded':>8} {'95% CI':>17} {'p':>9} "
           f"{'rand-ctl':>9} {'first-cand':>11} {'pos':>6}")
    print(hdr)
    for r in rows:
        g, rc = r["graded"], r["random_control"]
        if not g:
            print(f"  {r['group'][-44:]:<44}  (no usable rollouts)")
            continue
        ci = f"[{g['ci95'][0]:.3f},{g['ci95'][1]:.3f}]"
        pos = r["position_of_pick"]["mean"] if r["position_of_pick"] else float("nan")
        print(f"  {r['group'][-44:]:<44} {r['n_prompts']:>7} {g['mean']:>8.4f} {ci:>17} "
              f"{g['p']:>9.2e} {rc['mean']:>9.4f} "
              f"{r['first_candidate']['mean']:>11.4f} {pos:>6.3f}")

    print("\n  binary check (P(pick GT | covered) vs 1/|candidates|):")
    for r in rows:
        b = r["binary"]
        print(f"    {r['group'][-44:]:<44} {b['n_pick_gt']}/{b['n_cov_ans']} = {b['pick_gt']:.4f}  "
              f"chance {b['chance']:.4f}  lift {b['lift']:.2f}x")
    print("\n  first-position share (pick == first retrieved item):")
    for r in rows:
        print(f"    {r['group'][-44:]:<44} {r['first_position_share']:.4f}  "
              f"(mean |candidates| {r['mean_candidates']:.1f})")

    cal = None
    if args.calibrate:
        cal = calibrate(args.preds, name2id, emb, random.Random(args.seed),
                        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
        print("\n  calibration -- oracle preference sweep (percentile -> selection lift):")
        print(f"    {'beta':>6} {'mean_pct':>9} {'pick_gt':>8} {'chance':>8} {'lift':>7}")
        for r in cal:
            print(f"    {r['beta']:>6.1f} {r['mean_pct']:>9.4f} {r['pick_gt']:>8.4f} "
                  f"{r['chance']:>8.4f} {r['lift']:>7.2f}x")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "calibration": cal}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
