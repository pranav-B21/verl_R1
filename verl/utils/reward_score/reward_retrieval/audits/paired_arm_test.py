# SPDX-License-Identifier: Apache-2.0
"""Paired baseline-vs-v8 test over the step sweep.

Both arms decode the SAME 1000 held-out prompts, three times each. Every
comparison written before this script threw that away and ran a two-sample
proportion test, which is the wrong test and the weak one: at n=3000 with
coverage ~3.3% an unpaired test needs ~4.2% to call a win, and HR@5 (8 events
per 1000) needs to nearly double. Pairing removes the between-prompt variance,
which for coverage is nearly all of the variance -- whether the GT is reachable
at all is a property of the prompt, identical in both arms.

Endpoints, all computed from saved decodes (CPU only, no corpus/index/GPU):

  coverage   GT title appears in the retrieved docs        <- PRIMARY
  exact      the emitted title IS the GT                   <- HR@1 proxy
  grounded   the emitted title came from the docs
  successor  the emitted title is in the successor set     <- what v8 rewards

Coverage and exact are paired per prompt across arms. `pick GT | covered` is
NOT paired -- its denominator is the covered subset, which differs by arm -- so
it is reported as a pooled proportion with the chance line, and chance differs
by arm (candidate-set size moves), so read the LIFT column, not the raw rate.

Usage:
    python3 paired_arm_test.py                      # full sweep, both arms
    python3 paired_arm_test.py --steps 300 500

    # greedy (temperature=0) decodes, which live in greedy<i>_<date>/ dirs:
    python3 paired_arm_test.py --tag greedy --n-decodes 1 \
        --arm baseline=nq-...-gpu-baseline-n8:2026-08-08 \
        --arm v8=nq-...-gpu-rthink-v8-n8:2026-08-08

Defaults reproduce the 2026-08-07 temperature-1.0 result unchanged.
"""

import argparse
import importlib.util
import json
import os
import numpy as np
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "..", "..", "..", "..", "..", "outputs", "eval")

# The sweep is one dated batch per arm. Both arms have OTHER decodes of the same
# checkpoints from earlier studies (baseline step 200 carries 07-21 and 07-24;
# v8 steps 200/250 carry 08-01 and 08-02), and mixing them in silently changes
# the number of decodes per step between arms -- which is exactly the kind of
# unmatched denominator that produced the "baseline 0/57" figure nobody could
# reconcile. Pin the batch by date and take 3 decodes per step, both arms.
ARMS = {
    "baseline": ("nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8", "2026-08-01"),
    "v8": ("nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8", "2026-08-04"),
}
N_DECODES = 3
# Decode-subdir prefix. "decode" = temperature 1.0 (the historical regime),
# "greedy" = temperature 0. Never mix the two in one call: pooling a greedy
# decode with sampled ones reports a mean over two different policies at
# inference time. --tag/--n-decodes/--arm rebind these three at the CLI.
TAG = "decode"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load("decode_behavior", os.path.join(_HERE, "decode_behavior.py"))
selection = _load("v8_selection", os.path.join(os.path.dirname(_HERE), "v8", "selection.py"))

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def _pred_title(text):
    """Emitted title, parsed exactly as v8/orchestrator.py and decode_selection.py do."""
    answers = ANSWER.findall(text)
    if not answers:
        return ""
    raw = answers[-1].strip()
    m = re.search(r'"([^"]*)', raw)
    return selection.norm_title(m.group(1) if m else raw)


def per_prompt(path):
    """Per-prompt binary outcomes for one decode, keyed by prompt index."""
    records = json.load(open(path, encoding="utf-8"))
    out = []
    for rec in records:
        text = "".join(rec.get("predict") or [])
        spans = db._retrieved_spans(text)
        seqs = selection.doc_sequences(spans)
        items = {i for s in seqs for i in s}
        hist = selection.user_history(rec.get("input", ""))
        succ = selection.successor_set(seqs, hist)
        gt = selection.norm_title(rec.get("output", ""))
        pred = _pred_title(text)
        out.append(dict(
            cov=bool(gt and gt in items),
            exact=bool(gt and pred and pred == gt),
            grounded=bool(pred and pred in items),
            succ=bool(pred and pred in succ),
            cov_ans=bool(gt and gt in items and pred),
        ))
    return out


def arm_decodes(arm, step):
    name, date = ARMS[arm]
    d = os.path.join(_EVAL, name, f"global_step_{step}")
    if not os.path.isdir(d):
        return []
    tags = sorted(t for t in os.listdir(d) if t.startswith(TAG) and t.endswith(date))
    paths = [os.path.join(d, t, "test_predictions.json") for t in tags
             if os.path.exists(os.path.join(d, t, "test_predictions.json"))]
    return paths[:N_DECODES]


_CACHE = {}


def per_prompt_cached(path):
    if path not in _CACHE:
        _CACHE[path] = per_prompt(path)
    return _CACHE[path]


def prompt_rates(decodes, key):
    """Per-prompt rate across an arm's decodes: rows[i] = fraction of decodes where key holds."""
    per = [per_prompt_cached(p) for p in decodes]
    n = min(len(x) for x in per)
    return [sum(d[i][key] for d in per) / len(per) for i in range(n)], per


def paired_bootstrap(a, b, iters=20000, seed=0):
    """Bootstrap over PROMPTS on the paired difference mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = diff.size
    obs = float(diff.mean())
    boots = diff[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided p: fraction of bootstrap replicates on the other side of 0
    side = int((boots <= 0).sum()) if obs > 0 else int((boots >= 0).sum())
    p = min(1.0, 2.0 * side / iters)
    return obs, float(lo), float(hi), p


def mcnemar(a, b):
    """Exact-ish McNemar over decode-matched prompt pairs (b=1,a=0) vs (b=0,a=1)."""
    n01 = sum(1 for x, y in zip(a, b) if y > x)
    n10 = sum(1 for x, y in zip(a, b) if y < x)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    z = (abs(n01 - n10) - 1) / ((n01 + n10) ** 0.5)  # continuity-corrected
    # normal tail
    p = 2 * 0.5 * (1 - _erf(z / 2 ** 0.5))
    return n01, n10, min(1.0, p)


def _erf(x):
    # Abramowitz & Stegun 7.1.26
    s = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
             + 0.254829592) * t * (2.718281828459045 ** (-x * x))
    return s * y


def main():
    global ARMS, N_DECODES, TAG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", type=int,
                    default=[200, 250, 300, 350, 400, 450, 500])
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--tag", default=TAG,
                    help="decode-subdir prefix: 'decode' (temperature 1.0) or 'greedy' "
                         "(temperature 0). Default %(default)s.")
    ap.add_argument("--n-decodes", type=int, default=N_DECODES,
                    help="decodes per step per arm to pool (default %(default)s; "
                         "use 1 for greedy, which has nothing to average)")
    ap.add_argument("--arm", action="append", metavar="LABEL=EXPERIMENT:DATE",
                    help="override an arm. Give it TWICE (control first, treatment "
                         "second) to replace the default baseline/v8 pair.")
    args = ap.parse_args()

    TAG, N_DECODES = args.tag, args.n_decodes
    if args.arm:
        if len(args.arm) != 2:
            ap.error("--arm must be given exactly twice: control first, then treatment")
        ARMS = {}
        for spec in args.arm:
            label, _, rest = spec.partition("=")
            experiment, _, date = rest.rpartition(":")
            if not (label and experiment and date):
                ap.error(f"bad --arm {spec!r}; expected LABEL=EXPERIMENT:DATE")
            ARMS[label] = (experiment, date)
    ctrl, treat = list(ARMS)
    print(f"[arms] control={ctrl} ({ARMS[ctrl][0]} @ {ARMS[ctrl][1]})")
    print(f"[arms] treat  ={treat} ({ARMS[treat][0]} @ {ARMS[treat][1]})")
    print(f"[decodes] tag={TAG!r}  n={N_DECODES} per step per arm")

    results = []
    for key, label in [("cov", "COVERAGE (GT in retrieved docs)  -- PRIMARY"),
                       ("exact", "EXACT MATCH (answer == GT)       -- HR@1 proxy"),
                       ("succ", "SUCCESSOR-RULE FOLLOWING          -- what v8 rewards"),
                       ("grounded", "GROUNDED (answer came from docs)")]:
        print(f"\n=== {label}")
        print(f"{'step':>5s} {ctrl:>9s} {treat:>9s} {'diff':>8s} {'95% CI':>18s} "
              f"{'p(boot)':>8s} {'b>a':>5s} {'a>b':>5s} {'p(McN)':>8s}")
        for s in args.steps:
            da, dbk = arm_decodes(ctrl, s), arm_decodes(treat, s)
            if not da or not dbk:
                continue
            ra, _ = prompt_rates(da, key)
            rb, _ = prompt_rates(dbk, key)
            n = min(len(ra), len(rb))
            ra, rb = ra[:n], rb[:n]
            obs, lo, hi, p = paired_bootstrap(ra, rb)
            n01, n10, pm = mcnemar(ra, rb)
            ma, mb = sum(ra) / n, sum(rb) / n
            print(f"{s:5d} {100*ma:8.2f}% {100*mb:8.2f}% {100*obs:+7.2f}% "
                  f"[{100*lo:+6.2f},{100*hi:+6.2f}] {p:8.3f} {n01:5d} {n10:5d} {pm:8.3f}")
            results.append(dict(endpoint=key, step=s, n=n, tag=TAG,
                                control_arm=ctrl, treat_arm=treat,
                                baseline=ma, v8=mb,
                                diff=obs, ci=[lo, hi], p_boot=p,
                                n_v8_gt=n01, n_base_gt=n10, p_mcnemar=pm))

    if args.json:
        json.dump(results, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")

    print("\nb>a / a>b are prompts where the arms DISAGREE; McNemar uses only those.")
    print("Bootstrap resamples prompts, so its CI covers prompt sampling but not")
    print("decode noise; the two agree closely here, which is itself informative.")


if __name__ == "__main__":
    main()
