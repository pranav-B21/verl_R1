# SPDX-License-Identifier: Apache-2.0
"""Decode-time SELECTION funnel — the Gate-1 instrument for v8.

`decode_behavior.py` measures the retrieval half of the funnel (retrieval rate,
docs, coverage). It stops exactly where v8's hypothesis starts. The v8 decision
rule (../v8/README.md, Gate 1) needs the *next* stage:

    HR@1  =  coverage  x  P(pick GT | GT in retrieved docs)
             ~3%          ~chance  (baseline 0/57, v7b 2/53 at step 200)

and, because v8 credits a structural rule rather than the GT itself, the
successor-set diagnostics that say whether that rule can even reach the GT:

    P(GT in successor set | GT retrieved)   -- the pre-registered 13% ceiling
    P(answer in successor set)              -- did the policy learn the rule
    |successor set| vs |retrieved items|    -- how much the rule narrows the pick

Everything is computed from saved decodes only: no corpus, no index, no GPU.
The successor relation is parsed by ../v8/selection.py -- the SAME code the
reward uses, so a gap between the training curve (`val-aux/*/is_successor`) and
this report is a real train/decode gap, not two different definitions.

Usage:
    python3 decode_selection.py outputs/eval/<arm>/global_step_200/*/test_predictions.json
    python3 decode_selection.py --json out.json <preds>...

Read the pooled row: per-decode selection counts are ~1-3 events out of ~50
winnable cases, which is one binomial draw wide. The pooled column across
EVAL_REPEATS decodes is the only number with enough events to compare arms.
"""

import argparse
import importlib.util
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    """Load a sibling module BY PATH, not through `verl.…`.

    Importing the package pulls in verl/__init__.py, which needs `packaging`
    and the rest of the training stack; the whole point of the audits is that
    they run on a bare login-node python with no verl install (same convention
    as decode_behavior.py / audit_a_selection.py, which are standalone).
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load("decode_behavior", os.path.join(_HERE, "decode_behavior.py"))
selection = _load("v8_selection", os.path.join(os.path.dirname(_HERE), "v8", "selection.py"))

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def _pred_title(text):
    """The emitted item title, parsed exactly as v8/orchestrator.py does.

    Answers are "Artist - Album" or a quoted title; the orchestrator takes the
    first quoted segment when one exists. Matching that here is load-bearing:
    a looser parse would report a selection rate the reward never saw.
    """
    answers = ANSWER.findall(text)
    if not answers:
        return ""
    raw = answers[-1].strip()
    m = re.search(r'"([^"]*)', raw)
    return selection.norm_title(m.group(1) if m else raw)


def analyze(path):
    records = json.load(open(path, encoding="utf-8"))
    n = len(records)
    if not n:
        raise SystemExit(f"{path}: no records")

    c = dict(
        n=n, n_gt=0, n_ret=0, n_answered=0,
        n_cov=0,            # GT present in the retrieved items
        n_cov_ans=0,        # ...and the model emitted an answer
        n_cov_grnd=0,       # ...and that answer came from the docs (README denom)
        n_pick_gt=0,        # ...and that answer IS the GT   <- Gate 1
        n_cov_succ=0,       # GT reachable by the successor rule  <- 13% ceiling
        n_grounded=0, n_successor=0, n_repeat=0,
        n_has_succ=0,
    )
    sum_items = sum_succ = 0

    for rec in records:
        text = "".join(rec.get("predict") or [])
        spans = db._retrieved_spans(text)
        seqs = selection.doc_sequences(spans)
        items = {i for s in seqs for i in s}
        hist = selection.user_history(rec.get("input", ""))
        succ = selection.successor_set(seqs, hist)
        gt = selection.norm_title(rec.get("output", ""))
        pred = _pred_title(text)

        if spans:
            c["n_ret"] += 1
        sum_items += len(items)
        sum_succ += len(succ)
        if succ:
            c["n_has_succ"] += 1
        if pred:
            c["n_answered"] += 1
            if pred in items:
                c["n_grounded"] += 1
            if pred in succ:
                c["n_successor"] += 1
            if pred in hist:
                c["n_repeat"] += 1
        if gt:
            c["n_gt"] += 1
            if gt in items:
                c["n_cov"] += 1
                if gt in succ:
                    c["n_cov_succ"] += 1
                if pred:
                    c["n_cov_ans"] += 1
                    if pred in items:
                        c["n_cov_grnd"] += 1
                    if pred == gt:
                        c["n_pick_gt"] += 1

    c["file"] = path
    c["items_mean"] = sum_items / n
    c["succ_mean"] = sum_succ / n
    return c


def _rates(c):
    d = lambda a, b: (a / b) if b else 0.0  # noqa: E731
    return dict(
        coverage=d(c["n_cov"], c["n_gt"]),
        pick_gt=d(c["n_pick_gt"], c["n_cov_ans"]),
        # same numerator over the grounded denominator -- the form the v8 README
        # quotes (baseline 0/57, v7b 2/53), kept so the two are comparable
        pick_gt_grnd=d(c["n_pick_gt"], c["n_cov_grnd"]),
        gt_in_succ=d(c["n_cov_succ"], c["n_cov"]),
        successor=d(c["n_successor"], c["n_answered"]),
        grounded=d(c["n_grounded"], c["n_answered"]),
        repeat=d(c["n_repeat"], c["n_answered"]),
        has_succ=d(c["n_has_succ"], c["n"]),
        # chance selection = 1 / candidates, the bar P(pick GT) must clear
        chance=d(1.0, c["items_mean"]) if c["items_mean"] else 0.0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", help="test_predictions.json path(s)")
    ap.add_argument("--json", help="also write the rows to this JSON file")
    args = ap.parse_args()

    rows = []
    for p in args.preds:
        try:
            rows.append(analyze(p))
        except Exception as exc:  # a bad decode must not lose the good ones
            print(f"[skip] {p}: {exc}")
    if not rows:
        raise SystemExit("no readable prediction files")

    pooled = {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith("n")}
    pooled["items_mean"] = sum(r["items_mean"] * r["n"] for r in rows) / pooled["n"]
    pooled["succ_mean"] = sum(r["succ_mean"] * r["n"] for r in rows) / pooled["n"]
    pooled["file"] = f"POOLED ({len(rows)} decodes)"

    hdr = (f"{'decode':34s} {'cov%':>6s} {'GTinDocs':>9s} "
           f"{'pickGT|cov':>12s} {'chance':>7s} {'GTinSucc%':>10s} "
           f"{'succ%':>7s} {'grnd%':>7s} {'rep%':>6s} {'|succ|':>7s} {'|items|':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows + [pooled]:
        q = _rates(r)
        name = os.path.basename(os.path.dirname(r["file"])) if r["file"].endswith(".json") else r["file"]
        if r["file"].startswith("POOLED"):
            name = r["file"]
            print("-" * len(hdr))
        print(f"{name[:34]:34s} {100 * q['coverage']:6.2f} "
              f"{r['n_cov']:>4d}/{r['n_gt']:<4d} "
              f"{r['n_pick_gt']:>4d}/{r['n_cov_ans']:<4d}{'':>3s} "
              f"{100 * q['chance']:6.1f}% {100 * q['gt_in_succ']:9.1f} "
              f"{100 * q['successor']:6.1f} {100 * q['grounded']:6.1f} "
              f"{100 * q['repeat']:5.1f} {r['succ_mean']:7.2f} {r['items_mean']:8.2f}")

    q = _rates(pooled)
    print()
    print("GATE 1 (selection branch): pick GT | GT in docs = "
          f"{100 * q['pick_gt']:.1f}%  ({pooled['n_pick_gt']}/{pooled['n_cov_ans']}) "
          f"— threshold 15%, chance {100 * q['chance']:.1f}%")
    print("  over the grounded-answer denominator (the v8 README's form): "
          f"{100 * q['pick_gt_grnd']:.1f}%  ({pooled['n_pick_gt']}/{pooled['n_cov_grnd']})")
    print("Reference @ step 200, pooled: baseline-n8 0/57, v7b 2/53 — both at chance.")
    print("If pickGT|cov is at chance, an HR gain is NOT attributable to selection,")
    print("and coverage x chance still caps HR@1 near 0.001-0.002.")
    print()
    print("GTinSucc% is the pre-registered ceiling for r_select (~13% measured on")
    print("the v7b decodes). succ% is how often the policy actually follows the rule.")
    print("succ% high with pickGT|cov at chance = the rule was learned but does not")
    print("carry the GT: the cue is exhausted, not mis-weighted.")

    if args.json:
        json.dump({"rows": rows, "pooled": pooled}, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
