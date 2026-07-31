# SPDX-License-Identifier: Apache-2.0
"""Offline reward replay -- the audit-before-train gate for v7b (M3 fix).

Re-scores ALREADY-LOGGED rollouts (eval ``test_predictions.json``) with the v7
reward and reports, split by whether the rollout retrieved:

    E[r_total | retrieve]  -  E[r_total | abstain]

under the CURRENT (buggy) length penalty vs the PATCHED one (which excludes the
retriever's ``<tool_response>`` doc text from the word count -- Fix 1 in
``v7/orchestrator.py``). No corpus, no GPU, no re-generation.

Why this exists
---------------
M3 collapse hypothesis (REWARD_REASONING_ANALYSIS.md Part 8): ``len_penalty``
counts the ``<tool_response>`` documents the retriever returns, so every
retrieving rollout is pushed past the 600-word budget and eats ~0.2, while the
retrieval-quality bonus it earns is only ~+0.004. Net: retrieve ~= -0.2 vs
abstain ~= 0 -> GRPO drives retrieval to extinction (retr% 99% -> 5%).

This script confirms that directly and, critically, sizes the fix: if the
PATCHED gap is still < 0, the retqual magnitude (RTHINK_SCALE / _CAP / _W_RETQUAL,
Fix 2) must be raised BEFORE launching v7b -- learned offline, not after a
24h GPU window fails.

Exactness
---------
The ONLY term that differs between buggy and patched is ``len_penalty`` (the
gate/self-repetition terms are tag-based). So we score each rollout ONCE with
the patched reward (the current code) and derive the buggy score analytically:

    r_total = r_out + real + applied - (format_penalty + len_penalty)

    score_buggy = score_patched + len_penalty_patched - len_penalty_buggy_raw

except when LongPAS fired (r_answer >= 0.5), where BOTH len penalties are forced
to 0 and the two scores are identical. This avoids re-encoding the retrieved
docs twice and keeps the retqual/rank computation identical across the two.

Usage (retriever conda env; single-thread to avoid login-node TLS OOM):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /work/11138/pranavbelligundu/vista/anaconda/envs/retriever/bin/python \\
        offline_reward_replay.py \\
        --preds <retrieve-heavy preds.json> [<abstain preds.json> ...] \\
        --data-source amazon --limit 1200
"""

import argparse
import contextlib
import io
import json
import os
import sys
import types

# keep the login node from exhausting thread-local storage on torch ---------- #
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# import the v7 reward as a package member so its relative imports resolve ---- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 5))  # -> .../verl_R1 (has verl/)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The orchestrator uses package-relative imports (`from ... import reward_SPRec`)
# so it must load as `verl.utils.reward_score...`. But `verl/__init__.py` imports
# ray (absent from the retriever env). Stub the two heavy top packages as bare
# namespace packages (real __path__, no __init__ run) so the real reward
# submodules still load from disk. `verl.utils.import_utils` (needed by
# reward_score/__init__) is light and loads normally through the stubbed parent.
for _name in ("verl", "verl.utils"):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = [os.path.join(_REPO, *_name.split("."))]
        sys.modules[_name] = _mod

from verl.utils.reward_score.reward_retrieval.v7 import orchestrator  # noqa: E402

# Silence the orchestrator's 1/64 debug print so the report stays clean.
orchestrator.random = types.SimpleNamespace(randint=lambda a, b: 0)

TOOL_CALL = "<tool_call>"


def _buggy_length_penalty(text: str) -> float:
    """The pre-fix behavior: word-count the WHOLE rollout, tool docs included."""
    if orchestrator._LEN_W <= 0 or orchestrator._LEN_CAP <= 0:
        return 0.0
    n_words = len(text.split())
    excess = max(0, n_words - orchestrator._LEN_SOFT)
    return min(orchestrator._LEN_CAP, orchestrator._LEN_W * excess)


def _solution_str(rec: dict) -> str:
    p = rec.get("predict")
    return "".join(p) if isinstance(p, list) else (p or "")


def _target(rec: dict) -> str:
    o = rec.get("output")
    return "".join(o) if isinstance(o, list) else (o or "")


def replay(paths, data_source, limit):
    # rows: one per rollout, with retrieve flag + patched/buggy score + len terms
    # limit is PER-FILE so pooling a retrieve-heavy and an abstain-heavy file
    # still populates both arms.
    rows = []
    _devnull = io.StringIO()
    for path in paths:
        records = json.load(open(path, encoding="utf-8"))
        n_file = 0
        for rec in records:
            if limit and n_file >= limit:
                break
            n_file += 1
            sol = _solution_str(rec)
            gt = _target(rec)
            if not sol:
                continue
            # the reward code prints per call (reward_SPRec golden/extracted);
            # swallow it so the report stays readable.
            with contextlib.redirect_stdout(_devnull):
                out = orchestrator.compute_score(sol, {"target": gt}, data_source)
            score_patched = out["score"]
            len_patched = out["len_penalty"]          # applied (post-LongPAS)
            longpas = out["r_answer"] >= 0.5           # both len penalties -> 0
            if longpas:
                score_buggy = score_patched
                len_buggy = 0.0
            else:
                len_buggy = _buggy_length_penalty(sol)
                score_buggy = score_patched + len_patched - len_buggy
            rows.append({
                "retrieved": TOOL_CALL in sol,
                "score_patched": score_patched,
                "score_buggy": score_buggy,
                "len_patched": len_patched,
                "len_buggy": len_buggy,
                "retqual": out["retqual"],
                "best_sim": out["best_sim"],
                "shaping": out["r_think"],
                "r_answer": out["r_answer"],
            })
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def report(rows):
    ret = [r for r in rows if r["retrieved"]]
    abst = [r for r in rows if not r["retrieved"]]
    n, nr, na = len(rows), len(ret), len(abst)

    def arm(rs, key):
        return _mean([r[key] for r in rs])

    print("=" * 74)
    print(f"Offline reward replay -- {n} rollouts  "
          f"(retrieve={nr}, abstain={na})")
    print(f"  tau={orchestrator._RETQUAL_TAU}  floor={orchestrator._RETQUAL_FLOOR}  "
          f"W_RETQUAL={orchestrator._W_RETQUAL}  SCALE={orchestrator._SCALE}  "
          f"CAP={orchestrator._CAP}  LEN_SOFT={orchestrator._LEN_SOFT}  "
          f"LEN_CAP={orchestrator._LEN_CAP}")
    print("=" * 74)
    if not nr or not na:
        print("WARNING: need BOTH retrieve and abstain rollouts for a gap. "
              "Pass a retrieve-heavy preds file (e.g. baseline @200) AND an "
              "abstain-heavy one (v7 @200).")

    print(f"{'':22s}{'retrieve':>12s}{'abstain':>12s}{'gap (ret-abs)':>16s}")
    for label, key in [("E[r_total] BUGGY", "score_buggy"),
                       ("E[r_total] PATCHED", "score_patched"),
                       ("mean len_penalty BUGGY", "len_buggy"),
                       ("mean len_penalty PATCHED", "len_patched"),
                       ("mean retqual", "retqual"),
                       ("mean shaping(applied)", "shaping"),
                       ("mean best_sim", "best_sim"),
                       ("mean r_answer", "r_answer")]:
        er, ea = arm(ret, key), arm(abst, key)
        print(f"{label:22s}{er:>12.4f}{ea:>12.4f}{er - ea:>16.4f}")

    print("-" * 74)
    gap_buggy = arm(ret, "score_buggy") - arm(abst, "score_buggy")
    gap_patched = arm(ret, "score_patched") - arm(abst, "score_patched")
    print(f"GATE: E[r|retrieve]-E[r|abstain]   BUGGY={gap_buggy:+.4f}   "
          f"PATCHED={gap_patched:+.4f}")
    verdict = ("PASS -- retrieving now pays; Fix 1 alone suffices."
               if gap_patched >= 0 else
               "FAIL -- retrieving still loses; RAISE retqual magnitude (Fix 2) "
               "before v7b.")
    print(f"VERDICT: {verdict}")
    print("=" * 74)
    return {"n": n, "n_retrieve": nr, "n_abstain": na,
            "gap_buggy": gap_buggy, "gap_patched": gap_patched,
            "pass": gap_patched >= 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="test_predictions.json path(s); pool retrieve- and "
                         "abstain-heavy files for a full gap.")
    ap.add_argument("--data-source", default="amazon")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rollouts")
    ap.add_argument("--json", help="also write the summary to this JSON file")
    args = ap.parse_args()

    rows = replay(args.preds, args.data_source, args.limit)
    summary = report(rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
