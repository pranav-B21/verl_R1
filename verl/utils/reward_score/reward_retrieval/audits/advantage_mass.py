# SPDX-License-Identifier: Apache-2.0
"""G0-A: where does the GRPO gradient MASS actually come from?

Every audit in this project so far has asked about the reward as a *function of a
rollout* -- does the term vary within the group (`selection_gradient_density.py`),
does it correlate with the outcome (`correlation_audit.py`), does its shape resolve
the rank distribution (`reward_shape_replay.py`). None of them asked what the
optimizer does with it afterwards. verl computes the GRPO advantage at
`verl/trainer/ppo/core_algos.py:323`:

    A_i = (R_i - mean_j R_j) / (std_j R_j + eps)

`selection_gradient_density.py` even writes that formula in its own docstring, and
then reasons *only about the numerator*: "a component constant within the group is
subtracted out". True. But the DENOMINATOR is the group's own std, so every
non-degenerate group is rescaled to unit variance regardless of what created the
variance. Concretely:

    outcome-dead group  (all 8 rollouts past GT rank 500, differing only in how much
                         shaping/length they earned; raw spread <= ~0.16)
    outcome-live group  (one rollout hits rank 1; raw spread ~1.0)

    -> after /(std+eps) BOTH groups contribute advantages of order +-1.3.

So a group carrying no accuracy information at all pushes the policy exactly as hard
as a group that does. `reward_shape_replay` measured 72.6% of groups outcome-dead;
`selection_gradient_density` measured the shaping term non-constant in 45-93% of
groups. Nobody multiplied the two.

This also predicts the asymmetry that no single-arm analysis could explain: the
outcome-only BASELINE has no shaping, so its dead groups have std == 0 exactly and
contribute 0/(0+1e-6) = 0 -- they vanish. Only the SHAPED arms convert dead groups
into full-scale gradient. Eight arms, one asymmetry.

Published statements of the same mechanism:
  * Dr. GRPO, "Understanding R1-Zero-Like Training" (arXiv 2503.20783): std
    normalisation is a question-level difficulty bias that OVERWEIGHTS questions with
    low reward std. Outcome-dead groups have the lowest std that exists.
  * "The Dark Room in the Reward Channel" (arXiv 2607.21273): under GRPO, dense
    shaping collapses agents via all-fail groups "where the only variance stems from
    the shaping signal"; removing std normalisation restores baseline parity.

WHAT THIS MEASURES

Replays pooled held-out decodes through the real v7/v8 reward assembly, groups them
by prompt (the repeats of one prompt stand in for the n=8 rollout group -- same
temperature, same policy), and reports:

  1. group census: outcome-dead / outcome-dead-but-total-live / fully dead
  2. advantage mass  sum|A|  split by dead-vs-live, under BOTH normalisations
  3. component attribution: of the centered deviation that produces the gradient,
     how much is outcome vs shaping vs length vs format penalty

GATE (pre-registered before running): if >= 40% of total advantage mass under
std-norm sits in outcome-dead groups, the mechanism is real and v9 proceeds. Below
that, the hypothesis is wrong and no GPU time is spent on it.

  INSTRUMENT CORRECTION (2026-08-10, first run). The pre-registered statistic was
  written as "outcome-dead" = `pstdev(r_outcome) == 0`, borrowed from
  `reward_shape_replay.py` where the reward is a TIERED STEP FUNCTION and exact
  ties are the common case. v6's outcome is *continuous in rank*, so exact equality
  across 9 rollouts essentially never occurs -- ranks 8000 vs 9000 differ by 4.8e-7,
  which the test scores as "live". The first run duly reported 0/983 dead and a 0.0%
  mass share: a null produced entirely by the definition, not by the data. Three
  replacements are reported below, and the pre-registered gate is evaluated against
  the tolerance version so the original threshold still has to be cleared:
    (a) `outcome_negligible` -- pstdev(r_outcome) < OUT_TOL (default 0.01, an order
        of magnitude below the shaping cap): the tolerance form of the original test.
    (b) `shaping_dominated`  -- pstdev(non-outcome components) > pstdev(r_outcome):
        threshold-free, asks directly which side drives THIS group's gradient.
    (c) COMPONENT ATTRIBUTION -- the headline, and threshold-free: of the centered
        deviation that becomes the gradient, what share is outcome vs shaping vs
        length vs format? This needs no notion of "dead" at all.
  This is the sixth instrument in this project to fail in the direction of its
  author's expectation; per the standing rule, the census row is printed next to
  the verdict so a definitional null can never again be read as a result.

CAVEATS, stated up front and inherited from `reward_shape_replay.py`
  * Groups are pooled DECODES of one held-out prompt, not true training groups:
    temperature 1.0 matches training, but these are held-out prompts at
    max_response_length 3072 vs training's 2048, and the policy is frozen. This
    measures the reward assembly's gradient structure on a realistic rollout
    distribution -- it is not a simulation of a training step.
  * `r_cover` is scored with its binary exact-title core rather than the graded
    cosine (which needs the reward's embedding model), exactly as
    `selection_gradient_density.py:186-192` does, so the two instruments agree.
  * The baseline arm's decodes replayed under v8's reward answer the counterfactual
    "what would the shaped reward have done to these rollouts". Pass --preds from
    the v8 arm itself for the as-run measurement; both are reported in the writeup.

Usage (bare login-node python3):

    python3 advantage_mass.py --label baseline@200 \\
        outputs/eval/<arm>/global_step_200/decode*/test_predictions.json \\
        --json advantage_mass_baseline200.json
"""

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from memtype_payoff import CATALOG, extract_last_quoted_answer_span, tiered_match  # noqa: E402


def _load(name, path):
    """Load a sibling module BY PATH -- these audits must run without importing verl."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load("decode_behavior", os.path.join(_HERE, "decode_behavior.py"))
selection = _load("v8_selection", os.path.join(os.path.dirname(_HERE), "v8", "selection.py"))
dsel = _load("decode_selection", os.path.join(_HERE, "decode_selection.py"))

# ---------------------------------------------------------------------------
# Reward constants, mirrored from the shipped launcher (run_in_container_rthink.sh)
# and v7/v8 orchestrator defaults. Kept literal so this runs on a bare python.
# ---------------------------------------------------------------------------
HIT_K, TAIL_W, TAIL_P = 10, 0.10, 4.0            # v6 outcome
FORMAT_PENALTY = 0.5                             # v5 gate
LEN_SOFT, LEN_W, LEN_CAP = 600, 0.0005, 0.2      # v7b turns-aware budget
LEN_PER_TURN, LEN_TURN_CAP = 400, 3
SCALE, CAP = 0.10, 0.08                          # shaping scale/cap
AS_RUN = dict(w_sel=0.5, w_grnd=0.4, w_rep=0.3, w_cov=0.4)   # v8 as it actually ran
DESIGNED = dict(w_sel=0.5, w_grnd=0.1, w_rep=0.3, w_cov=0.4)  # v8 code default

GRPO_EPS = 1e-6                                  # core_algos.py:323 default

TOOL_RESPONSE_SPAN = re.compile(r"<tool_response>.*?</tool_response>", re.S)
DOC_MARKER = re.compile(r"Doc\s+\d+\s+\(Title:")


def ndcg_reward(rank_1based, n_items):
    """v7/orchestrator.py:_ndcg_reward, verbatim."""
    if not rank_1based or not n_items or n_items <= 1:
        return 0.0
    r = min(max(int(rank_1based), 1), int(n_items))
    if r <= HIT_K:
        return 1.0 / math.log2(r + 1)
    base = min(1.0, max(0.0, 1.0 - (math.log(r) / math.log(n_items))))
    return TAIL_W * (base ** TAIL_P)


def length_penalty(text):
    """v7/orchestrator.py:_length_penalty, verbatim (turns-aware, tool docs stripped)."""
    credited = sum(1 for m in TOOL_RESPONSE_SPAN.finditer(text) if DOC_MARKER.search(m.group(0)))
    budget = LEN_SOFT + LEN_PER_TURN * min(credited, LEN_TURN_CAP)
    n_words = len(TOOL_RESPONSE_SPAN.sub(" ", text).split())
    return min(LEN_CAP, LEN_W * max(0, n_words - budget))


def shaping(is_succ, is_grnd, is_rep, r_cover, w):
    """v8/orchestrator.py:239-241, verbatim."""
    raw = w["w_sel"] * is_succ + w["w_grnd"] * is_grnd - w["w_rep"] * is_rep + w["w_cov"] * r_cover
    return max(-CAP, min(CAP, SCALE * raw))


def assemble(rank1, well_formed, r_answer, text, state, w):
    """The v7/v8 reward assembly, including the LongPAS asymmetry.

    Returns the four components separately so the gradient can be attributed.
    """
    r_out = ndcg_reward(rank1, N_ITEMS)
    fmt_pen = 0.0
    if not well_formed:
        r_out = 0.0
        fmt_pen = FORMAT_PENALTY
    len_pen = length_penalty(text)
    applied = shaping(state["is_succ"], state["is_grnd"], state["is_rep"],
                      float(state["covered"]), w)
    # LongPAS: a genuinely correct AND well-formed answer is never punished.
    if r_answer >= 0.5:
        applied = max(0.0, applied)
        fmt_pen = 0.0
        len_pen = 0.0
    return dict(out=r_out, shape=applied, len_pen=-len_pen, fmt_pen=-fmt_pen)


def pstd(vals):
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", help="pooled test_predictions.json for ONE checkpoint")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--weights", choices=("as_run", "designed"), default="as_run")
    ap.add_argument("--out-tol", type=float, default=0.01,
                    help="pstdev(r_outcome) below this counts as outcome-negligible "
                         "(default 0.01 = an order of magnitude below the shaping cap)")
    ap.add_argument("--json")
    args = ap.parse_args()
    w = AS_RUN if args.weights == "as_run" else DESIGNED

    # ---------------- load + parse every rollout ----------------
    rollouts = []
    for p in args.preds:
        for rec in json.load(open(p, encoding="utf-8")):
            text = "".join(rec.get("predict") or [])
            title = extract_last_quoted_answer_span(text)
            n_open, n_close = text.count("<answer>"), text.count("</answer>")
            rollouts.append({
                "pid": rec["input"],
                "gt": rec.get("output", "").strip().strip('"'),
                "title": title or "NAN",
                "has_title": bool(title),
                "well_formed": (n_open == 1 and n_close == 1),
                "n_open": n_open, "n_close": n_close,
                "text": text,
                "state": _state(rec),
            })

    # ---------------- rank the GT for every distinct title (eval.py numerics) ------
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
    embeddings = torch.tensor(torch.load(os.path.join(CATALOG, "embeddings.pt")), device="cpu")
    name2id = json.load(open(os.path.join(CATALOG, "name2id.json"), encoding="utf-8"))
    global N_ITEMS
    N_ITEMS = embeddings.shape[0]
    uniq = sorted({r["title"] for r in rollouts})
    vecs = torch.tensor(model.encode(uniq, batch_size=256, show_progress_bar=False))
    ranks = torch.cdist(vecs, embeddings, p=2).argsort(dim=-1).argsort(dim=-1)
    idx = {t: i for i, t in enumerate(uniq)}

    groups, n_dropped = {}, 0
    for r in rollouts:
        if r["gt"] not in name2id:
            n_dropped += 1
            continue
        rank1 = int(ranks[idx[r["title"]]][name2id[r["gt"]]].item()) + 1
        # baseline arm reward: reward_SPRec tiered match, multi-answer rule included
        r_answer = tiered_match(rank1) if r["has_title"] else 0.0
        base = r_answer
        if r["has_title"] and (r["n_open"] > 1 or r["n_close"] > 1):
            base = base / 4 if base else -0.5
        comp = assemble(rank1, r["well_formed"], r_answer, r["text"], r["state"], w)
        comp["base"] = base
        comp["rank"] = rank1
        comp["total"] = comp["out"] + comp["shape"] + comp["len_pen"] + comp["fmt_pen"]
        groups.setdefault(r["pid"], []).append(comp)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    # ---------------- census + advantage mass ----------------
    ng = len(groups)
    census = dict(outcome_exact_dead=0, outcome_negligible=0, shaping_dominated=0,
                  fully_dead=0, base_dead=0)
    mass = dict(std_dead=0.0, std_live=0.0, nostd_dead=0.0, nostd_live=0.0,
                base_std=0.0, base_nostd=0.0)
    l1 = dict(out=0.0, shape=0.0, len_pen=0.0, fmt_pen=0.0)
    out_spreads = []

    for rows in groups.values():
        outs = [x["out"] for x in rows]
        tots = [x["total"] for x in rows]
        bases = [x["base"] for x in rows]
        # non-outcome part of the reward, as one series
        others = [x["shape"] + x["len_pen"] + x["fmt_pen"] for x in rows]

        s_out = pstd(outs)
        out_spreads.append(s_out)
        # (a) tolerance form of the pre-registered test; (b) threshold-free dominance
        out_dead = s_out < args.out_tol
        census["outcome_exact_dead"] += s_out == 0.0
        census["outcome_negligible"] += out_dead
        census["shaping_dominated"] += pstd(others) > s_out
        census["fully_dead"] += pstd(tots) == 0.0
        census["base_dead"] += pstd(bases) == 0.0

        m_t, s_t = statistics.mean(tots), pstd(tots)
        for x in rows:
            a_std = abs((x["total"] - m_t) / (s_t + GRPO_EPS))
            a_nostd = abs(x["total"] - m_t)
            mass["std_dead" if out_dead else "std_live"] += a_std
            mass["nostd_dead" if out_dead else "nostd_live"] += a_nostd

        # component attribution: A_i = (oc + sc + lc + fc) / (s + eps)
        for k in ("out", "shape", "len_pen", "fmt_pen"):
            mk = statistics.mean(x[k] for x in rows)
            l1[k] += sum(abs(x[k] - mk) for x in rows) / (s_t + GRPO_EPS)

        m_b, s_b = statistics.mean(bases), pstd(bases)
        for x in rows:
            mass["base_std"] += abs((x["base"] - m_b) / (s_b + GRPO_EPS))
            mass["base_nostd"] += abs(x["base"] - m_b)

    tot_std = mass["std_dead"] + mass["std_live"]
    tot_nostd = mass["nostd_dead"] + mass["nostd_live"]
    l1_tot = sum(l1.values())

    # ---------------- report ----------------
    p = print
    p(f"\n{'=' * 78}\nG0-A  advantage mass   [{args.label}]  weights={args.weights}\n{'=' * 78}")
    p(f"groups={ng}  rollouts={sum(len(v) for v in groups.values())}  "
      f"dropped (GT not in name2id)={n_dropped}  catalog N={N_ITEMS}")

    qs = statistics.quantiles(out_spreads, n=100) if len(out_spreads) > 2 else [0, 0, 0]
    p("\n1. GROUP CENSUS")
    p(f"  outcome EXACTLY constant (pre-registered test)       "
      f"{census['outcome_exact_dead']:5d}/{ng}  ({census['outcome_exact_dead'] / ng:6.1%})"
      f"   <- mis-specified: r_outcome is continuous")
    p(f"  within-group spread of r_outcome: median {statistics.median(out_spreads):.2e}   "
      f"p90 {qs[89]:.2e}   p99 {qs[98]:.2e}")
    p(f"  outcome NEGLIGIBLE (spread < {args.out_tol})                 "
      f"{census['outcome_negligible']:5d}/{ng}  ({census['outcome_negligible'] / ng:6.1%})"
      f"   <- the gated statistic")
    p(f"  shaping/penalties DOMINATE outcome (threshold-free) "
      f"{census['shaping_dominated']:5d}/{ng}  ({census['shaping_dominated'] / ng:6.1%})")
    p(f"  fully dead (no gradient at all)                      "
      f"{census['fully_dead']:5d}/{ng}  ({census['fully_dead'] / ng:6.1%})")
    p(f"  [baseline arm, no shaping] dead groups               "
      f"{census['base_dead']:5d}/{ng}  ({census['base_dead'] / ng:6.1%})   <- these contribute 0")

    p(f"\n2. ADVANTAGE MASS  sum|A|,  shaped arm   (dead = outcome spread < {args.out_tol})")
    p(f"  {'':22s} {'in outcome-dead':>16s} {'in live':>12s} {'dead share':>12s}")
    p(f"  {'WITH std-norm (as run)':22s} {mass['std_dead']:16.1f} {mass['std_live']:12.1f} "
      f"{mass['std_dead'] / tot_std:11.1%}")
    p(f"  {'WITHOUT std-norm':22s} {mass['nostd_dead']:16.3f} {mass['nostd_live']:12.3f} "
      f"{mass['nostd_dead'] / tot_nostd:11.1%}")
    p(f"\n  baseline arm (outcome only): sum|A| std-norm = {mass['base_std']:.1f}   "
      f"no-std = {mass['base_nostd']:.3f}")

    p("\n3. COMPONENT ATTRIBUTION  (L1 of each centered component / (group std + eps))")
    p("   Threshold-free, and the headline: of the deviation that BECOMES the gradient,")
    p("   how much is the thing we are trying to optimise?")
    for k, name in (("out", "outcome (GT rank)"), ("shape", "shaping (v8 r_select)"),
                    ("len_pen", "length penalty"), ("fmt_pen", "format penalty")):
        p(f"  {name:24s} {l1[k]:10.1f}   ({l1[k] / l1_tot:6.1%})")
    p(f"  {'-> non-outcome total':24s} {l1_tot - l1['out']:10.1f}   "
      f"({(l1_tot - l1['out']) / l1_tot:6.1%})")

    dead_share = mass["std_dead"] / tot_std
    out_share = l1["out"] / l1_tot
    p(f"\n{'-' * 78}")
    p(f"GATE (pre-registered, tolerance form): dead-group share of advantage mass "
      f"under std-norm = {dead_share:.1%}   (threshold >= 40%)")
    p(f"VERDICT: {'PASS -- mechanism confirmed, v9 proceeds' if dead_share >= 0.40 else 'FAIL -- mechanism refuted, do not spend GPU on v9'}")
    p(f"Removing std-norm moves that share to {mass['nostd_dead'] / tot_nostd:.1%}.")
    p(f"Corroborating, threshold-free: the outcome supplies {out_share:.1%} of the "
      f"gradient-producing deviation.")

    res = dict(label=args.label, weights=args.weights, files=args.preds, n_groups=ng,
               catalog_n=N_ITEMS, out_tol=args.out_tol, census=census,
               out_spread_median=statistics.median(out_spreads),
               out_spread_p90=qs[89], out_spread_p99=qs[98],
               mass={k: round(v, 4) for k, v in mass.items()},
               dead_share_std=round(dead_share, 5),
               dead_share_nostd=round(mass["nostd_dead"] / tot_nostd, 5),
               component_l1={k: round(v, 4) for k, v in l1.items()},
               component_share={k: round(v / l1_tot, 5) for k, v in l1.items()},
               gate_pass=bool(dead_share >= 0.40))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        p(f"\nwrote {args.json}")


def _state(rec):
    """(is_succ, is_grnd, is_rep, covered) for ONE rollout.

    Mirrors selection_gradient_density._rollout_state so the two instruments cannot
    drift; `covered` is r_cover's binary exact-title core.
    """
    text = "".join(rec.get("predict") or [])
    spans = db._retrieved_spans(text)
    seqs = selection.doc_sequences(spans)
    items = {i for s in seqs for i in s}
    hist = selection.user_history(rec.get("input", ""))
    succ = selection.successor_set(seqs, hist)
    gt = selection.norm_title(rec.get("output", ""))
    pred = dsel._pred_title(text)
    return dict(
        is_succ=int(bool(pred) and pred in succ),
        is_grnd=int(bool(pred) and pred in items),
        is_rep=int(bool(pred) and pred in hist),
        covered=int(bool(gt) and gt in items),
    )


if __name__ == "__main__":
    main()
