"""
test_reward_system.py — a *visual* simulator of the REC-R1 reasoning reward.

Goal
----
Reproduce, on a laptop / login node (NO retriever, NO training run, NO GPU
required), exactly the math that runs on the GPU nodes for every rollout — so you
can SEE what the reward system is doing and judge whether it actually helps the
policy learn.

It does three things:

  1. MATH ENGINE  — replays 10 real reasoning rollouts taken from the
     `[Rthink-v3]` debug lines in
       nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v3.log
     through the *live* v3 orchestrator equations (shaping → cap → LongPAS
     asymmetry) and verifies the simulator reproduces the logged shaping/R_total
     to the 3rd decimal. This proves the simulator == the backend.

  2. GRPO SIMULATOR — groups the rollouts the way GRPO does (a group = several
     rollouts of the same prompt) and computes the group-normalised advantage
     `A_i = (R_i - mean) / std` for BOTH the answer-only baseline and v3.
     This is the part that answers "is there actually improvement?": under GRPO
     only *within-group variance* survives, so the test shows whether v3 creates
     a useful, correctness-aligned gradient in the majority all-wrong groups
     where the baseline has none.

  3. RUBRIC VERDICT — scores the run against the §3 success rubric in
     REWARD_REASONING_ANALYSIS.md (cap invariant, no tier-reorder, two-sided
     shaping, corr(r_think, r_answer) > 0, live gradient in dead groups).

Optional 4th layer (`--recompute`, needs torch + sentence-transformers, i.e. run
inside the singularity container): re-derives the four process components
{tool_use, grounding, synthesis, self_rep} from the raw solution strings using
the REAL v3 `reasoning.py`, to smoke-test the text→components pipeline end to end.

Usage  (lives in verl/utils/reward_score/reward_reasoning/; CWD-independent)
-----
    # pure math + GRPO + verdict (no heavy deps, runs anywhere):
    python verl/utils/reward_score/reward_reasoning/test_reward_system.py

    # also re-run the real component extractor on the solution strings
    # (inside the container):
    python verl/utils/reward_score/reward_reasoning/test_reward_system.py --recompute

    # try a different config / future iteration without editing code:
    RTHINK_SCALE=0.15 RTHINK_CAP=0.1 \
        python verl/utils/reward_score/reward_reasoning/test_reward_system.py
    RTHINK_W_GROUND=0.6 \
        python verl/utils/reward_score/reward_reasoning/test_reward_system.py

The math always tracks the LIVE orchestrator constants (env-overridable), so a
v4 retune is reflected automatically.
"""

import os
import sys
import math
import argparse

# Make `verl...` importable regardless of CWD. This file lives at
#   verl_R1/verl/utils/reward_score/reward_reasoning/test_reward_system.py
# so the import root (verl_R1/, the dir that contains the `verl` package) is a
# few directories up. Walk up to find it so the path stays correct even if the
# tree is relocated or the test is run from another CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_root = _THIS_DIR
for _ in range(6):
    _root = os.path.dirname(_root)
    if os.path.isdir(os.path.join(_root, "verl", "utils", "reward_score")):
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break


# --------------------------------------------------------------------------- #
# Colour helpers (degrade gracefully when piped / NO_COLOR set).
# --------------------------------------------------------------------------- #
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(s, code):
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else str(s)


def red(s):    return _c(s, "31")
def green(s):  return _c(s, "32")
def yellow(s): return _c(s, "33")
def blue(s):   return _c(s, "34")
def cyan(s):   return _c(s, "36")
def grey(s):   return _c(s, "90")
def bold(s):   return _c(s, "1")


def signed_bar(value, lo, hi, width=21):
    """A centred-at-zero bar for values in [lo, hi] (lo<0<hi)."""
    mid = width // 2
    cells = [" "] * width
    cells[mid] = "│"
    if hi == lo:
        return "".join(cells)
    if value >= 0:
        n = round(value / hi * mid) if hi else 0
        for i in range(mid + 1, min(width, mid + 1 + n)):
            cells[i] = "█"
        s = "".join(cells)
        return green(s) if _USE_COLOR else s
    else:
        n = round(-value / -lo * mid) if lo else 0
        for i in range(max(0, mid - n), mid):
            cells[i] = "█"
        s = "".join(cells)
        return red(s) if _USE_COLOR else s


# --------------------------------------------------------------------------- #
# Live v3 orchestrator constants (env-overridable). Falls back to documented
# defaults if the package can't be imported (e.g. path problems).
# --------------------------------------------------------------------------- #
def _load_constants():
    try:
        # orchestrator.py only imports os/random at module load (reward_SPRec
        # and torch are imported lazily inside compute_score), so this is safe
        # without torch installed.
        from verl.utils.reward_score.reward_reasoning.v3 import orchestrator as o
        return dict(SCALE=o._SCALE, CAP=o._CAP, W_TOOL=o._W_TOOL,
                    W_GROUND=o._W_GROUND, W_SYNTH=o._W_SYNTH, W_REP=o._W_REP,
                    source="live orchestrator.py")
    except Exception as e:  # pragma: no cover - defensive
        def envf(n, d):
            try: return float(os.environ.get(n, d))
            except (TypeError, ValueError): return d
        return dict(SCALE=envf("RTHINK_SCALE", 0.10), CAP=envf("RTHINK_CAP", 0.08),
                    W_TOOL=envf("RTHINK_W_TOOL", 0.30), W_GROUND=envf("RTHINK_W_GROUND", 0.40),
                    W_SYNTH=envf("RTHINK_W_SYNTH", 0.30), W_REP=envf("RTHINK_W_REP", 0.50),
                    source=f"defaults (import failed: {e})")


K = _load_constants()

# R_answer partial-credit tiers (reward_SPRec.py) are 1.0/0.8/0.5/0.1/0.001/0;
# the smallest nonzero gap is 0.1 — the cap must stay below it (see invariants).
_SMALLEST_TIER_GAP = 0.1


def score_components(tool_use, grounding, synthesis, self_rep, r_answer):
    """The EXACT v3 orchestrator math (orchestrator.py:97-112), as a pure fn.

    Returns (r_think_raw, shaping_applied, r_total).
    """
    raw = (K["W_TOOL"] * tool_use
           + K["W_GROUND"] * grounding
           + K["W_SYNTH"] * synthesis
           - K["W_REP"] * self_rep)
    shaping = max(-K["CAP"], min(K["CAP"], K["SCALE"] * raw))
    # LongPAS asymmetry: never penalise an already-correct answer.
    applied = max(0.0, shaping) if r_answer >= 0.5 else shaping
    return raw, applied, r_answer + applied


# --------------------------------------------------------------------------- #
# The 10 reasoning runs.
#
# Each is a REAL rollout: tool/grnd/syn/rep and r_answer are the values printed
# by the live reward on the GPU node (copied verbatim from `[Rthink-v3]` debug
# lines in the v3 log), so the math engine replays genuine backend output. The
# `solution` field is an illustrative reconstruction of that archetype's text,
# used only by the optional --recompute pipeline smoke-test.
#
# `group` ties rollouts into GRPO groups (same prompt → one group), chosen to
# exercise the three regimes that matter: an all-wrong group (the majority),
# a group containing a correct answer, and a penalty/degenerate group.
# --------------------------------------------------------------------------- #
def _sol(think_pre, search, info, think_post, answer, multi_answer=False):
    """Assemble a multi-turn solution string in the model's output format."""
    parts = [f"<think>\n{think_pre}\n</think>"]
    if search:
        parts.append(f"<search>{search}</search>")
    if info:
        parts.append(f"<info>{info}</info>")
    if think_post:
        parts.append(f"<think>\n{think_post}\n</think>")
    if answer:
        parts.append(f'<answer>"{answer}"</answer>')
        if multi_answer:
            parts.append(f'<answer>"{answer}"</answer>')
    return "\n".join(parts)


RUNS = [
    # ---------------- GROUP A — all-wrong (the typical majority) ------------- #
    dict(id="A1", group="A-allwrong", desc="full agentic, strongly grounded answer",
         tool=1.000, grnd=0.855, syn=0.338, rep=0.216, r_answer=0.000,
         logged_shaping=+0.0635, logged_total=0.064,
         solution=_sol(
             "The user recently played classic-rock studio albums from 1969-1983. "
             "I should look at what listeners of that era gravitate to next.",
             "what albums do fans of 1970s classic rock listen to next?",
             "Top related: 'The Essential Duke Ellington', 'Kind of Blue', "
             "'Bitches Brew', 'A Love Supreme' — frequently co-listened jazz-rock crossovers.",
             "Given the retrieved co-listening data, a jazz-rock crossover fits their era "
             "preference. The Essential Duke Ellington appears most central in the results.",
             "The Essential Duke Ellington")),
    dict(id="A2", group="A-allwrong", desc="full agentic but weak grounding",
         tool=1.000, grnd=0.159, syn=0.391, rep=0.172, r_answer=0.000,
         logged_shaping=+0.0395, logged_total=0.039,
         solution=_sol(
             "The user likes a mix of rock and pop. Let me search for similar artists.",
             "music similar to 1980s pop rock",
             "Results: 'Private Eyes', 'Loud Hailer', 'On Stage Remastered'.",
             "These results suggest mainstream pop rock. I'll pick something energetic and "
             "contemporary rather than echoing the list, leaning to a newer release.",
             "Revenge")),
    dict(id="A3", group="A-allwrong", desc="searched, got results, no real synthesis",
         tool=0.600, grnd=0.000, syn=0.000, rep=0.199, r_answer=0.000,
         logged_shaping=+0.0080, logged_total=0.008,
         solution=_sol(
             "I need to recommend music. Let me search.",
             "recommend music for this user",
             "Results returned: several albums.",
             "",  # no post-retrieval thinking -> tool_use loses the 0.4 term
             "Some Album")),
    dict(id="A4", group="A-allwrong", desc="lazy: no search at all",
         tool=0.000, grnd=0.262, syn=0.000, rep=0.000, r_answer=0.000,
         logged_shaping=+0.0105, logged_total=0.010,
         solution=_sol(
             "The user listened to rock. I'll just guess a popular rock album without "
             "looking anything up.",
             "", "", "",
             "Greatest Hits")),

    # ---------------- GROUP B — contains a correct answer ------------------- #
    dict(id="B1", group="B-haswinner", desc="CORRECT answer, decent reasoning (asymmetric: +only)",
         tool=1.000, grnd=0.255, syn=0.374, rep=0.189, r_answer=1.000,
         logged_shaping=+0.0420, logged_total=1.042,
         solution=_sol(
             "User's history centres on jazz and big band. Let me retrieve co-listens.",
             "what do big-band jazz listeners enjoy?",
             "Top: 'The Essential Duke Ellington', 'Sing Sing Sing', 'Mood Indigo'.",
             "The retrieved results strongly point to Ellington's essentials, which matches "
             "the user's big-band leaning. That is the best-supported pick.",
             "The Essential Duke Ellington")),
    dict(id="B2", group="B-haswinner", desc="WRONG but maximally grounded (cap must keep it < B1)",
         tool=1.000, grnd=0.957, syn=0.394, rep=0.367, r_answer=0.000,
         logged_shaping=+0.0617, logged_total=0.062,
         solution=_sol(
             "User likes jazz. Retrieve neighbours.",
             "jazz albums similar to user history",
             "Top: 'Kind of Blue', 'Blue Train', 'Giant Steps'.",
             "Kind of Blue is the most central jazz album in the retrieved neighbourhood, "
             "so it is the most evidence-supported recommendation here.",
             "Kind of Blue")),
    dict(id="B3", group="B-haswinner", desc="degenerate / near-empty output",
         tool=0.000, grnd=0.000, syn=0.000, rep=0.000, r_answer=0.000,
         logged_shaping=+0.0000, logged_total=0.000,
         solution="<think>\n.\n</think>\n<answer>\"\"</answer>"),

    # ---------------- GROUP C — penalties / two-sided shaping --------------- #
    dict(id="C1", group="C-penalty", desc="multi-<answer> penalty (-0.5), but grounded",
         tool=0.300, grnd=0.711, syn=0.000, rep=0.000, r_answer=-0.500,
         logged_shaping=+0.0375, logged_total=-0.463,
         solution=_sol(
             "User likes folk rock. Search.",
             "folk rock recommendations",
             "Results: 'Blue', 'Harvest', 'Tapestry'.",
             "",
             "Harvest", multi_answer=True)),
    dict(id="C2", group="C-penalty", desc="NEGATIVE shaping: self-rep penalty beats positives",
         tool=0.700, grnd=0.000, syn=0.205, rep=0.600, r_answer=0.000,
         logged_shaping=-0.0028, logged_total=-0.003,
         solution=_sol(
             "I need to recommend music. I need to recommend music. I need to recommend "
             "music for this user. I need to recommend music for this user now.",
             "recommend music",
             "Results returned.",
             "Let me think. Let me think. Let me think about this carefully now.",
             "Album X")),
    dict(id="C3", group="C-penalty", desc="top-500 partial hit (0.001), good reasoning",
         tool=1.000, grnd=0.302, syn=0.430, rep=0.222, r_answer=0.001,
         logged_shaping=+0.0439, logged_total=0.045,
         solution=_sol(
             "User enjoys soul and R&B. Let me find what fits.",
             "soul and r&b album recommendations",
             "Top: 'What's Going On', 'Songs in the Key of Life', 'Innervisions'.",
             "Based on the retrieved soul classics, Innervisions aligns closely with the "
             "user's stated taste and the co-listening signal, so I'll go with it.",
             "Innervisions")),
]


# --------------------------------------------------------------------------- #
# 1 + 2. Math engine + per-run table.
# --------------------------------------------------------------------------- #
def run_math_engine():
    print(bold("\n" + "=" * 92))
    print(bold(" 1. MATH ENGINE — replay 10 real v3 rollouts through the live orchestrator equations"))
    print("=" * 92)
    print(grey(f"   constants ({K['source']}):  "
               f"SCALE={K['SCALE']}  CAP={K['CAP']}  "
               f"w_tool={K['W_TOOL']}  w_ground={K['W_GROUND']}  "
               f"w_synth={K['W_SYNTH']}  w_rep={K['W_REP']}"))
    print(grey("   R_total = R_answer + shaping;  shaping = clip(SCALE*raw, ±CAP);  "
               "raw = w·[tool,grnd,syn] − w_rep·self_rep"))
    print(grey("   LongPAS: if R_answer ≥ 0.5 only the POSITIVE part of shaping is applied.\n"))

    hdr = (f"   {'id':>3} {'tool':>5} {'grnd':>5} {'syn':>5} {'rep':>5} │ "
           f"{'raw':>6} {'shaping':>8} │ {'R_ans':>6} {'R_tot':>6} │ {'vs log':>7}")
    print(bold(hdr))
    print("   " + "─" * (len(hdr) - 3))

    all_ok = True
    rows = []
    for r in RUNS:
        raw, shaping, total = score_components(
            r["tool"], r["grnd"], r["syn"], r["rep"], r["r_answer"])
        rows.append(dict(r, raw=raw, shaping=shaping, total=total))

        ok = (abs(shaping - r["logged_shaping"]) < 1e-3
              and abs(total - r["logged_total"]) < 1e-3)
        all_ok &= ok
        tag = green("MATCH") if ok else red(f"DIFF!")

        sh = f"{shaping:+.4f}"
        sh = green(sh) if shaping > 0 else (red(sh) if shaping < 0 else grey(sh))
        print(f"   {r['id']:>3} {r['tool']:>5.2f} {r['grnd']:>5.2f} {r['syn']:>5.2f} "
              f"{r['rep']:>5.2f} │ {raw:>6.3f} {sh:>8} │ {r['r_answer']:>+6.3f} "
              f"{total:>+6.3f} │ {tag:>7}")
        print(grey(f"        └─ {r['desc']}"))

    print("\n   " + (green("✓ simulator reproduces every logged shaping/R_total (≤1e-3)")
                     if all_ok else red("✗ simulator DIVERGED from the logged values")))
    return rows


# --------------------------------------------------------------------------- #
# 2. GRPO simulator — the heart of "is there improvement?".
# --------------------------------------------------------------------------- #
def _normalise(values):
    """GRPO advantage: (x - mean) / std  (std-normalised, as in the run config).
    If std == 0 the whole group cancels → zero gradient (this is the point)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * n, mean, 0.0
    return [(v - mean) / std for v in values], mean, std


def run_grpo_sim(rows):
    print(bold("\n" + "=" * 92))
    print(bold(" 2. GRPO SIMULATOR — group-normalised advantage:  baseline (R_answer)  vs  v3 (R_total)"))
    print("=" * 92)
    print(grey("   Under GRPO only WITHIN-group variance survives. A group where every rollout"))
    print(grey("   scores the same gives ZERO gradient. The question: does v3's shaping turn the"))
    print(grey("   majority all-wrong groups from 'no signal' into a USEFUL, correctness-aligned one?\n"))

    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    revived = 0
    for gname, members in groups.items():
        base_vals = [m["r_answer"] for m in members]
        v3_vals = [m["total"] for m in members]
        base_adv, base_mean, base_std = _normalise(base_vals)
        v3_adv, v3_mean, v3_std = _normalise(v3_vals)

        base_dead = base_std < 1e-8
        v3_live = v3_std >= 1e-8
        if base_dead and v3_live:
            revived += 1

        status = ""
        if base_dead and v3_live:
            status = green("  ← baseline DEAD (no gradient); v3 REVIVES it")
        elif base_dead and not v3_live:
            status = grey("  ← both dead")
        else:
            status = blue("  ← baseline already has a gradient")

        print(bold(f"   group '{gname}'  ({len(members)} rollouts)") + status)
        print(f"      {'id':>3} {'R_ans':>6} {'R_tot':>6} │ {'baseline adv':>22} │ {'v3 adv':>22}")
        print("      " + "─" * 70)
        for m, ba, va in zip(members, base_adv, v3_adv):
            print(f"      {m['id']:>3} {m['r_answer']:>+6.2f} {m['total']:>+6.3f} │ "
                  f"{ba:>+5.2f} {signed_bar(ba, -2, 2)} │ "
                  f"{va:>+5.2f} {signed_bar(va, -2, 2)}")
        # Does the v3 gradient point the right way? Correlate advantage with a
        # quick "reasoning quality" proxy (tool+grnd+syn-rep) within the group.
        print(grey(f"      baseline: mean={base_mean:+.3f} std={base_std:.3f}   "
                   f"v3: mean={v3_mean:+.3f} std={v3_std:.3f}"))
        print()

    print(f"   {green(str(revived))} of {len(groups)} groups were dead under the baseline but "
          f"carry a live gradient under v3.")
    print(grey("   (In the real run the all-wrong regime is ~87% of groups — every one of those"))
    print(grey("    is a no-op for the baseline. v3's value rests entirely on that gradient being"))
    print(grey("    correctness-aligned, which §3-criterion-2 / the correlation check below tests.)"))
    return groups


# --------------------------------------------------------------------------- #
# 3. Rubric verdict (REWARD_REASONING_ANALYSIS.md §3).
# --------------------------------------------------------------------------- #
def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def run_verdict(rows):
    print(bold("\n" + "=" * 92))
    print(bold(" 3. RUBRIC VERDICT — scored against REWARD_REASONING_ANALYSIS.md §3"))
    print("=" * 92)

    shapings = [r["shaping"] for r in rows]
    r_answers = [r["r_answer"] for r in rows]

    checks = []

    # (a) Cap invariant: |shaping| ≤ CAP < smallest tier gap (0.1).
    max_abs = max(abs(s) for s in shapings)
    cap_ok = max_abs <= K["CAP"] + 1e-9 and K["CAP"] < _SMALLEST_TIER_GAP
    checks.append(("Cap invariant  |shaping| ≤ CAP < 0.1 tier gap",
                   cap_ok,
                   f"max|shaping|={max_abs:.4f}, CAP={K['CAP']}, gap={_SMALLEST_TIER_GAP}"))

    # (b) No tier reorder: a wrong answer (best case 0+CAP) can never beat a
    #     top-100 hit (0.1 - CAP, but correct answers only get +shaping).
    worst_correct = 0.1  # smallest "real hit" tier; gets +shaping ≥ 0 → ≥ 0.1
    best_wrong = 0.0 + K["CAP"]
    reorder_ok = best_wrong < worst_correct
    checks.append(("No tier reorder  best-wrong < worst-real-hit",
                   reorder_ok,
                   f"best wrong={best_wrong:.3f} < top-100={worst_correct:.3f}"))

    # (c) Two-sided shaping (not a one-sided penalty like v2).
    has_pos = any(s > 1e-9 for s in shapings)
    has_neg = any(s < -1e-9 for s in shapings)
    twosided_ok = has_pos and has_neg
    checks.append(("Two-sided shaping  (both + and − occur)",
                   twosided_ok,
                   f"{sum(s>0 for s in shapings)} positive / "
                   f"{sum(s<0 for s in shapings)} negative / "
                   f"{sum(abs(s)<=1e-9 for s in shapings)} zero"))

    # (d) corr(shaping, R_answer) > 0  (process signal is a leading indicator).
    corr = _pearson(shapings, r_answers)
    corr_ok = corr > 0
    checks.append(("corr(shaping, R_answer) > 0  (process tracks correctness)",
                   corr_ok,
                   f"corr={corr:+.3f}  (v2 redundancy was ≈+0.10 ≈ noise; "
                   f"real v3 log ≈ +0.13)"))

    # (e) Asymmetry never demoted a correct answer.
    asym_ok = all(r["shaping"] >= 0 for r in rows if r["r_answer"] >= 0.5)
    checks.append(("LongPAS  correct answers never penalised",
                   asym_ok,
                   "all R_answer≥0.5 rollouts received shaping ≥ 0"))

    name_w = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        mark = green("PASS") if ok else red("FAIL")
        print(f"   [{mark}] {name:<{name_w}}   {grey(detail)}")

    print()
    n_pass = sum(ok for _, ok, _ in checks)
    if n_pass == len(checks):
        print(green(bold(f"   {n_pass}/{len(checks)} invariants hold — the v3 MATH is wired correctly "
                         f"and is GRPO-safe.")))
    else:
        print(red(bold(f"   {n_pass}/{len(checks)} invariants hold — investigate the failures above.")))

    print(grey(
        "\n   Caveat (from §4 of the analysis): passing these invariants means the math is\n"
        "   sound and the gradient is live & correctness-aligned ON THIS SAMPLE. It does NOT\n"
        "   by itself prove held-out improvement — the real run showed val r_answer stayed\n"
        "   flat (~0) and training collapsed after ~step 300 from entropy collapse. The\n"
        "   north-star remains: held-out r_answer rising and beating the baseline. This\n"
        "   simulator validates the reward DEFINITION; the training run validates the OUTCOME."))


# --------------------------------------------------------------------------- #
# 4. Optional: recompute components from raw text via the real v3 reasoning.py.
# --------------------------------------------------------------------------- #
def run_recompute():
    print(bold("\n" + "=" * 92))
    print(bold(" 4. PIPELINE SMOKE-TEST — recompute components from raw text (real reasoning.py)"))
    print("=" * 92)
    try:
        from verl.utils.reward_score.reward_reasoning.v3 import reasoning
    except Exception as e:
        print(red(f"   SKIPPED — could not import v3 reasoning.py: {e}"))
        print(grey("   This layer needs torch + sentence-transformers; run it inside the"))
        print(grey("   singularity container (the same place the GPU nodes run the reward)."))
        return

    print(grey("   Re-deriving {tool,grnd,syn,rep} from each illustrative solution string.\n"))
    print(grey("   NOTE: text is a reconstruction, so values approximate (not equal) the logged"))
    print(grey("   line; what we verify is that the extractor RUNS and is directionally sane.\n"))
    hdr = (f"   {'id':>3} │ {'tool':>16} │ {'grnd':>16} │ {'syn':>16} │ {'rep':>16}")
    print(bold(hdr)); print("   " + "─" * (len(hdr) - 3))
    for r in RUNS:
        try:
            comp = reasoning.compute_process_components(
                r["solution"], data_source="amazon",
                prompt_text="The user recently played classic rock and jazz albums.")
        except Exception as e:
            print(f"   {r['id']:>3} │ {red('error: ' + str(e)[:50])}")
            continue

        print(f"   {r['id']:>3} │ "
              f"{comp['tool_use']:.2f} (log {r['tool']:.2f}) │ "
              f"{comp['grounding']:.2f} (log {r['grnd']:.2f}) │ "
              f"{comp['synthesis']:.2f} (log {r['syn']:.2f}) │ "
              f"{comp['self_rep']:.2f} (log {r['rep']:.2f})")
    print(grey("\n   If these columns are populated and broadly track the logged magnitudes,"))
    print(grey("   the text→component pipeline is healthy end to end."))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 4. v4 dense-reward demo (the GRPO root-cause fix, --v4).
# --------------------------------------------------------------------------- #
def _tier(rank_id):
    """The v3/baseline tiered r_answer as a pure fn of rank (reward_SPRec.py)."""
    if rank_id == 1:    return 1.0
    if rank_id <= 5:    return 0.8
    if rank_id <= 10:   return 0.5
    if rank_id <= 100:  return 0.1
    if rank_id <= 500:  return 0.001
    return 0.0


def run_v4_dense_demo():
    """Show, on representative all-wrong GRPO groups, that v4's r_dense creates a
    live, correctness-aligned gradient exactly where the tiered reward is dead.

    Uses the *live* v4 `_dense_from_rank` (so RTHINK_DENSE_P / RTHINK_RANK_FLOOR /
    N are honoured) — no torch needed; it's pure math."""
    # Load the v4 orchestrator by file path so we don't trigger verl/__init__.py
    # (which needs heavy deps). orchestrator.py imports only math/os/random.
    try:
        from verl.utils.reward_score.reward_reasoning.v4 import orchestrator as v4o
    except Exception:
        import importlib.util
        _v4_path = os.path.join(os.path.dirname(__file__), "v4", "orchestrator.py")
        _spec = importlib.util.spec_from_file_location("_v4_orchestrator", _v4_path)
        v4o = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(v4o)

    N = 13079
    print(bold("\n" + "=" * 92))
    print(bold(" 4. v4 DENSE REWARD — does it revive the dead all-wrong groups?"))
    print("=" * 92)
    print(grey(f"   r_dense = (1 - log(rank)/log(N))**p   |   N={N}  p={v4o._DENSE_P}  "
               f"floor={v4o._RANK_FLOOR}  dense_only={v4o._DENSE_ONLY}"))
    print(grey("   A group = several rollouts of ONE prompt. 'all-wrong' = every rollout ranks"))
    print(grey("   the target past the top-10, so the tiered reward is a flat 0 → zero gradient.\n"))

    # Three all-wrong groups with realistic spreads (none in the top-10).
    groups = {
        "all-wrong (close-ish)": [120, 340, 800],
        "all-wrong (scattered)": [600, 3000, 9000],
        "all-wrong (hopeless)":  [8000, 11000, 13000],
    }
    print(f"   {'group':<24} {'rank':>6}  {'tier':>6} {'dense':>7}   "
          f"{'baseline_adv':>12}  {'v4_adv':>8}")
    all_ranks, all_dense = [], []
    for name, ranks in groups.items():
        tiers = [_tier(r) for r in ranks]
        dense = [v4o._dense_from_rank(r, N) for r in ranks]
        base_adv, _, base_std = _normalise(tiers)
        v4_adv, _, v4_std = _normalise(dense)
        for i, r in enumerate(ranks):
            tag = name if i == 0 else ""
            verdict = ""
            if i == 0:
                if base_std < 1e-8 and v4_std >= 1e-8:
                    verdict = green("  ← DEAD for baseline; v4 REVIVES it")
                elif base_std < 1e-8:
                    verdict = grey("  ← dead for both")
            print(f"   {tag:<24} {r:>6}  {tiers[i]:>6.3f} {dense[i]:>7.3f}   "
                  f"{base_adv[i]:>+12.3f}  {v4_adv[i]:>+8.3f}{verdict}")
            all_ranks.append(r)
            all_dense.append(dense[i])
        print()

    corr = _pearson(all_dense, [-r for r in all_ranks])
    ok = corr > 0.9
    line = green("PASS") if ok else red("FAIL")
    print(f"   corr(r_dense, -rank) = {corr:+.3f}   [{line}]  "
          + grey("(dense must track correctness: closer rank → higher reward)"))
    print(grey("\n   Takeaway: where the tiered reward gives every all-wrong rollout 0 "
               "(baseline_adv≈0,\n   no gradient), v4's r_dense spreads them by how close "
               "they got — so GRPO finally\n   has a 'name something closer' gradient in "
               "the 87% of groups it spends its time in."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true",
                    help="also re-run the real v3 component extractor on the "
                         "solution strings (needs torch + sentence-transformers)")
    ap.add_argument("--v4", action="store_true",
                    help="run the v4 dense-reward demo: show r_dense reviving the "
                         "dead all-wrong GRPO groups (pure math, no GPU)")
    args = ap.parse_args()

    if args.v4:
        print(bold(cyan("\n  REC-R1 reasoning-reward simulator  —  v4 dense-reward demo")))
        run_v4_dense_demo()
        print()
        return

    print(bold(cyan("\n  REC-R1 reasoning-reward simulator  —  v3 math, no retriever, no GPU needed")))
    print(grey("  data: 10 real rollouts from nq-...-rthink-v3.log  |  "
               "math: live orchestrator constants"))

    rows = run_math_engine()
    run_grpo_sim(rows)
    run_verdict(rows)
    if args.recompute:
        run_recompute()
    else:
        print(grey("\n  (re-run with --recompute, inside the container, to also test the "
                   "text→component pipeline)"))
    print()


if __name__ == "__main__":
    main()
