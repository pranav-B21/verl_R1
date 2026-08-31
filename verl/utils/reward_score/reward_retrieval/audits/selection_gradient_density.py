# SPDX-License-Identifier: Apache-2.0
"""Probe (a) -- does the v8 selection term carry a GRPO gradient AT ALL?

The v8 null (`REWARD_REASONING_ANALYSIS.md` Part 10) is read as "reward shaping
does not move selection". That reading is only valid if the selection term ever
*differed between rollouts of the same prompt*. In GRPO the advantage is

    A_ij = (R_ij - mean_j R_ij) / std_j R_ij

so any reward component that takes the SAME value on all `n` rollouts of a
prompt is subtracted out exactly and contributes zero gradient. A term can be
large, well-motivated, and perfectly correlated with the outcome, and still
teach nothing if it is constant within the group.

This script measures that, two independent ways, with no GPU and no corpus.

PART 1 -- state collapse (exact, analytic)
    v8 shaping is
        r_select = w_sel*is_succ + w_grnd*is_grnd - w_rep*is_rep
        applied  = clip(SCALE * (r_select + w_cov*r_cover), -CAP, +CAP)
    `clip` is not order-preserving. Enumerate the discrete state space and
    report, for every (is_grnd, is_rep, r_cover) cell, the gap
        d_succ = applied(is_succ=1) - applied(is_succ=0)
    d_succ == 0 means the successor signal -- the whole point of v8 -- is
    absorbed by the cap and is invisible to the optimizer in that cell.
    Run for the AS-RUN weights (w_grnd=0.4, the launcher collision, see
    `v8-grounding-weight-collision`) and the DESIGNED weights (w_grnd=0.1).

PART 2 -- where training actually sat (empirical, from the training log)
    v8/orchestrator.py prints one rollout in 64 (`random.randint(1,64)==1`).
    Parse those lines out of the run logs to get the real joint distribution of
    (is_succ, is_grnd, is_rep, r_cover), and weight Part 1's cells by it: what
    fraction of scored rollouts sat in a cell where d_succ == 0?

PART 3 -- group gradient density (the headline number)
    Eval decodes are sampled at temperature 1.0 -- the SAME distribution the
    training rollouts are drawn from -- and EVAL_REPEATS gives 3 independent
    draws per prompt. Recompute `applied` for each draw with the real v8 code,
    then ask what fraction of prompts have a non-constant `applied`.
    That is measured directly at n=3; the n=8 figure the trainer actually saw
    is a Beta-Binomial extrapolation (closed form, assumption stated inline).

Usage (bare login-node python3; no verl install needed):

    python3 selection_gradient_density.py \
        --logs ../../../../../output_rthink_v8_n8*.log \
        --preds outputs/eval/<arm>/global_step_*/decode*/test_predictions.json \
        --json selection_gradient_density.json
"""

import argparse
import importlib.util
import json
import os
import re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    """Load a sibling module BY PATH -- same convention as decode_selection.py.

    Importing through `verl.…` would pull in verl/__init__.py and the whole
    training stack; these audits must run on a bare login-node python.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load("decode_behavior", os.path.join(_HERE, "decode_behavior.py"))
selection = _load("v8_selection", os.path.join(os.path.dirname(_HERE), "v8", "selection.py"))
dsel = _load("decode_selection", os.path.join(_HERE, "decode_selection.py"))

# v8/orchestrator.py defaults, mirrored so this runs without importing verl.
SCALE, CAP = 0.10, 0.08
W_SELECT, W_REPEAT, W_COVER = 0.5, 0.3, 0.4
AS_RUN = dict(w_sel=W_SELECT, w_grnd=0.4, w_rep=W_REPEAT, w_cov=W_COVER)   # launcher collision
DESIGNED = dict(w_sel=W_SELECT, w_grnd=0.1, w_rep=W_REPEAT, w_cov=W_COVER)  # code default


def applied(is_succ, is_grnd, is_rep, r_cover, w):
    """v8/orchestrator.py:239-241, verbatim."""
    r_select = w["w_sel"] * is_succ + w["w_grnd"] * is_grnd - w["w_rep"] * is_rep
    raw = r_select + w["w_cov"] * r_cover
    return max(-CAP, min(CAP, SCALE * raw))


# --------------------------------------------------------------------------- #
# PART 1 -- exact state collapse
# --------------------------------------------------------------------------- #
COVER_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def part1(w):
    rows = []
    for is_grnd in (0, 1):
        for is_rep in (0, 1):
            for cov in COVER_GRID:
                a0 = applied(0, is_grnd, is_rep, cov, w)
                a1 = applied(1, is_grnd, is_rep, cov, w)
                rows.append(dict(is_grnd=is_grnd, is_rep=is_rep, r_cover=cov,
                                 applied_succ0=round(a0, 4), applied_succ1=round(a1, 4),
                                 d_succ=round(a1 - a0, 4), collapsed=(a1 == a0)))
    return rows


# --------------------------------------------------------------------------- #
# PART 2 -- empirical state distribution from the 1-in-64 training prints
# --------------------------------------------------------------------------- #
LINE = re.compile(
    r"\[Rthink-v8\].*?cover=(?P<cover>[-\d.]+).*?"
    r"succ=(?P<succ>\d+)\s+grnd=(?P<grnd>\d+)\s+rep=(?P<rep>\d+)\s+"
    r"n_succ=(?P<n_succ>\d+)\s+n_items=(?P<n_items>\d+)\s+"
    r"shaping=(?P<shaping>[-+\d.]+)"
)


def part2(log_paths):
    """Parse the 1-in-64 prints. NOTE: do NOT dedupe identical lines.

    Distinct rollouts routinely print byte-identical payloads -- the modal state
    (r_ans=0, succ=0, grnd=0, cover=0) is a large share of the batch, and rank/N
    do not disambiguate it. Deduping collapsed 3555 real observations in one log
    to 1060 and biased the distribution hard toward rare states. The caller must
    instead pass DISJOINT log segments: the sbatch -o logs, never also the tee'd
    nq-*.log copy of one of them (the fingerprint check below catches that).
    """
    states, fingerprints = [], {}
    for p in log_paths:
        lines = []
        with open(p, "rb") as fh:
            for raw in fh:
                if b"[Rthink-v8]" not in raw:
                    continue
                m = LINE.search(raw.decode("utf-8", "replace"))
                if m:
                    lines.append(m.groupdict())
        if not lines:
            continue
        fp = (len(lines), str(lines[0]), str(lines[-1]))
        if fp in fingerprints:
            print(f"[warn] {os.path.basename(p)} looks like a duplicate of "
                  f"{os.path.basename(fingerprints[fp])} ({len(lines)} identical "
                  f"prints, same first/last) -- skipping it")
            continue
        fingerprints[fp] = p
        for g in lines:
            states.append(dict(is_succ=int(g["succ"]), is_grnd=int(g["grnd"]),
                               is_rep=int(g["rep"]), r_cover=float(g["cover"]),
                               shaping=float(g["shaping"]),
                               n_items=int(g["n_items"])))
    if not states:
        return None

    out = {"n_sampled_rollouts": len(states), "n_logs_used": len(fingerprints)}
    for tag, w in (("as_run", AS_RUN), ("designed", DESIGNED)):
        collapsed = sum(
            1 for s in states
            if applied(1, s["is_grnd"], s["is_rep"], s["r_cover"], w)
            == applied(0, s["is_grnd"], s["is_rep"], s["r_cover"], w)
        )
        at_cap = sum(
            1 for s in states
            if abs(applied(s["is_succ"], s["is_grnd"], s["is_rep"], s["r_cover"], w)) >= CAP - 1e-12
        )
        out[tag] = dict(succ_signal_collapsed=collapsed,
                        succ_signal_collapsed_frac=collapsed / len(states),
                        at_cap=at_cap, at_cap_frac=at_cap / len(states))
    for k in ("is_succ", "is_grnd", "is_rep"):
        out[f"mean_{k}"] = sum(s[k] for s in states) / len(states)
    out["mean_r_cover"] = sum(s["r_cover"] for s in states) / len(states)
    out["frac_cover_ge_0.99"] = sum(1 for s in states if s["r_cover"] >= 0.99) / len(states)
    # what the run actually logged, as a cross-check on the recomputation
    out["observed_shaping_hist"] = dict(
        sorted(Counter(round(s["shaping"], 4) for s in states).items()))
    return out


# --------------------------------------------------------------------------- #
# PART 3 -- group gradient density
# --------------------------------------------------------------------------- #
def _rollout_state(rec):
    """(is_succ, is_grnd, is_rep, covered) for ONE decoded rollout.

    Mirrors decode_selection.analyze()'s per-record block so the two instruments
    cannot drift. r_cover is the graded coverage term of v8/orchestrator.py,
    which needs the reward's embedding model; here `covered` is its binary
    exact-title core (GT literally among the retrieved items), which is what the
    r_cover=1.0 spike in the training log corresponds to.
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
        picked_gt=int(bool(gt) and pred == gt),
        n_items=len(items),
    )


def _beta_binom_moments(counts, n_draws):
    """Method-of-moments Beta fit to per-prompt success counts out of n_draws."""
    k = len(counts)
    m1 = sum(counts) / k / n_draws
    m2 = sum(c * (c - 1) for c in counts) / k / (n_draws * (n_draws - 1))
    if m1 <= 0 or m1 >= 1 or m2 <= m1 * m1:
        return None  # degenerate / under-dispersed: no Beta fits
    # standard MoM: a = (m1*(m1 - m2)) / (m2 - m1^2);  b = a*(1 - m1)/m1
    a = (m1 * (m1 - m2)) / (m2 - m1 * m1)
    if a <= 0:
        return None
    b = a * (1 - m1) / m1
    return a, b


def _e_p_pow(a, b, n):
    """E[p^n] for p ~ Beta(a, b), closed form."""
    out = 1.0
    for j in range(n):
        out *= (a + j) / (a + b + j)
    return out


def part3(pred_paths, group_n=8):
    # group decodes by (arm, step); the repeats of one step are the replicates
    by_step = {}
    for p in pred_paths:
        step_dir = os.path.dirname(os.path.dirname(os.path.abspath(p)))
        by_step.setdefault(step_dir, []).append(p)

    results = []
    for step_dir, paths in sorted(by_step.items()):
        if len(paths) < 2:
            continue
        per_prompt = {}
        for p in sorted(paths):
            for i, rec in enumerate(json.load(open(p, encoding="utf-8"))):
                per_prompt.setdefault(i, []).append(_rollout_state(rec))
        reps = [v for v in per_prompt.values() if len(v) == len(paths)]
        if not reps:
            continue
        n_draws = len(paths)

        row = dict(step_dir=os.path.relpath(step_dir), n_prompts=len(reps), n_draws=n_draws)
        for tag, w in (("as_run", AS_RUN), ("designed", DESIGNED)):
            vals = [[applied(s["is_succ"], s["is_grnd"], s["is_rep"],
                             float(s["covered"]), w) for s in v] for v in reps]
            varying = [len(set(round(x, 6) for x in v)) > 1 for v in vals]
            d_direct = sum(varying) / len(varying)

            # n=8 extrapolation: per prompt, "applied == that prompt's modal
            # value" is Bernoulli(p_i); fit p_i ~ Beta by moments across prompts
            # and read off P(all `group_n` draws identical) = E[p^group_n].
            # ASSUMPTION: within a prompt the draws are iid (true -- they are
            # independent samples at the same temperature) and the non-modal
            # mass behaves as a single alternative (conservative: splitting it
            # further would only make groups MORE degenerate, i.e. density lower).
            counts = []
            for v in vals:
                mode = Counter(round(x, 6) for x in v).most_common(1)[0][0]
                counts.append(sum(1 for x in v if round(x, 6) == mode))
            fit = _beta_binom_moments(counts, n_draws)
            if fit:
                a, b = fit
                d_n = 1.0 - _e_p_pow(a, b, group_n)
                beta = dict(alpha=round(a, 4), beta=round(b, 4))
            else:  # no Beta fits -> fall back to the iid-per-prompt plug-in
                pbar = sum(counts) / (len(counts) * n_draws)
                d_n, beta = 1.0 - pbar ** group_n, None
            row[tag] = dict(density_measured_at_n=round(d_direct, 5),
                            density_extrapolated_n8=round(d_n, 5), beta_fit=beta)
        # context: how often the group could carry ANY selection signal at all
        row["coverage"] = sum(any(s["covered"] for s in v) for v in reps) / len(reps)
        row["mean_items"] = sum(s["n_items"] for v in reps for s in v) / sum(len(v) for v in reps)
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", default=[], help="training log(s) with [Rthink-v8] prints")
    ap.add_argument("--preds", nargs="*", default=[], help="test_predictions.json replicates")
    ap.add_argument("--group-n", type=int, default=8, help="rollout.n the trainer used")
    ap.add_argument("--json", help="write the full result to this path")
    args = ap.parse_args()

    out = {"part1_state_collapse": {t: part1(w) for t, w in
                                    (("as_run", AS_RUN), ("designed", DESIGNED))}}

    print("=" * 78)
    print("PART 1 -- does the cap erase the successor signal? (exact)")
    print("=" * 78)
    for tag in ("as_run", "designed"):
        w = AS_RUN if tag == "as_run" else DESIGNED
        rows = out["part1_state_collapse"][tag]
        ncol = sum(r["collapsed"] for r in rows)
        print(f"\n[{tag}] w_grnd={w['w_grnd']}   SCALE={SCALE} CAP={CAP}")
        print(f"  {'grnd':>4} {'rep':>4} {'cover':>6} {'succ=0':>8} {'succ=1':>8} {'d_succ':>8}")
        for r in rows:
            flag = "  <- COLLAPSED" if r["collapsed"] else ""
            print(f"  {r['is_grnd']:>4} {r['is_rep']:>4} {r['r_cover']:>6.2f} "
                  f"{r['applied_succ0']:>8.4f} {r['applied_succ1']:>8.4f} "
                  f"{r['d_succ']:>8.4f}{flag}")
        print(f"  cells where the successor term is invisible: {ncol}/{len(rows)}")

    if args.logs:
        p2 = part2(args.logs)
        out["part2_training_states"] = p2
        print("\n" + "=" * 78)
        print("PART 2 -- where training actually sat (1-in-64 sampled rollouts)")
        print("=" * 78)
        if not p2:
            print("  no [Rthink-v8] lines parsed")
        else:
            print(f"  sampled rollouts: {p2['n_sampled_rollouts']} "
                  f"(from {p2['n_logs_used']} log segment(s))")
            print(f"  mean is_succ={p2['mean_is_succ']:.4f}  is_grnd={p2['mean_is_grnd']:.4f}  "
                  f"is_rep={p2['mean_is_rep']:.4f}  r_cover={p2['mean_r_cover']:.4f}")
            print(f"  fraction with r_cover >= 0.99: {p2['frac_cover_ge_0.99']:.4f}")
            for tag in ("as_run", "designed"):
                d = p2[tag]
                print(f"  [{tag}] rollouts where succ signal is collapsed: "
                      f"{d['succ_signal_collapsed']}/{p2['n_sampled_rollouts']} "
                      f"({d['succ_signal_collapsed_frac']:.1%})   at cap: {d['at_cap_frac']:.1%}")

    if args.preds:
        p3 = part3(args.preds, args.group_n)
        out["part3_group_density"] = p3
        print("\n" + "=" * 78)
        print(f"PART 3 -- fraction of GRPO groups (n={args.group_n}) with a non-constant "
              f"selection term")
        print("=" * 78)
        print(f"  {'step':<52} {'cov':>6} {'as_run n3':>10} {'as_run n8':>10} "
              f"{'des n3':>8} {'des n8':>8}")
        for r in p3:
            print(f"  {r['step_dir'][-52:]:<52} {r['coverage']:>6.3f} "
                  f"{r['as_run']['density_measured_at_n']:>10.4f} "
                  f"{r['as_run']['density_extrapolated_n8']:>10.4f} "
                  f"{r['designed']['density_measured_at_n']:>8.4f} "
                  f"{r['designed']['density_extrapolated_n8']:>8.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
