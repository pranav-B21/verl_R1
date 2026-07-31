# SPDX-License-Identifier: Apache-2.0
"""Report decode-time retrieval behavior from an eval ``test_predictions.json``.

Why this exists: the M3 A/B (v7 retrieval-quality reward vs outcome-only
baseline) turns on *whether the model still retrieves*. During training the v7
arm collapsed from ~0.80 to ~0.02 tool-call turns per rollout, so an HR number
alone cannot tell the two arms apart -- a v7 checkpoint that never retrieves is
a different system from the baseline even when their HR ties. eval.py reports
only HR/NDCG/ORRatio, so this fills the gap.

The prediction JSON keeps the full multi-turn response text (tool_call and
tool_response markup included), so behavior is recoverable without re-running
generation.

Usage:
    python3 decode_behavior.py preds1.json [preds2.json ...]
    python3 decode_behavior.py --json out.json preds*.json
"""

import argparse
import json
import re

# Qwen3-native tool calling (type: native in search_tool_config.yaml). The
# <search>/<info> forms exist elsewhere in this codebase but are dead at runtime
# under the current config -- see reward_retrieval/v7/README.md "Tags parsed".
TOOL_CALL = re.compile(r"<tool_call>")
TOOL_RESP = re.compile(r"<tool_response>")
TOOL_RESP_END = re.compile(r"</tool_response>")
DOC = re.compile(r"Doc \d+ \(Title:")
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
QUERY_LIST = re.compile(r'"query_list"\s*:\s*\[(.*?)\]', re.S)

# Retrieved docs are user-history sequences listing catalog items as quoted
# titles: Doc 1 (Title: A user played ... "12 Classic Albums: 1956-1962", ...).
# The payload is JSON-encoded inside <tool_response>, so the quotes arrive
# escaped; fall back to bare quotes for any decode that was not escaped.
ITEM_TITLE = re.compile(r'\\"([^"\\]{1,200})\\"')
ITEM_TITLE_BARE = re.compile(r'"([^"\\]{1,200})"')


def _retrieved_spans(text):
    """Text of each <tool_response>, STOPPING at the closing tag.

    The window must be bounded. An unbounded read (the original `m.end():
    m.end()+4000`) runs past </tool_response> on essentially every rollout --
    median response is ~1.1-1.8k chars -- and swallows the model's own <think>
    text and <answer>. Coverage then counts "the model mentioned the GT" as
    "retrieval returned the GT", and does so more often for arms that generate
    more post-retrieval text, which is exactly what the length penalty changes.
    """
    spans = []
    for m in TOOL_RESP.finditer(text):
        end = TOOL_RESP_END.search(text, m.end())
        spans.append(text[m.end(): end.start() if end else len(text)])
    return spans


def retrieved_titles(text):
    """Set of normalized catalog titles actually present in the retrieved docs.

    Exact title membership, not substring containment. A substring test counts
    GT 'boy' as covered by the retrieved title 'boys & girls', and GT '19' by
    'the best of 1980-1990'; ~40% of apparent hits were of that kind, and the
    rate scales with how much text an arm retrieves, so it inflates whichever
    arm retrieves more.
    """
    titles = set()
    for span in _retrieved_spans(text):
        for raw in (ITEM_TITLE.findall(span) or ITEM_TITLE_BARE.findall(span)):
            name = _norm(raw)
            # Drop the boilerplate lead-in of each history doc.
            if name and not name.startswith("a user played"):
                titles.add(name)
    return titles


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower().strip().strip('"')).strip()


def analyze(path):
    records = json.load(open(path, encoding="utf-8"))
    n = len(records)
    if not n:
        raise SystemExit(f"{path}: no records")

    turns, docs, queries = [], [], []
    n_any_retrieval = n_answer = n_multi_answer = n_grounded = 0
    n_gt = n_gt_in_docs = n_gt_in_docs_ret = 0

    for rec in records:
        text = "".join(rec.get("predict") or [])
        t = len(TOOL_CALL.findall(text))
        d = len(DOC.findall(text))
        turns.append(t)
        docs.append(d)
        if t:
            n_any_retrieval += 1
        for q in QUERY_LIST.findall(text):
            queries.append(q.count('"') // 2)

        # COVERAGE (the retrieval-quality dev metric): does the ground-truth next
        # item appear anywhere in the retrieved docs? This is what r_retqual is
        # meant to raise; it has ~10x the statistical power of HR (~31/1000 vs
        # ~3/1000). `gt_in_docs_of_retrieving` isolates query quality from the
        # retrieval RATE. GT is the record's held-out target (`output`).
        #
        # Matching is EXACT-TITLE over a span bounded by </tool_response>; see
        # retrieved_titles(). The earlier unbounded-substring form overstated
        # coverage by ~1.9x (baseline 5.90% -> 3.13% corrected) and overstated
        # it more for text-heavier arms, which manufactured a spurious
        # baseline-vs-v7b gap. Do not loosen this back to `gt in text`.
        gt = _norm(rec.get("output", ""))
        if gt:
            n_gt += 1
            if gt in retrieved_titles(text):
                n_gt_in_docs += 1
                if t:
                    n_gt_in_docs_ret += 1

        answers = ANSWER.findall(text)
        if answers:
            n_answer += 1
            if len(answers) > 1:
                n_multi_answer += 1
            # "Grounded" = the emitted answer string appears in a retrieved doc.
            # Cheap substring test, matching the grounding forensics convention.
            final = answers[-1].strip().strip('"').strip()
            if final and TOOL_RESP.search(text):
                resp = "".join(
                    text[m.end():m.end() + 4000] for m in TOOL_RESP.finditer(text)
                )
                if final and final in resp:
                    n_grounded += 1

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    # POSITIVE CONTROL. A dead retriever and a collapsed policy both drive
    # retrieval_rate toward zero, so a low rate alone cannot tell "the model chose
    # not to retrieve" from "retrieval was broken". docs_per_retrieval is measured
    # only over rollouts that DID issue a tool call: if it stays healthy (~7, the
    # topk*queries the service returns) then the retriever was alive and the low
    # rate is a genuine policy result. Near 0 here means the service failed and the
    # decode must be discarded, not interpreted.
    docs_when_retrieved = [d for d, t in zip(docs, turns) if t >= 1]
    return {
        "file": path,
        "n": n,
        "retrieval_rate": n_any_retrieval / n,
        "turns_mean": mean(turns),
        "turns_max": max(turns),
        "docs_mean": mean(docs),
        "docs_per_retrieval": mean(docs_when_retrieved),
        "queries_per_call_mean": mean(queries),
        "answer_rate": n_answer / n,
        "multi_answer_rate": n_multi_answer / n,
        "grounded_rate_of_answered": (n_grounded / n_answer) if n_answer else 0.0,
        # coverage — the retrieval-quality dev metric (see loop comment)
        "gt_in_docs_rate": (n_gt_in_docs / n_gt) if n_gt else 0.0,
        "gt_in_docs_of_retrieving": (n_gt_in_docs_ret / n_any_retrieval) if n_any_retrieval else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preds", nargs="+", help="test_predictions.json path(s)")
    ap.add_argument("--json", help="also write the rows to this JSON file")
    args = ap.parse_args()

    rows = []
    for p in args.preds:
        try:
            rows.append(analyze(p))
        except Exception as exc:  # a missing/short decode should not kill the batch
            print(f"[skip] {p}: {exc}")

    if not rows:
        raise SystemExit("no readable prediction files")

    hdr = ("decode", "n", "retr%", "turns", "docs", "docs/ret", "q/call", "ans%", "multi%",
           "grnd%", "GTdocs%", "GTdocs%|ret")
    print("%-46s %5s %7s %7s %7s %9s %7s %7s %7s %7s %8s %11s" % hdr)
    print("%-46s %5s %7s %7s %7s %9s %7s %7s %7s %7s %8s %11s" % tuple("-" * len(h) for h in hdr))
    for r in rows:
        # Keep the last two path components: <step>/<decode_tag>.
        label = "/".join(r["file"].split("/")[-3:-1]) or r["file"]
        print(
            "%-46s %5d %6.1f%% %7.3f %7.2f %9.2f %7.2f %6.1f%% %6.1f%% %6.1f%% %7.1f%% %10.1f%%"
            % (
                label[-46:],
                r["n"],
                100 * r["retrieval_rate"],
                r["turns_mean"],
                r["docs_mean"],
                r["docs_per_retrieval"],
                r["queries_per_call_mean"],
                100 * r["answer_rate"],
                100 * r["multi_answer_rate"],
                100 * r["grounded_rate_of_answered"],
                100 * r["gt_in_docs_rate"],
                100 * r["gt_in_docs_of_retrieving"],
            )
        )

    print()
    print("retr%    = share of test prompts where the model issued >=1 <tool_call>.")
    print("docs/ret = docs returned per RETRIEVING rollout -- the positive control.")
    print("q/call   = query_list entries per tool call. docs/ret ~= q/call * topk(3), so read")
    print("           a docs/ret difference here BEFORE calling it a targeting difference.")
    print("GTdocs%      = share of prompts where the GT next-item appears in the retrieved docs,")
    print("               by EXACT title match inside <tool_response>...</tool_response>.")
    print("GTdocs%|ret  = same, over RETRIEVING rollouts only (isolates query quality from rate).")
    print("               This is the retrieval-quality DEV METRIC r_retqual should raise")
    print("               (~31/1000, ~10x the statistical power of HR).")
    print("MEASURED @ global_step_200 (corrected metric, 2026-07-30):")
    print("               baseline-n8  3.13%  (188/6000, 6 decodes)")
    print("               v7b-n8       2.77%  ( 83/3000, 3 decodes)   z=0.96, p=0.34 -> tie")
    print("               v7-n8        0.13%  (  4/3000)  -- retrieval had collapsed")
    print("NOTE: the pre-2026-07-30 form of this metric (unbounded 4000-char window +")
    print("      substring test) read ~1.9x high -- baseline 5.90%, v7b 4.70% -- and")
    print("      inflated text-heavier arms more. Coverage figures quoted from logs")
    print("      dated before 2026-07-30, including the '~4.6% ceiling', are on that")
    print("      inflated scale and are NOT comparable to the numbers above.")
    print()
    print("Read them together. A dead retriever and a collapsed policy both push")
    print("retr% toward 0, and only docs/ret separates them:")
    print("  retr% low,  docs/ret ~7  -> real policy collapse; the finding is valid.")
    print("  retr% low,  docs/ret ~0  -> the retrieval service failed. DISCARD the")
    print("                              decode; do not report it as a result.")
    print("Baseline-n8 @ step 200 reference: retr% ~99, docs/ret ~7.3.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
