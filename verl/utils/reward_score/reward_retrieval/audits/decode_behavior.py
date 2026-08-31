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
TOOL_CALL_OBJ = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

# Retrieved docs are user-history sequences listing catalog items as quoted
# titles: Doc 1 (Title: A user played ... "12 Classic Albums: 1956-1962", ...).
# The payload is JSON-encoded inside <tool_response>, so the quotes arrive
# escaped; fall back to bare quotes for any decode that was not escaped.
ITEM_TITLE = re.compile(r'\\"([^"\\]{1,200})\\"')
ITEM_TITLE_BARE = re.compile(r'"([^"\\]{1,200})"')


def _query_counts(text):
    """(n_queries, n_quoted_titles_inside_them) for each <tool_call>.

    CORRECTED 2026-08-07. The previous form counted quote PAIRS in the raw
    `"query_list": [...]` slice:

        queries.append(q.count('"') // 2)

    which conflates the number of queries with the number of quoted item titles
    the model pastes INTO a query. The policy learns over training to quote the
    user's history verbatim into a single query, so the old metric read

        broken = n_queries + n_quoted_titles

    and climbed 1.08 -> 9.39 (v8 steps 200->600) while the true count never left
    1.0. That artifact is what made it look like the model emitted nine queries
    and the retriever returned only three; measured directly, the service returns
    exactly topk(3) x n_queries on every call, with no exceptions. Do not read a
    docs/ret vs q/call gap off the old numbers -- there was never a gap.

    n_quoted_titles is kept as its own column because the shift it tracks is
    real and is the more interesting behavioural change of the two.
    """
    out = []
    for m in TOOL_CALL_OBJ.finditer(text):
        try:
            ql = json.loads(m.group(1))["arguments"]["query_list"]
        except Exception:  # malformed tool call -- counted by the JSON audit, not here
            continue
        if isinstance(ql, str):  # single query emitted unwrapped
            ql = [ql]
        out.append((len(ql), sum(s.count('"') // 2 for s in ql if isinstance(s, str))))
    return out


# --- memory type -------------------------------------------------------------
# RRCM is a DUAL-memory system: collaborative history memory and item-metadata
# memory, served from one corpus (data/amazon_data/corpora.jsonl = 72,191 CF docs
# + 13,111 metadata docs) through one retrieval interface. Every audit in this
# directory parsed retrieved docs for *item sequences*, so all of them silently
# dropped metadata docs and none reported a memory-type breakdown -- which is how
# the extinction in Part 11 went unnoticed for the whole v2-v8 arc.
#
# The two doc shapes, as they arrive JSON-escaped inside <tool_response>:
#   CF   Doc 1 (Title: A user played the following musics before in sequence: \"X\", \"Y\")
#   META Doc 2 (Title: Title: Central Heating: Expanded Edition, Price: , SalesRank: None,
#                Brand: Heatwave, Categories: )
CF_DOC_LEAD = re.compile(r"a user played the following", re.I)
META_DOC_FIELDS = re.compile(r"Price:.*?SalesRank:", re.S)


def _doc_bodies(text):
    """Body text of every retrieved doc, split at the `Doc N (Title:` markers.

    Split on marker POSITIONS rather than matching a closing `)`. The v7 `DOC`
    regex in retrieval.py required a literal `)\\n` and therefore dropped exactly
    one doc per response -- always the last -- in 100% of responses (recall 67.2%).
    A position split cannot have that failure mode, and `_memtype_counts` asserts
    its own total against `DOC.findall` so any drift is loud.
    """
    bodies = []
    for span in _retrieved_spans(text):
        marks = [m.start() for m in DOC.finditer(span)]
        for i, start in enumerate(marks):
            end = marks[i + 1] if i + 1 < len(marks) else len(span)
            bodies.append(span[start:end])
    return bodies


def _memtype_counts(text):
    """(n_cf, n_meta, n_other) retrieved docs by memory type."""
    n_cf = n_meta = n_other = 0
    for body in _doc_bodies(text):
        if CF_DOC_LEAD.search(body):
            n_cf += 1
        elif META_DOC_FIELDS.search(body):
            n_meta += 1
        else:
            n_other += 1
    return n_cf, n_meta, n_other


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

    turns, docs, queries, quoted_titles = [], [], [], []
    n_any_retrieval = n_answer = n_multi_answer = n_grounded = 0
    n_gt = n_gt_in_docs = n_gt_in_docs_ret = 0
    cf_docs = meta_docs = other_docs = n_touch_meta = n_split_mismatch = 0

    for rec in records:
        text = "".join(rec.get("predict") or [])
        t = len(TOOL_CALL.findall(text))
        d = len(DOC.findall(text))
        turns.append(t)
        docs.append(d)
        if t:
            n_any_retrieval += 1

        n_cf, n_meta, n_other = _memtype_counts(text)
        cf_docs += n_cf
        meta_docs += n_meta
        other_docs += n_other
        if n_meta:
            n_touch_meta += 1
        # Self-check: the position split must account for every `Doc N (Title:`
        # marker inside a bounded <tool_response>. A nonzero count here means the
        # doc parser drifted -- do not trust the memtype columns until it is 0.
        if n_cf + n_meta + n_other != len(DOC.findall("".join(_retrieved_spans(text)))):
            n_split_mismatch += 1
        for nq, nquoted in _query_counts(text):
            queries.append(nq)
            quoted_titles.append(nquoted)

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
        "quoted_titles_per_query_mean": mean(quoted_titles),
        "answer_rate": n_answer / n,
        "multi_answer_rate": n_multi_answer / n,
        "grounded_rate_of_answered": (n_grounded / n_answer) if n_answer else 0.0,
        # coverage — the retrieval-quality dev metric (see loop comment)
        "gt_in_docs_rate": (n_gt_in_docs / n_gt) if n_gt else 0.0,
        "gt_in_docs_of_retrieving": (n_gt_in_docs_ret / n_any_retrieval) if n_any_retrieval else 0.0,
        # memory type — which of RRCM's two memories the policy actually read
        "cf_docs": cf_docs,
        "meta_docs": meta_docs,
        "other_docs": other_docs,
        "meta_doc_share": meta_docs / (cf_docs + meta_docs + other_docs) if (cf_docs + meta_docs + other_docs) else 0.0,
        "meta_touch_rate": n_touch_meta / n,
        "doc_split_mismatch": n_split_mismatch,
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

    hdr = ("decode", "n", "retr%", "turns", "docs", "docs/ret", "q/call", "qt/q", "ans%", "multi%",
           "grnd%", "GTdocs%", "GTdocs%|ret", "META%", "METAdoc%")
    fmt = "%-46s %5s %7s %7s %7s %9s %7s %6s %7s %7s %7s %8s %11s %7s %9s"
    print(fmt % hdr)
    print(fmt % tuple("-" * len(h) for h in hdr))
    for r in rows:
        # Keep the last two path components: <step>/<decode_tag>.
        label = "/".join(r["file"].split("/")[-3:-1]) or r["file"]
        print(
            "%-46s %5d %6.1f%% %7.3f %7.2f %9.2f %7.2f %6.2f %6.1f%% %6.1f%% %6.1f%% %7.1f%% %10.1f%% %6.1f%% %8.1f%%"
            % (
                label[-46:],
                r["n"],
                100 * r["retrieval_rate"],
                r["turns_mean"],
                r["docs_mean"],
                r["docs_per_retrieval"],
                r["queries_per_call_mean"],
                r["quoted_titles_per_query_mean"],
                100 * r["answer_rate"],
                100 * r["multi_answer_rate"],
                100 * r["grounded_rate_of_answered"],
                100 * r["gt_in_docs_rate"],
                100 * r["gt_in_docs_of_retrieving"],
                100 * r["meta_touch_rate"],
                100 * r["meta_doc_share"],
            )
        )
        if r["doc_split_mismatch"]:
            print(f"    [warn] doc-split mismatch on {r['doc_split_mismatch']}/{r['n']} rollouts "
                  f"-- memtype columns are unreliable for this decode")

    print()
    print("retr%    = share of test prompts where the model issued >=1 <tool_call>.")
    print("docs/ret = docs returned per RETRIEVING rollout -- the positive control.")
    print("q/call   = query_list entries per tool call, parsed from the tool-call JSON.")
    print("           docs/ret == q/call * topk(3) EXACTLY -- verified call-by-call over")
    print("           v8 steps 200/400/600, zero exceptions. The service honours every")
    print("           query; a docs/ret 'shortfall' vs q/call is a metric bug, not a bug")
    print("           in retrieval. (See _query_counts: the pre-2026-08-07 form counted")
    print("           quote pairs and read q/call = n_queries + qt/q.)")
    print("qt/q     = quoted item titles the model pastes INSIDE its query text. This is")
    print("           the axis that actually moves: v8 goes 0.04 -> 8.43 over steps")
    print("           200->600 (the policy learns to quote the user's history verbatim)")
    print("           while q/call stays pinned at 1.0. GTdocs% does not move with it.")
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
    print("META%     = share of rollouts that retrieved >=1 ITEM-METADATA doc (the second")
    print("            of RRCM's two memories: Title/Price/SalesRank/Brand/Categories).")
    print("METAdoc%  = share of all retrieved docs that were metadata rather than CF history.")
    print("            THE FINDING (2026-08-08, Part 11): both collapse to ~0 over training in")
    print("            EVERY arm, including the unshaped outcome-only baseline --")
    print("               baseline-n8  META% 41.3 @200 -> 0.3 @450 -> 0.1 @500")
    print("               v8-n8        META%  3.8 @200 -> 0.0 @600")
    print("            RRCM's own ablation (paper Table 2, Amazon CDs & Vinyl) prices the")
    print("            memory the policy trains itself out of using at HR@5 0.0064 (w/o META)")
    print("            vs 0.0102 (full). Our converged baseline is 0.0080 -- behaviourally")
    print("            and numerically the w/o-META ablation. Every audit in this directory")
    print("            was memory-type-blind before these two columns existed.")
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
