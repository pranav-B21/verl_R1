# SPDX-License-Identifier: Apache-2.0
"""Reproduce corpus-build bugs 2, 3 and 4 from `data/data_process.ipynb`, with receipts.

Written to answer a reviewer question -- "how was '5.3% of the catalog carries another
record's Brand/Price' determined?" -- with a rerunnable derivation instead of a claim.

Bug 1 (the 2014/2018 schema key mismatch that emptied SalesRank/Categories) is NOT
covered here: it is already fixed, shipped as `corpora_v2.jsonl`, and measured -- see
`PREREG_metafix_2026-08-24.md`.

  BUG 2  no html.unescape() before matching.  Raw 2018 titles are HTML-escaped
         ("808s &amp; Heartbreak"), so escaped titles miss the exact-match dict and
         fall through to the fuzzy branch.  Measured here as: of the titles that took the fuzzy branch, how many exact-match against an UNESCAPED index.
         the fuzzy branch, how many exact-match against an UNESCAPED index (the
         escaping lives in the DUMP keys: it holds "808s &amp; Heartbreak" while the
         harvested corpus title is the plain "808s & Heartbreak").

  BUG 3  the fuzzy fallback assigns a DIFFERENT album's metadata.  Notebook cell 2:

             for meta_title, item in title_to_meta.items():
                 if title in meta_title or meta_title in title:
                     meta_item = item
                     break

         First hit in dict-insertion order over ~339k raw titles wins.  The dump
         contains degenerate titles ('A', 'H', 'Live', 'Greatest Hits', and 39 records
         whose title is the EMPTY STRING -- and '' in anything is always True), so the
         `meta_title in title` half fires on garbage almost every time.  This is a
         plain Python substring test over raw strings at BUILD time.  No embedding
         model is involved: e5 encodes the finished doc text afterwards, for retrieval.
         A robust encoder cannot repair a doc whose Brand field already names the
         wrong artist -- it encodes the wrong text faithfully.

         Wrongness (not just oddness) is established by recovering the intended record
         via html.unescape and showing the assigned Brand/Price differ from it.

  BUG 4  quote-pairing desync.  Cell 1 harvests titles with re.findall(r'"([^"]*)"'),
         which mis-pairs quotes on titles that contain embedded quotes.  Reported as
         the set relationship between the corpus title universe and eval's name2id.

Usage (login node, no GPU; ~2 min, dominated by the 666MB dump parse):
    python3 corpus_build_audit.py
    python3 corpus_build_audit.py --json corpus_build_audit_<date>.json --examples 12
"""

import argparse
import collections
import html
import json
import os
import sys

AP = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--data-dir", default="data/amazon_data")
AP.add_argument("--json", help="write the full result dict here")
AP.add_argument("--examples", type=int, default=8)
args = AP.parse_args()

D = args.data_dir
RAW_META = os.path.join(D, "meta_CDs_and_Vinyl.json")
MATCHED = os.path.join(D, "matched_titles_metadata.json")
CORPUS = os.path.join(D, "corpora.jsonl")
NAME2ID = os.path.join(D, "CDs_and_Vinyl/name2id.json")

for p in (RAW_META, MATCHED):
    if not os.path.exists(p):
        sys.exit(f"missing {p} (run from verl_R1/)")

out = {}


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ build the index
# Rebuilt EXACTLY as notebook cell 2 does it: JSON-lines, later duplicate titles
# overwrite earlier ones, and Python dicts preserve insertion order -- which is what
# makes `break` deterministic and reproducible here.
print(f"[1/4] parsing {RAW_META} ...", flush=True)
title_to_meta = {}
unesc = {}
n_records = 0
with open(RAW_META, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_records += 1
        if "title" in item:
            title_to_meta[item["title"]] = item
            # BUG 2 repair, built alongside for comparison: index by the UNESCAPED key.
            # First writer wins here so a later escaped duplicate cannot shadow a
            # record that already matched cleanly.
            unesc.setdefault(html.unescape(item["title"]), item)
order = list(title_to_meta.keys())
print(f"      {n_records} records -> {len(order)} distinct titles")
out["raw_records"] = n_records
out["raw_distinct_titles"] = len(order)

empties = sum(1 for t in order if t == "")
short = sum(1 for t in order if len(t) <= 3)
print(f"      degenerate keys: {empties} empty-string titles, {short} titles of length <= 3")
out["raw_empty_titles"] = empties
out["raw_len_le_3"] = short

matched = json.load(open(MATCHED, encoding="utf-8"))
by_type = collections.Counter(r.get("match_type") for r in matched)
fuzzy = [r for r in matched if r.get("match_type") == "fuzzy"]
print(f"      matched_titles_metadata.json: {len(matched)} records, {dict(by_type)}")
out["catalog_size"] = len(matched)
out["match_type_census"] = dict(by_type)
out["fuzzy_pct"] = 100.0 * len(fuzzy) / len(matched)

# ------------------------------------------------------------------- BUG 2 + BUG 3
rule("BUG 3 -- replay the fuzzy fallback exactly as the notebook runs it")
print("      for each fuzzy title, find which raw title wins the `break` and via which branch\n")

branch = collections.Counter()
winners = collections.Counter()
win_len = []
examples = []
recovered = 0
wrong_brand = 0
wrong_price = 0
changed_any = 0

for rec in fuzzy:
    title = rec["title"]
    winner = None
    which = None
    for meta_title in order:
        if title in meta_title:
            winner, which = meta_title, "title in meta_title"
            break
        if meta_title in title:
            winner, which = meta_title, "meta_title in title"
            break
    if winner is None:
        branch["no hit (unreachable)"] += 1
        continue
    branch[which] += 1
    winners[winner] += 1
    win_len.append(len(winner))

    # Establish WRONGNESS: does the intended record exist, and does it differ?
    truth = unesc.get(title)
    if truth is None:
        continue
    recovered += 1
    got = title_to_meta[winner]
    b_got, b_true = got.get("brand"), truth.get("brand")
    p_got, p_true = got.get("price"), truth.get("price")
    if b_got != b_true:
        wrong_brand += 1
    if p_got != p_true:
        wrong_price += 1
    if b_got != b_true or p_got != p_true:
        changed_any += 1
        if len(examples) < args.examples:
            examples.append(
                {"corpus_title": title, "won_by": winner, "branch": which,
                 "assigned_brand": b_got, "correct_brand": b_true,
                 "assigned_price": p_got, "correct_price": p_true}
            )

nf = len(fuzzy)
print(f"  fuzzy titles replayed        : {nf}")
for k, v in branch.most_common():
    print(f"    via `{k}`".ljust(38) + f": {v}  ({100*v/nf:.1f}%)")
print(f"  winning raw title <= 12 chars: {sum(1 for L in win_len if L <= 12)}/{len(win_len)}"
      f" = {100*sum(1 for L in win_len if L<=12)/max(len(win_len),1):.1f}%")
print("\n  most frequent 'winners' (the strings that swallowed the match):")
for w, c in winners.most_common(10):
    print(f"    {c:>4}x  {w!r}")

print(f"\n  BUG 2: exact-match against an UNESCAPED index: {recovered}/{nf}"
      f" = {100*recovered/nf:.1f}%  -- these were never ambiguous, just escaped")
print(f"  BUG 3: of those {recovered} recoverable, the assigned record differs on")
print(f"           brand : {wrong_brand} ({100*wrong_brand/max(recovered,1):.1f}%)")
print(f"           price : {wrong_price} ({100*wrong_price/max(recovered,1):.1f}%)")
print(f"           either: {changed_any} = {100*changed_any/len(matched):.2f}% of the {len(matched)}-item catalog")

print("\n  examples (assigned vs. the record html.unescape recovers):")
for e in examples:
    print(f"    {e['corpus_title']!r}")
    print(f"       won by {e['won_by']!r}  via `{e['branch']}`")
    print(f"       brand: {e['assigned_brand']!r}  ->  should be {e['correct_brand']!r}")

out["bug3"] = {"n_fuzzy": nf, "branch": dict(branch),
               "top_winners": winners.most_common(15),
               "recoverable_via_unescaped_index": recovered,
               "wrong_brand": wrong_brand, "wrong_price": wrong_price,
               "wrong_either": changed_any,
               "wrong_either_pct_of_catalog": 100.0 * changed_any / len(matched),
               "examples": examples}
out["bug2"] = {"fuzzy_explained_by_html_escape_pct": 100.0 * recovered / nf}

# ------------------------------------------------------------------------- BUG 4
rule("BUG 4 -- quote-pairing desync: corpus title universe vs eval name2id")
if not (os.path.exists(CORPUS) and os.path.exists(NAME2ID)):
    print("  (corpora.jsonl or name2id.json missing -- skipped)")
else:
    corpus_titles = set()
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line).get("contents", "")
            if t.startswith("Title: "):
                corpus_titles.add(t[len("Title: "):].split(", Price:")[0])
    name2id = json.load(open(NAME2ID, encoding="utf-8"))
    ev = set(name2id)
    only_corpus = corpus_titles - ev
    only_eval = ev - corpus_titles
    print(f"  corpus metadata titles : {len(corpus_titles)}")
    print(f"  eval name2id keys      : {len(ev)}")
    print(f"  intersection           : {len(corpus_titles & ev)}")
    print(f"  corpus-only (junk keys, harmless distractors) : {len(only_corpus)}")
    print(f"  eval-only  (real items whose metadata doc is filed under a mangled key,")
    print(f"              so exact-title coverage can never score them)    : {len(only_eval)}")
    print(f"              = {100*len(only_eval)/len(ev):.2f}% of the eval catalog")
    print("\n  corpus-only examples:", sorted(only_corpus)[:6])
    print("  eval-only examples  :", sorted(only_eval)[:6])
    out["bug4"] = {"corpus_titles": len(corpus_titles), "eval_name2id": len(ev),
                   "intersection": len(corpus_titles & ev),
                   "corpus_only": len(only_corpus), "eval_only": len(only_eval),
                   "eval_only_pct": 100.0 * len(only_eval) / len(ev),
                   "corpus_only_examples": sorted(only_corpus)[:20],
                   "eval_only_examples": sorted(only_eval)[:20]}

if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
    print(f"\nwrote {args.json}")
