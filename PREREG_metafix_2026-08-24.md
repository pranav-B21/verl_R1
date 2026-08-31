# Pre-registration — repaired-metadata corpus, 2 arms

Written **2026-08-24, before either run was launched and before any result existed.**
Do not edit the gates below after seeing data. Record outcomes in a separate section.

## Change under test

Exactly one variable vs the historical runs: the metadata half of the corpus.
`SalesRank` 0.0% → 99.5%, `Categories` 0.0% → 100.0% (genre), from fixing a 2014-vs-2018
Amazon schema key mismatch (`salesRank`/`categories` read against a dump keyed
`rank`/`category`; both `.get()` calls missed silently).

Verified single-variable: across all 13,111 records `title`/`price`/`brand`/`match_type`
differ in **0**; history docs 0–72,190 byte-identical; index rows reproduce
`emb_e5.memmap` at cosine ≥0.99999889 on the unchanged half.

Deliberately NOT fixed in this arm (would move Price/Brand and break attribution):
`html.unescape` on titles, and the first-substring-hit fuzzy fallback that mis-assigns
metadata to 5.3% of the catalog.

## Arms

| arm | config | comparator (old corpus) |
|---|---|---|
| `…-gpu-baseline-n8-metafix` | `USE_RTHINK=0` | `…-gpu-baseline-n8` |
| `…-gpu-rthink-v8-n8-metafix` | `RTHINK_MODE=v8`, `RTHINK_W_GROUND=0.40` | `…-gpu-rthink-v8-n8` |

`w_ground=0.40` is pinned deliberately: it is off-spec vs v8's code default of 0.1, but it
is what the historical v8 ran, so pinning it keeps corpus the only variable. v8-at-0.1 has
never been run and is a separate future arm.

## Primary gate — behavioural, checked at step 300

**Metric:** `META%` from `audits/decode_behavior.py` (share of rollouts retrieving ≥1
item-metadata doc). Chosen over HR@5 deliberately: the historical effect is ~42 points
(42.67% @200 → 0.07% @500), whereas HR@5 is ~8 hits/1000 and has returned a tight null or
been underpowered in every prior comparison in this project.

* **KILL** — if both arms have `META%` < 5% at step 300, the repair did not make the
  memory worth using. Stop both runs. Do not continue to 500+ hoping HR moves.
* **PROCEED** — if either arm holds `META%` ≥ 20% at step 300, the policy is now choosing
  to use the memory. Continue training long (target 900–1300 steps, past the 500–600 where
  every prior run stopped) and only then read HR.
* **AMBIGUOUS** — 5–20%: extend to step 400 and re-read this gate once. No further extensions.

Falsifiable prediction being tested: *given metadata docs that carry real genre, an
outcome-only policy will no longer extinguish metadata retrieval.* If it extinguishes
anyway, the repair is refuted and the negative result stands as the deliverable.

## Secondary readouts (NOT gates — reported, never used to justify continuing)

1. HR@1 / HR@5 / NDCG@5 at matched steps 200/250/300/450/500, greedy
   (`DECODE_TEMPERATURE=0`), paired within prompt via `audits/paired_arm_test.py`.
2. `audits/memtype_payoff.py` paired-within-prompt metadata payoff. Prior on the broken
   corpus: HR@5 −0.00034 [−0.00339, +0.00245]. A repaired corpus should move this positive
   if the mechanism is real.
3. Coverage — expected **unchanged**. Metadata docs carry no candidate items, so they
   cannot touch the ~3.1% ceiling. A coverage move would indicate an instrument bug, not a win.

## Interpretation rules, fixed in advance

* Both arms move same direction ⇒ corpus effect, replicated across two independent runs.
* Only v8 moves ⇒ arm×corpus interaction (shaping pays only once the memory is real).
* Exactly one arm moves and it is the baseline ⇒ **not separable from training-run
  variance.** No within-corpus replicate exists anywhere in this project, so this outcome
  is NOT reportable as a corpus effect without a fresh-seed control run.
* Any unpaired cross-arm comparison is invalid. Three prior instrument-bias findings
  (coverage window, `q/call`, unpaired metadata split) all flipped under pairing.

## Known caveats accepted at launch

* 146 metadata docs (1.1%) still lose the new `Categories` to 256-token truncation because
  upstream HTML-scrape junk fills their `price` field first. Pre-existing, identical in
  both corpus versions.
* HR gain is not predicted with confidence: metadata docs carry no candidate items, so any
  payoff must route through selection, which `selection_capability_probe.py` closed
  (pooled graded pct 0.504 [0.493, 0.515]). Genre is a content signal that probe never had
  available — that is the reason to run this, and the reason to gate it cheaply.

## Outcome — fill in after the fact, do not edit above

### Result — filled in 2026-08-29, gates applied exactly as written above

### Run health (both arms clean)

Both jobs trained **from scratch** on the repaired corpus (`resume_mode=auto` →
`Training from scratch`), `[pre-flight] batch geometry OK: 56 x n=8 = 448 = 8 mini-batches of 56`,
and ended on SLURM **wall-time, not a crash**: zero hits for CUDA error 803, `pidfd_getfd`,
OOM/SIGKILL, Lustre Errno 108, or any traceback in either log.

| arm | job | steps reached | min/step | reward wiring seen in log |
|---|---|---|---|---|
| `…-baseline-n8-metafix` | 934860 | **425** | 6.1 | 0 `[Rthink-*]` prints (correct for `USE_RTHINK=0`) |
| `…-rthink-v8-n8-metafix` | 934861 | **625** | 4.6 | 4,617 `[Rthink-v8]` prints, `w_sel=0.5 w_grnd=0.4 w_rep=0.3 w_cov=0.4` — the pinned off-spec `w_ground=0.40`, as intended |

The baseline is 33% slower per step because it keeps a second query alive longer (see below), so
the arms are not step-matched at wall-clock end.

### The repair is live — verified at four levels, not assumed

1. `corpora_v2.jsonl`: SalesRank non-empty **13,041/13,111 = 99.5%**, Categories **13,111/13,111 =
   100.0%** (was 0.0% / 0.0% in `corpora.jsonl`). Doc count identical, 85,302.
2. Both training logs and both eval-sweep logs print
   `[retrieval_launch] serving data/amazon_data/corpora_v2.jsonl (85302 docs) + …/e5_Flat_v2.index`,
   and the corpus↔index row-count guard passed.
3. In the decodes themselves, the metadata docs the policy actually read carry real fields —
   baseline-metafix @200: **2,383/2,393 (99.6%) SalesRank, 2,393/2,393 (100%) Categories**.
   Same files from the broken-corpus control: **0/1,788 (0.0%) and 0/1,788 (0.0%)**.
4. Sample doc as served: `Title: Mirror Ball, Price: $10.98, SalesRank: 31507, Brand: Neil Young,
   Categories: CDs & Vinyl, Classic Rock, Album-Oriented Rock (AOR)`.

### PRIMARY GATE — META% at step 300  →  **KILL**

`audits/decode_behavior.py`, greedy, 3 decodes/step:

| step | baseline broken | baseline **repaired** | v8 broken | v8 **repaired** |
|---:|---:|---:|---:|---:|
| 200 | 38.2% | **52.1%** | 4.3% | **10.6%** |
| 250 | 12.8% | **20.8%** | 1.3% | **5.2%** |
| **300** | 1.4% | **2.9%** | 1.1% | **2.7%** |
| 350 | 0.6% | 1.0% | 1.2% | 2.3% |
| 400 | 0.4% | 0.6% | 0.9% | 1.5% |
| 450–600 | 0.2 → 0.1% | — | 0.4 → 0.0% | 0.6 → 0.0% |

Both arms are **below 5% at step 300**. Per the gate as written: **KILL. Stop both runs. Do not
continue to 500+ hoping HR moves.** Neither run was extended.

The falsifiable prediction — *given metadata docs that carry real genre, an outcome-only policy will
no longer extinguish metadata retrieval* — is **refuted**. It extinguishes anyway.

The repair did have a real, replicated behavioural effect, just not a decisive one. Both arms move
the **same direction** at every early step (baseline +13.9 pp @200, +8.0 pp @250; v8 +6.3 pp @200,
+3.9 pp @250), which under the interpretation rule above is a **corpus effect replicated across two
independent runs** — not run-to-run variance. METAdoc% moves with it (27.8% → 37.8% @200). The
policy does find the repaired memory more attractive. It just still drops it, on the same schedule.

**Mechanism (not pre-registered; read as a hypothesis).** META% extinction is a side effect of
**query-count collapse**, not an independent verdict on metadata. Effective queries per call
(`docs/ret ÷ 3`, the parser-independent form) track META% across all four arms:

| | 200 | 250 | 300 | 350+ |
|---|---:|---:|---:|---:|
| baseline repaired: docs/ret | 6.88 | 4.47 | 3.05 | 3.07–3.10 |
| baseline repaired: META% | 52.1 | 20.8 | 2.9 | ≤1.0 |
| baseline broken: docs/ret | 7.34 | 4.21 | 3.06 | 3.03–3.09 |
| baseline broken: META% | 38.2 | 12.8 | 1.4 | ≤0.6 |

Once `docs/ret` reaches 3.0 — one query — META% is ~0–3% in every arm and every corpus version.
v8 was already at one query by step 200, which is why its META% started low in both corpus
versions. The metadata memory dies **with the second query**, and the repair did nothing to keep
that second query alive.

### SECONDARY READOUTS (reported, not used to justify anything)

**1. Accuracy — null.** Greedy, paired within prompt via `audits/paired_arm_test.py`, corpus as the
treatment, 1 matched decode per arm per step:

* Coverage: null at every step in the baseline arm (all p ≥ 0.27). v8 arm null except @400
  (+0.70 pp, McNemar p=0.046) and @450 (+0.40 pp, p=0.13) — 2 of 12 cells nominally moving is what
  multiple comparisons predicts, and the gate above pre-declared a coverage move to be an instrument
  signal, not a win. Instrument reads sane.
* Exact match (HR@1 proxy): null at every step in both arms. The largest cell (baseline @200,
  +0.40 pp, p_boot 0.035) rests on **4 discordant prompts** and dies under McNemar (p=0.134).

HR@5 over the sweep, greedy: baseline repaired **0.0063 / 0.0060 / 0.0047 / 0.0050 / 0.0060**
(steps 200–400, mean 0.0056) against the broken-corpus arm's 8-decode mean of **0.0055**. v8
repaired 0.0013 → 0.0057 over steps 200–600 (mean 0.0041) against broken v8's **0.0044**. Both
corpus versions land in the same place. Best single repaired-corpus measurement is **0.0063**
against the **0.0102** target.

**2. Metadata payoff — NOT ANSWERABLE from these decodes.** `audits/memtype_payoff.py` pairs within
prompt, so it needs prompts where some rollouts read metadata and some did not. Greedy decoding plus
only 3 decodes/step leaves almost no such discordance: n=136 paired prompts at baseline@200, n=36 at
baseline@250, **n=5** at v8@200 — against n=960 for the 9-decode broken-corpus prior. The step-200
figure (HR@5 −0.00368, CI [−0.01103, +0.00000]) is ≈1 event and must not be read as a negative
result; the other two are all-zero for lack of a sample. **No positive payoff was detected, and none
could have been.** Getting a real number here needs ~9 decodes at a step where META% is high, i.e.
steps 200–250 only. JSON written to `audits/memtype_payoff_metafix_*_2026-08-29.json`.

**3. Coverage unchanged — confirmed**, as predicted. GTdocs% sits at 2.6–3.6% in both corpus
versions at every step. The ~3.1% ceiling is untouched, exactly as expected from metadata docs that
carry no candidate items.

### What this does and does not settle

Settled: **repairing SalesRank/Categories does not rescue the item-metadata memory, and does not
move held-out accuracy.** Bug #1 of the four corpus bugs is fixed, verified live, and priced at
approximately zero. The `w/o META` reading of our baseline is no longer available as an excuse —
we now have a run *with* a working META memory and it lands in the same place.

Not settled, and out of scope for this arm by construction: `html.unescape` (bug 2), the
first-substring-hit fuzzy fallback that mis-assigns Brand/Price (bug 3), and the quote-pairing
desync between the corpus and the eval catalog (bug 4).

**Bugs 3 and 4 re-derived and re-sized 2026-08-29** (`audits/corpus_build_audit.py`, in response to
a reviewer asking how the figure was obtained). Both of my earlier numbers were too big:

* **Bug 3 is 4.35%, not 5.3%.** 696 titles (5.31%) take the fuzzy branch; of those, 570 (81.9%) are
  recoverable against an html-unescaped index and **all 570 carry a different record's Brand or
  Price** (567 brand, 550 price). The other 126 are unrecoverable, so they cannot be *shown* wrong.
  5.31% is the fuzzy rate; **4.35% is the demonstrably-wrong rate**. 691 of the 696 fire through the
  `meta_title in title` half of the `or` — the winning raw title is ≤12 chars in 90.7% of cases
  (`'A'` 70×, `'Greatest Hits'` 42×, `''` 39×, `'E'` 24×). Not bad luck; structural.
* **Bug 2's direction was stated backwards in earlier notes.** The HTML escaping is in the *dump*
  keys (`808s &amp; Heartbreak`), not the harvested corpus titles (`808s & Heartbreak`), so the
  repair is `html.unescape()` on the index keys. Applying it to the corpus title recovers 0/696.
* **Bug 4 is 4 items, not 104.** Of the 104 eval-only keys, **100 are upstream HTML-scrape junk**
  (`… ,Capitol Records,Dance & DJ - General" />`) that is contamination in the frozen source, not a
  build bug. Only **4** are real albums, and all four contain embedded double quotes — exactly the
  desync mechanism: `Evelyn "Champagne" King`, `Marc Anthony "El Cantante"`, `The "Chirping"
  Crickets`, `The Secret Of Movin' On`. That is **0.03% of the eval catalog**. The 133 corpus-only
  keys are harmless extra distractor docs.

Both repairs already exist as opt-in flags on `data/regenerate_metadata.py` (`--unescape`,
`--scored-fuzzy`); neither was enabled here, by design, to keep this arm single-variable.
Those move Price/Brand and would have broken single-variable attribution here. Also unaddressed: the
upstream HTML-scrape contamination in the frozen source split (~9–10% of records), which is not ours
to fix.

Not tested: whether *forcing* a second, metadata-directed query would pay. The mechanism above says
that is the only remaining lever on this memory, and the payoff measurement that would justify
building it is currently underpowered rather than negative.
