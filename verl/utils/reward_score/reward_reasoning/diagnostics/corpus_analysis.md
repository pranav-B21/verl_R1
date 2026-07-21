# CF-Continuation Corpus — What It Is and Why

A plain-language walkthrough of the new corpus built under `verl_R1/cf_corpus/`.
For the exact commands and numbers see `verl_R1/cf_corpus/README.md`; this doc
explains the *idea* and the *reasoning*.

---

## 1. The problem, in one sentence

The model already knows how to search and how to pick the right answer — but the
**corpus has no document that connects a user's listening history to what comes
next in a way a test user can actually find.** So retrieval hands the model docs
that don't contain the answer, and it can't recommend what it never sees.

### How we know that (the three diagnostics)

We ruled out every other suspect first:

| Suspect | Test | Verdict |
|---|---|---|
| The **reward** is wrong | train-greedy HR@1 reached 0.39 | ❌ not the problem — the model learns fine |
| The **model** is too weak | same 0.39 on train | ❌ not the problem |
| The answer **isn't in the corpus** | coverage@∞ = 98.1%, oracle query @1 = 89% | ❌ the answer exists and is findable *in principle* |
| The answer **can't be reached from a history query** | history queries hit only ~26% even at top-100 | ✅ **this is the problem** |

Test HR@1 was stuck at 0.002–0.005 (basically noise). The gap between "answer
exists" (98%) and "answer reachable from a realistic query" (26%) is the whole
story. **The fix is not a better reward or a bigger model — it's restructuring
the corpus.**

---

## 2. Why the *old* corpus couldn't work

The old corpus had two kinds of documents:

1. **Per-user history docs** (72k):
   `"A user played: A, B, C, D, E"` — one per user, listing that user's items.
2. **Item metadata docs** (13k):
   `"Title: Kind of Blue, Price: ..., Brand: ..."` — one per item.

Here's the trap. During **training**, a train user's own future items sit inside
their *own* history doc, so the model could "cheat" by retrieving the user's own
record. That doesn't transfer: a **test** user is not in the corpus, so their
continuation lives only inside *other* users' history docs — and there's no way
to search for "the other users who behaved like me." A history query just
retrieves other people's raw item lists, which rarely contain *your* next item.

**Nothing in the old corpus says "people who did X tend to do Y next."** That
sentence is exactly what a recommender needs, and it wasn't there.

---

## 3. What the new corpus does

It turns raw behaviour into **aggregated, user-independent transition
statements** — one document per item:

```
Users who played "Kind of Blue" next played:
  "Bitches Brew" (41), "In a Silent Way" (17), "A Love Supreme" (9), ...
```

Read that as: *across all training users, whoever played "Kind of Blue" tended to
play these things soon after, this many times.*

Why this shape fixes everything the old corpus got wrong:

- **User-independent.** It's about the *item*, not a person. A test user who just
  played "Kind of Blue" can retrieve this doc — the transition knowledge is no
  longer trapped inside strangers' private histories.
- **Reachable from a realistic query.** The query "I just played Kind of Blue"
  embeds right next to a doc literally about "Users who played Kind of Blue."
- **Carries a frequency cue.** The `(41)`, `(17)` counts let the model *reason*
  about which continuation is most likely — turning selection into a real
  decision instead of a copy.
- **Short.** Capping the list keeps docs small enough that many fit the model's
  context budget.

The numbers `(counts)` are the difference between "dump every co-occurrence"
(which just reproduces global popularity and explodes the index) and "aggregate
with support" (a compact, rankable signal).

---

## 4. The code, script by script

All under `verl_R1/cf_corpus/`. Steps 1–4a run on the login node in seconds
(pure CPU); step 4b–5 need one GPU pass.

### `build_cf_corpus.py` — build the transition docs (steps 1–3)

1. **Reconstruct each train user's sequence.** The training file stores
   cumulative prefixes (each row = the previous row + one new item), so the
   script stitches consecutive rows back into one full chronological history per
   user. **Only training users** — test users never enter, so there's no
   leakage.
2. **Count transitions.** For every item `X` a user played, look at the items
   that came *within a window after it* (default: the next 10 plays — and since
   sequences average ~10 items, that's effectively "the rest of the sequence").
   Tally `X → Y` across all users.
3. **Prune and render.** Drop rare transitions (`min-support`, default seen ≥ 2
   times) and keep the top-N most frequent continuations per item
   (`max-cont`, default 30). Write one doc per item in the format above.

It also prints a built-in **sanity check**: of the 1000 test cases, how many have
their true next-item somewhere in a reachable transition doc. This is the
corpus's *intrinsic ceiling* — the best retrieval could ever do — measured
without touching a GPU.

> **Why single-item anchors?** We could key docs on pairs/triples ("users who
> played X *and* Y next played…"). We tried it: coverage collapsed from 0.158 to
> 0.026 because multi-item keys are far rarer. Single-item docs generalize best,
> so that's the default. (Knob: `--anchor-len`.)

> **Validation.** With pruning switched off, the script reproduces the
> independently-computed transition ceiling almost exactly (0.313 vs 0.325),
> which confirms the counting logic is correct.

### `assemble_corpus.py` — put the corpus together (step 4a)

Splits the old corpus into its metadata half and its history half, then lets you
concatenate the pieces you want and **renumber the document ids**. The renumber
matters: the retriever maps a search hit back to a document purely by line
number, so the corpus file's order must match the index's order exactly.

The recommended corpus is **CF docs + item-metadata docs** (we keep metadata
because it's what makes individual items findable). We also build a variant that
*also* includes the old history docs, so we can test whether they help or just
crowd out the new CF docs.

### `build_index.py` — make it searchable (step 4b, GPU)

Turns each document into a vector with the **exact same encoder the live
retriever uses** (the e5 model, "passage:" prefix, mean-pooling, normalization,
half-precision) and builds the FAISS search index. "Exact same" is the whole
point — if the offline encoding differed from the online one, our measurements
wouldn't predict real serving behaviour.

### `coverage_sweep.py` (extended) — did it work? (step 5, GPU)

The existing diagnostic, now with **history-tail queries**: instead of only
testing the model's own queries, it also tries "use the last 3 / last 5 items the
user played" as the query — because that's the natural query shape for these new
docs. It reports how often the true answer shows up in the top-K retrieved docs.
This is the **gate**: if this number isn't high enough, we fix the corpus builder,
**not** the model, and we don't waste GPU on retraining.

### `run_reindex_and_sweep.sh` — the GPU driver

Runs `build_index.py` then `coverage_sweep.py` for both candidate corpora, so the
whole GPU stage is one command.

---

## 5. The numbers (headline config)

Config: single-item anchor, window 10, min-support 2, max-cont 30.

- **12,607 CF docs**, ~168 tokens each (short enough to retrieve ~10 at once).
- **Intrinsic reachability** of test answers: **15.8%** (tail-5), vs the old
  corpus's ~12% *at top-100*. And it's tunable — raising the continuation cap
  pushes it toward the ~31% theoretical ceiling, trading against doc length.

### The pre-registered prediction (write it down before retraining)

```
predicted test HR@1  ≈  (how often retrieval surfaces the answer)  ×  (how often
                          the model then picks it, proven ~50% on train)
                     ≈  coverage@20  ×  0.5
```

That puts the expected ceiling at roughly **0.079–0.10 HR@1** — a **16–50× lift**
over today's 0.002–0.005. Committing to that number *before* the run is what makes
the result convincing: a trained model that then approaches it proves the
diagnosis, rather than us explaining the outcome after the fact.

---

## 6. The one-paragraph takeaway

We spent several reward iterations (v2–v6) landing in a noise band, then a cheap
pair of diagnostics showed the real bottleneck was two layers below the reward —
in how the benchmark's corpus was built. The old corpus stored *who did what*;
the new corpus stores *what tends to follow what*, in a user-independent,
searchable, frequency-weighted form. That single change is what makes a test
user's next item findable, and it's the prerequisite before any further reward
work can have something real to act on.
