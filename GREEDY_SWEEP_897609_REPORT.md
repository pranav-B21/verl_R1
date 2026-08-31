# Greedy held-out sweep: v8 shaped reward vs outcome-only baseline

**Job:** SLURM 897609 · **Date:** 2026-08-08 · **Decode regime:** greedy (`temperature=0, top_p=1, top_k=1`)
**Data:** 1000 held-out Amazon *CDs & Vinyl* prompts · **Arms:** `gpu-baseline-n8` (outcome-only, `USE_RTHINK=0`) vs `gpu-rthink-v8-n8` (v8 shaped reward)
**Checkpoints:** global_step 200 → 500, every 50 · **Decodes:** 16 (step 200 decoded twice per arm as a determinism control; 250–500 once each), all against one warm retriever.

---

## Summary

**The v8 shaped reward does not beat the outcome-only baseline.** Averaged over its 8 greedy
decodes the baseline reaches **HR@5 0.0055** against v8's **0.0044**; at the pre-registered
comparison point (step 200) the two are **identical, 0.0035 each**. The best single measurement in
the whole sweep belongs to the baseline (step 450, HR@5 0.0090). No checkpoint favours v8 on HR@5
by more than the decode-to-decode wobble.

**The paired within-prompt test agrees, and it has power.** Both arms decode the same 1000 prompts,
so they can be compared prompt-by-prompt (exact McNemar) instead of as two noisy averages:

- **Retrieval coverage** — whether the ground-truth item was returned at all — is a **tight null at
  every step**: largest gap 0.4 pp, all p ≥ 0.26. The arms disagree on only 7–44 prompts per 1000.
- **Exact answer match** (HR@1 proxy) is a **null at every step**. The one nominally significant
  point (step 200, p_boot = 0.034) rests on 4 discordant prompts and does not survive McNemar
  (p = 0.13).

**v8 changed decisively what it was built to change — and it bought no accuracy.**

- **Successor-rule following:** +9.5 pp at step 200, +11.2 pp at step 500 (p < 0.001).
- **Grounding the answer in retrieved docs:** +51.3 pp at step 200, +13.4 pp at step 500 (p < 0.001).

Both effects are large and unambiguous. Neither moves HR@1 or HR@5.

**Why: coverage is the binding constraint, and it does not move.** The ground-truth item appears in
the retrieved documents on **2.9–3.5% of prompts — in every arm, at every checkpoint**. Downstream
of that ceiling the selection step operates **at chance**: picks-GT-when-covered is 0–9% for the
baseline and 3–12% for v8, against a chance rate of ~7% and a pre-registered bar of 15%. Coverage ×
chance caps HR@1 near 0.001–0.002, which is exactly what the accuracy table shows.

### Instrument notes

- **Greedy decoding is not reproducible on this stack.** Two greedy decodes of the *same* frozen
  checkpoint differ in text on **64.9%** (baseline) / **44.0%** (v8) of prompts — sglang's
  continuous batching makes logits depend on batch composition, which `temperature=0, top_k=1`
  cannot control. The metric-level consequence is small (baseline @200: HR@5 0.0040 vs 0.0030) and
  the paired test is immune to it, but single-decode point estimates should not be over-read.
- **The tool-call JSON repair layer is working.** 305 malformed tool calls across 16,000 prompts,
  **284 auto-repaired, 21 unrecovered = 0.13%**. Not a factor in any comparison here.
- **Both arms train themselves out of the item-metadata memory** (baseline 38.5% of prompts at step
  200 → 0.1% at 500; v8 4.4% → 0.1%), consistent with the metadata-schema finding.
- **v8's recommendations are markedly less concentrated** (ORRatio@1 0.023 vs 0.044 — it falls back
  on its three favourite items about half as often), but that diversity does not convert into
  accuracy.
- **Retrieval is healthy in both arms** (Gate 0 passes): ~100% of prompts issue a retrieval call
  from step 250 onward, at ~3.0 docs per retrieving rollout = 1 query × top-k 3. Nothing here is a
  collapse artifact. The one outlier is the baseline at step 200, which still issues 2.3 queries
  per call and answers on only 84% of prompts.

**Read-through:** on this measurement the shaped reward buys the intended behaviour and no accuracy.
The constraint is upstream of the reward — coverage at ~3% and selection at chance.

---

## Table 1 — Held-out accuracy, all 16 greedy decodes

| Arm | Step | Decode | HR@1 | HR@5 | NDCG@5 | ORRatio@1 |
|---|---:|---|---:|---:|---:|---:|
| baseline-n8 | 200 | greedy1 | 0.0010 | 0.0040 | 0.00256 | 0.0682 |
| baseline-n8 | 200 | greedy2 | 0.0010 | 0.0030 | 0.00206 | 0.0702 |
| baseline-n8 | 250 | greedy1 | 0.0020 | 0.0060 | 0.00415 | 0.0488 |
| baseline-n8 | 300 | greedy1 | 0.0020 | 0.0030 | 0.00263 | 0.0214 |
| baseline-n8 | 350 | greedy1 | 0.0030 | 0.0070 | 0.00519 | 0.0234 |
| baseline-n8 | 400 | greedy1 | 0.0000 | 0.0050 | 0.00269 | 0.0356 |
| baseline-n8 | 450 | greedy1 | **0.0030** | **0.0090** | **0.00632** | 0.0376 |
| baseline-n8 | 500 | greedy1 | 0.0020 | 0.0070 | 0.00475 | 0.0498 |
| v8-n8 | 200 | greedy1 | 0.0040 | 0.0040 | 0.00400 | 0.0163 |
| v8-n8 | 200 | greedy2 | 0.0030 | 0.0030 | 0.00300 | 0.0163 |
| v8-n8 | 250 | greedy1 | 0.0020 | 0.0040 | 0.00313 | 0.0264 |
| v8-n8 | 300 | greedy1 | 0.0030 | 0.0060 | 0.00469 | 0.0264 |
| v8-n8 | 350 | greedy1 | 0.0010 | 0.0050 | 0.00339 | 0.0254 |
| v8-n8 | 400 | greedy1 | 0.0020 | 0.0040 | 0.00306 | 0.0244 |
| v8-n8 | 450 | greedy1 | 0.0020 | 0.0030 | 0.00263 | 0.0203 |
| v8-n8 | 500 | greedy1 | 0.0040 | 0.0060 | 0.00502 | 0.0254 |

**Per-arm means over all 8 decodes**

| Arm | HR@1 | HR@5 | NDCG@5 | ORRatio@1 | HR@5 range |
|---|---:|---:|---:|---:|---|
| baseline-n8 | 0.00175 | **0.00550** | 0.00379 | 0.0444 | 0.0030 – 0.0090 |
| v8-n8 | 0.00263 | 0.00438 | 0.00362 | 0.0226 | 0.0030 – 0.0060 |

`ORRatio@1` = share of all top-1 recommendation slots taken by the three most-recommended items
(output concentration; lower = more diverse). Greedy rows are **not** comparable to the historical
temperature-1.0 rows in `TEST_OUTPUT.md`.

## Table 2 — HR@5 by step, arms side by side

| Step | baseline-n8 | v8-n8 | Δ (v8 − baseline) |
|---:|---:|---:|---:|
| 200 | 0.0035 | 0.0035 | 0.0000 |
| 250 | 0.0060 | 0.0040 | −0.0020 |
| 300 | 0.0030 | 0.0060 | +0.0030 |
| 350 | 0.0070 | 0.0050 | −0.0020 |
| 400 | 0.0050 | 0.0040 | −0.0010 |
| 450 | **0.0090** | 0.0030 | −0.0060 |
| 500 | 0.0070 | 0.0060 | −0.0010 |

Step 200 is the mean of two decodes per arm; all other steps are single decodes. At 3–9 hits per
1000 prompts, per-step differences are within decode noise — Table 3 is the test that carries the
claim.

## Table 3 — Paired within-prompt test (exact McNemar, same 1000 prompts)

`audits/paired_arm_test.py`, greedy decodes, one decode per step. Positive Δ favours v8.
"discordant" = prompts where only v8 succeeded / only the baseline succeeded.

| Endpoint | Step | baseline | v8 | Δ | 95% CI | p (boot) | discordant v8/base | p (McNemar) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| **Coverage** (GT in retrieved docs) | 200 | 0.0320 | 0.0340 | +0.0020 | [−0.0110, +0.0150] | 0.828 | 23 / 21 | 0.880 |
| | 250 | 0.0340 | 0.0300 | −0.0040 | [−0.0140, +0.0060] | 0.490 | 11 / 15 | 0.556 |
| | 300 | 0.0320 | 0.0330 | +0.0010 | [−0.0040, +0.0060] | 0.850 | 4 / 3 | 1.000 |
| | 350 | 0.0340 | 0.0320 | −0.0020 | [−0.0080, +0.0040] | 0.634 | 4 / 6 | 0.752 |
| | 400 | 0.0330 | 0.0290 | −0.0040 | [−0.0100, +0.0020] | 0.261 | 3 / 7 | 0.343 |
| | 450 | 0.0330 | 0.0320 | −0.0010 | [−0.0080, +0.0060] | 0.893 | 6 / 7 | 1.000 |
| | 500 | 0.0350 | 0.0340 | −0.0010 | [−0.0090, +0.0070] | 0.912 | 7 / 8 | 1.000 |
| **Exact match** (answer == GT) | 200 | 0.0000 | 0.0040 | +0.0040 | [+0.0010, +0.0080] | 0.034 | 4 / 0 | 0.134 |
| | 250 | 0.0020 | 0.0020 | 0.0000 | [−0.0040, +0.0040] | 1.000 | 2 / 2 | 1.000 |
| | 300 | 0.0020 | 0.0030 | +0.0010 | [−0.0030, +0.0050] | 0.834 | 3 / 2 | 1.000 |
| | 350 | 0.0030 | 0.0010 | −0.0020 | [−0.0060, +0.0020] | 0.452 | 1 / 3 | 0.617 |
| | 400 | 0.0000 | 0.0020 | +0.0020 | [+0.0000, +0.0050] | 0.271 | 2 / 0 | 0.479 |
| | 450 | 0.0040 | 0.0020 | −0.0020 | [−0.0060, +0.0020] | 0.455 | 1 / 3 | 0.617 |
| | 500 | 0.0020 | 0.0040 | +0.0020 | [−0.0020, +0.0060] | 0.446 | 3 / 1 | 0.617 |
| **Successor following** (what v8 rewards) | 200 | 0.0300 | 0.1250 | **+0.0950** | [+0.0730, +0.1180] | **<0.001** | 117 / 22 | **<0.001** |
| | 250 | 0.0600 | 0.1580 | **+0.0980** | [+0.0740, +0.1230] | **<0.001** | 131 / 33 | **<0.001** |
| | 300 | 0.1580 | 0.1450 | −0.0130 | [−0.0350, +0.0090] | 0.265 | 55 / 68 | 0.279 |
| | 350 | 0.1720 | 0.1820 | +0.0100 | [−0.0130, +0.0330] | 0.414 | 73 / 63 | 0.440 |
| | 400 | 0.1460 | 0.1970 | **+0.0510** | [+0.0280, +0.0740] | **<0.001** | 98 / 47 | **<0.001** |
| | 450 | 0.1580 | 0.2200 | **+0.0620** | [+0.0370, +0.0870] | **<0.001** | 113 / 51 | **<0.001** |
| | 500 | 0.1620 | 0.2740 | **+0.1120** | [+0.0850, +0.1390] | **<0.001** | 157 / 45 | **<0.001** |
| **Grounded** (answer came from docs) | 200 | 0.2520 | 0.7650 | **+0.5130** | [+0.4760, +0.5490] | **<0.001** | 566 / 53 | **<0.001** |
| | 250 | 0.5060 | 0.8520 | **+0.3460** | [+0.3110, +0.3810] | **<0.001** | 390 / 44 | **<0.001** |
| | 300 | 0.8700 | 0.8820 | +0.0120 | [−0.0100, +0.0340] | 0.313 | 70 / 58 | 0.331 |
| | 350 | 0.8550 | 0.8760 | +0.0210 | [−0.0020, +0.0440] | 0.075 | 78 / 57 | 0.085 |
| | 400 | 0.7920 | 0.8890 | **+0.0970** | [+0.0720, +0.1220] | **<0.001** | 136 / 39 | **<0.001** |
| | 450 | 0.7860 | 0.9180 | **+0.1320** | [+0.1060, +0.1580] | **<0.001** | 161 / 29 | **<0.001** |
| | 500 | 0.7940 | 0.9280 | **+0.1340** | [+0.1080, +0.1590] | **<0.001** | 161 / 27 | **<0.001** |

## Table 4 — Decode behaviour (Gate 0: is the policy healthy?)

| Arm | Step | Decode | retr% | docs/ret | q/call | qt/q | ans% | GTdocs% (coverage) | META% |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-n8 | 200 | greedy1 | 95.0 | 6.96 | 2.27 | 0.00 | 84.0 | 3.2 | 38.5 |
| baseline-n8 | 200 | greedy2 | 94.7 | 7.03 | 2.23 | 0.00 | 83.3 | 3.5 | 38.0 |
| baseline-n8 | 250 | greedy1 | 98.3 | 4.20 | 1.41 | 0.00 | 96.5 | 3.4 | 12.8 |
| baseline-n8 | 300 | greedy1 | 100.0 | 3.07 | 1.02 | 0.19 | 99.9 | 3.2 | 1.4 |
| baseline-n8 | 350 | greedy1 | 100.0 | 3.08 | 1.03 | 3.22 | 99.7 | 3.4 | 0.6 |
| baseline-n8 | 400 | greedy1 | 99.9 | 3.12 | 1.03 | 0.49 | 99.0 | 3.3 | 0.4 |
| baseline-n8 | 450 | greedy1 | 100.0 | 3.04 | 1.01 | 0.13 | 99.6 | 3.3 | 0.2 |
| baseline-n8 | 500 | greedy1 | 100.0 | 3.04 | 1.01 | 0.09 | 99.8 | 3.5 | 0.1 |
| v8-n8 | 200 | greedy1 | 99.9 | 3.15 | 1.05 | 0.03 | 99.6 | 3.4 | 4.4 |
| v8-n8 | 200 | greedy2 | 100.0 | 3.16 | 1.05 | 0.03 | 99.7 | 3.4 | 4.3 |
| v8-n8 | 250 | greedy1 | 100.0 | 3.01 | 1.00 | 0.02 | 99.7 | 3.0 | 1.3 |
| v8-n8 | 300 | greedy1 | 100.0 | 3.00 | 1.00 | 0.09 | 99.9 | 3.3 | 1.1 |
| v8-n8 | 350 | greedy1 | 100.0 | 3.03 | 1.01 | 0.15 | 99.9 | 3.2 | 1.2 |
| v8-n8 | 400 | greedy1 | 100.0 | 3.09 | 1.03 | 2.24 | 99.7 | 2.9 | 0.9 |
| v8-n8 | 450 | greedy1 | 100.0 | 3.05 | 1.02 | 6.29 | 99.9 | 3.2 | 0.4 |
| v8-n8 | 500 | greedy1 | 100.0 | 3.02 | 1.01 | 8.33 | 100.0 | 3.4 | 0.1 |

`retr%` = prompts issuing ≥1 retrieval call · `docs/ret` = docs returned per retrieving rollout
(= `q/call` × top-k 3, exactly) · `q/call` = queries per tool call · `qt/q` = item titles the model
pastes verbatim inside its query text · `ans%` = prompts that produced an answer ·
`GTdocs%` = prompts where the GT appears in the retrieved docs (exact title match) ·
`META%` = prompts retrieving ≥1 item-metadata doc.

**Note the two flat columns.** `GTdocs%` sits at 2.9–3.5 everywhere — no arm and no amount of
training moves it. Meanwhile `qt/q` on the v8 arm climbs 0.03 → 8.33 (the policy learns to quote the
user's history verbatim into its query) with **no** effect on coverage: the query text changes a
great deal and retrieves the same thing.

## Table 5 — Selection funnel (Gate 1: given the GT was retrieved, is it picked?)

| Arm | Step | cov% | GT in docs | picks GT \| covered | chance | successor-follow% | repeat% | candidates seen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-n8 | 200* | 3.35 | 67/2000 | 0.0% (0/56) | 4.4% | 3.6 | 3.8 | 22.75 |
| baseline-n8 | 250 | 3.40 | 34/1000 | 6.1% (2/33) | 6.4% | 6.2 | 6.1 | 15.72 |
| baseline-n8 | 300 | 3.20 | 32/1000 | 6.2% (2/32) | 7.0% | 15.8 | 12.0 | 14.33 |
| baseline-n8 | 350 | 3.40 | 34/1000 | 8.8% (3/34) | 6.8% | 17.3 | 15.4 | 14.67 |
| baseline-n8 | 400 | 3.30 | 33/1000 | 0.0% (0/33) | 6.7% | 14.7 | 11.6 | 15.03 |
| baseline-n8 | 450 | 3.30 | 33/1000 | 9.1% (3/33) | 6.7% | 15.9 | 16.3 | 14.86 |
| baseline-n8 | 500 | 3.50 | 35/1000 | 5.7% (2/35) | 6.7% | 16.2 | 13.4 | 15.02 |
| v8-n8 | 200* | 3.40 | 68/2000 | 10.3% (7/68) | 7.0% | 12.2 | 8.6 | 14.27 |
| v8-n8 | 250 | 3.00 | 30/1000 | 6.7% (2/30) | 7.0% | 15.8 | 7.8 | 14.21 |
| v8-n8 | 300 | 3.30 | 33/1000 | 9.1% (3/33) | 7.0% | 14.5 | 11.7 | 14.24 |
| v8-n8 | 350 | 3.20 | 32/1000 | 3.1% (1/32) | 7.1% | 18.2 | 11.5 | 14.12 |
| v8-n8 | 400 | 2.90 | 29/1000 | 6.9% (2/29) | 6.9% | 19.8 | 9.4 | 14.46 |
| v8-n8 | 450 | 3.20 | 32/1000 | 6.2% (2/32) | 7.0% | 22.0 | 9.1 | 14.37 |
| v8-n8 | 500 | 3.40 | 34/1000 | 11.8% (4/34) | 7.0% | 27.4 | 9.5 | 14.27 |

\* pooled over the 2 determinism-probe decodes. `chance` = 1 / candidates seen. The pre-registered
pass bar for the selection branch is **15%**; no arm at any step reaches it, and every cell is
within sampling error of chance on 29–68 winnable cases.

## Table 6 — Controls

| Control | baseline-n8 | v8-n8 |
|---|---|---|
| Greedy determinism @200 (prompts differing across 2 decodes) | 649/1000 (64.9%) | 440/1000 (44.0%) |
| HR@5 spread across those 2 decodes | 0.0040 vs 0.0030 | 0.0040 vs 0.0030 |
| Malformed tool-call JSON (8 decodes each) | 253 | 52 |
| …auto-repaired | 240 | 44 |
| …unrecovered | 13 (0.16% of prompts) | 8 (0.10% of prompts) |
| Retrieval service health | pass (docs/ret ≈ 3–7 throughout) | pass (docs/ret ≈ 3 throughout) |

---

*Source: `verl_R1/output_test_greedy_sweep_897609.log`;
`verl/utils/reward_score/reward_retrieval/audits/paired_arm_test_greedy_2026-08-08.json`;
per-decode `greedy_decode_behavior.json` / `greedy_decode_selection.json` under
`outputs/eval/<arm>/global_step_<n>/`.*
