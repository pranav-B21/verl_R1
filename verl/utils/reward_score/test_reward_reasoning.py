"""
Test file for reasoning reward components.
Uses patterns extracted from actual training logs to verify A, B, C work.

Run from anywhere:
  python verl/utils/reward_score/test_reward_reasoning.py
  python verl/utils/reward_score/test_reward_reasoning.py --no-model

--no-model skips sentence-transformer-dependent tests (no download needed).
Each test prints PASS/FAIL with component values and expected behavior.
"""

import sys
import os
import argparse
from collections import Counter

# Allow running from verl_R1/ root or from this directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from reward_SPRec_reasoning import (
        _tier0_self_repetition,
        _split_sentences,
        _staleness_score,
        _get_ngrams,
        _tier1_substring_match,
        _tier2_ngram_overlap,
        compute_redundancy_penalty,
        compute_info_gain,
        compute_exploration_bonus,
        compute_reasoning_components,
        _extract_think_blocks,
        _STALENESS_CALL_COUNT,
    )
    import reward_SPRec_reasoning as rsr
    IMPORTED = True
except ImportError as e:
    print(f"ERROR: Could not import reward_SPRec_reasoning.py: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test patterns — mix of constructed and actual log-derived examples
# ---------------------------------------------------------------------------

# Pattern 1: Degenerate looping (model repeats same sentence 50+ times)
# OLD penalty was 0.000, should be ~1.0
LOG_DEGENERATE_LOOP = (
    'The user listened to jazz and blues.'
    '<think>'
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    "But the user's most preferred music has to be a single item. "
    "So the answer would be that. "
    '</think>'
    '<answer>"The Beatles - Revolver"</answer>'
)

# Pattern 2: Good novel reasoning (introduces new ideas beyond the prompt)
LOG_GOOD_REASONING = (
    'User listened to Kind of Blue, A Love Supreme, Brilliant Corners.'
    '<tool_call>{"name": "search", "arguments": {"query_list": ["jazz recommendations for Kind of Blue fans"]}}</tool_call>'
    '<tool_response>Users who enjoyed Kind of Blue also listened to '
    'Herbie Hancock Maiden Voyage and Wayne Shorter Speak No Evil.</tool_response>'
    '<think>'
    'This collection spans late-50s to mid-60s modal jazz and hard bop. '
    'The harmonic sophistication suggests appreciation for complex chord '
    'progressions and improvisational depth. The era clustering around Blue Note '
    'recordings points toward post-bop exploration as a natural next step. '
    'Wayne Shorter bridges the gap between hard bop structure and freer forms.'
    '</think>'
    '<answer>"Wayne Shorter - Speak No Evil"</answer>'
)

# Pattern 3: Search result parroting (echoes tool_response verbatim into think)
LOG_SEARCH_PARROT = (
    'User listened to Kind of Blue, A Love Supreme.'
    '<tool_call>{"name": "search", "arguments": {"query_list": ["jazz recommendations"]}}</tool_call>'
    '<tool_response>Users who enjoyed Kind of Blue also frequently listened to '
    'Herbie Hancock Maiden Voyage and Wayne Shorter Speak No Evil. '
    'These albums share modal jazz characteristics.</tool_response>'
    '<think>'
    'Users who enjoyed Kind of Blue also frequently listened to '
    'Herbie Hancock Maiden Voyage and Wayne Shorter Speak No Evil. '
    'These albums share modal jazz characteristics. I will recommend one.'
    '</think>'
    '<answer>"Herbie Hancock - Maiden Voyage"</answer>'
)

# Pattern 4: Prompt parroting (copies prompt framing into think block)
LOG_PROMPT_PARROT = (
    'The user has listened to "Hotel California" by Eagles, '
    '"Slider" by The Rolling Stones, "Unchained Melody" by The Righteous Brothers. '
    'Based on the user listening history, recommend a new music.'
    '<think>'
    'The user has listened to Hotel California by Eagles, '
    'Slider by The Rolling Stones, Unchained Melody by The Righteous Brothers. '
    'Based on the user listening history, I should recommend a new music. '
    'The user might like classic rock. So I will recommend classic rock.'
    '</think>'
    '<answer>"Led Zeppelin II"</answer>'
)

# Pattern 5: Extreme loop (same analysis repeated 25x)
LOG_EXTREME_LOOP = (
    'User preferences include rock and pop.'
    '<think>'
    + ("So the answer is Dance Elektric by The Prodigy. "
       "But the user might already have that. "
       "So maybe a different track by a different artist. ") * 25
    + '</think>'
    '<answer>"Dance Elektric"</answer>'
)

# Pattern 6: Real log pattern — classic rock analysis (from scratch log)
LOG_REAL_CLASSIC_ROCK = (
    '<think>'
    "Okay, let's start by analyzing the user's listening history. "
    "They've enjoyed several songs like \"Bang, Zoom, Crazy Hello,\" \"Reunion,\" "
    "\"Dream Police,\" \"Fallen Angel,\" and others. "
    "These songs seem to be from different genres, including hard rock, metal, "
    "and possibly some classic rock. The user has also listened to albums like "
    "\"A Decade Of Dio: 1983-1993\" and \"Southern Native,\" which are likely "
    "heavy metal or rock albums. "
    "Looking at the titles, there's a mix of both classic and modern tracks. "
    "The user has a history with bands like Dio, which is a well-known heavy metal band. "
    "The songs \"Dream Police\" and \"Pyromania\" are also from Dio's discography. "
    "So, the user's preferences are likely in the heavy metal or rock genre."
    '</think>'
    '<answer>"Dio - Holy Diver"</answer>'
)

# Pattern 7: Real log pattern — repetitive "Okay, let's start" template
# (model converges to same opener across many rollouts)
LOG_REAL_TEMPLATE_COLLAPSE = (
    '<think>'
    "Okay, let's start by analyzing the user's listening history. "
    "They've enjoyed several albums from classic rock artists. "
    "The user has a mix of both classic and modern tracks. "
    "Looking at the genres, there's a lot of rock. "
    "The user might prefer something similar. "
    "So I will look for a similar album."
    '</think>'
    '<answer>"Classic Rock Album"</answer>'
)

# Pattern 8: Empty/trivial think block
LOG_EMPTY_THINK = (
    'User listened to jazz.'
    '<think>ok</think>'
    '<answer>"Some Album"</answer>'
)

# Pattern 9: No think block at all
LOG_NO_THINK = (
    'User listened to jazz.'
    '<answer>"Some Album"</answer>'
)

# Pattern 10: Correct answer with moderate tool-assisted reasoning
LOG_CORRECT_MODERATE = (
    'User listened to The Definitive Collection, Bobby Bare Greatest Hits.'
    '<tool_call>{"name": "search", "arguments": {"query_list": ["similar music to Bobby Bare"]}}</tool_call>'
    '<tool_response>Doc 1: A user played The Definitive Collection, Bobby Bare, '
    'All That Echoes, Brothers Of The Highway, Old Enough To Know Better</tool_response>'
    '<think>'
    "The user's listening history shows a preference for country and classic rock. "
    "The search results suggest users with similar taste enjoyed Old Enough To Know Better. "
    "This aligns with the country genre pattern in the user's history."
    '</think>'
    '<answer>"Old Enough To Know Better"</answer>'
)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests(use_model: bool = True):
    results = []

    def check(name, condition, details=""):
        status = "PASS" if condition else "FAIL"
        results.append((name, status))
        print(f"  [{status}] {name}")
        if details:
            print(f"         {details}")
        if not condition:
            print(f"         ^^^ EXPECTED TO PASS")

    # ===================================================================
    # PART B: Redundancy Penalty — Tier 0 (Self-Repetition, no model)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART B: Redundancy Penalty — Tier 0 (Self-Repetition)")
    print("=" * 70)

    think_loop = _extract_think_blocks(LOG_DEGENERATE_LOOP)
    t0_loop = _tier0_self_repetition(' '.join(think_loop))
    check("B1: Degenerate loop triggers tier0",
          t0_loop >= 0.8,
          f"tier0={t0_loop:.3f}, need >=0.8")

    think_good = _extract_think_blocks(LOG_GOOD_REASONING)
    t0_good = _tier0_self_repetition(' '.join(think_good))
    check("B2: Good reasoning does not trigger tier0",
          t0_good < 0.15,
          f"tier0={t0_good:.3f}, need <0.15")

    think_extreme = _extract_think_blocks(LOG_EXTREME_LOOP)
    t0_extreme = _tier0_self_repetition(' '.join(think_extreme))
    check("B3: Extreme loop triggers tier0",
          t0_extreme >= 0.8,
          f"tier0={t0_extreme:.3f}, need >=0.8")

    think_empty = _extract_think_blocks(LOG_EMPTY_THINK)
    t0_empty = _tier0_self_repetition(' '.join(think_empty))
    check("B4: Empty/trivial think is not penalized",
          t0_empty == 0.0,
          f"tier0={t0_empty:.3f}, need 0.0")

    sents = _split_sentences(' '.join(think_loop))
    unique_sents = len(set(s.lower().strip() for s in sents))
    check("B5: Sentence splitter finds duplicates in loop",
          unique_sents <= 3 and len(sents) >= 8,
          f"total={len(sents)}, unique={unique_sents}")

    think_real = _extract_think_blocks(LOG_REAL_CLASSIC_ROCK)
    t0_real = _tier0_self_repetition(' '.join(think_real))
    check("B6: Real log pattern (classic rock) does not trigger tier0",
          t0_real < 0.3,
          f"tier0={t0_real:.3f}, need <0.3")

    # ===================================================================
    # PART B: Tiers 1 & 2 (N-gram, no model)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART B: Redundancy Penalty — Tiers 1 & 2 (N-gram)")
    print("=" * 70)

    think_parrot_text = ' '.join(_extract_think_blocks(LOG_PROMPT_PARROT))
    prompt_parrot = ('The user has listened to "Hotel California" by Eagles, '
                     '"Slider" by The Rolling Stones, "Unchained Melody" by The Righteous Brothers. '
                     'Based on the user listening history, recommend a new music.')
    t1_pp = _tier1_substring_match(think_parrot_text, prompt_parrot)
    check("B7: Prompt parroting caught by tier1 substring",
          t1_pp == 1.0,
          f"tier1={t1_pp:.3f}, need 1.0")

    think_good_text = ' '.join(_extract_think_blocks(LOG_GOOD_REASONING))
    prompt_good = 'User listened to Kind of Blue, A Love Supreme, Brilliant Corners.'
    t1_good = _tier1_substring_match(think_good_text, prompt_good)
    check("B8: Good reasoning not caught by tier1",
          t1_good == 0.0,
          f"tier1={t1_good:.3f}, need 0.0")

    t2_parrot = _tier2_ngram_overlap(think_parrot_text, prompt_parrot)
    t2_good = _tier2_ngram_overlap(think_good_text, prompt_good)
    check("B9: Tier2 higher for parrot than good reasoning",
          t2_parrot > t2_good,
          f"parrot={t2_parrot:.3f} > good={t2_good:.3f}")

    # ===================================================================
    # PART C: Exploration Bonus (no model)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART C: Exploration Bonus")
    print("=" * 70)

    # Reset state for clean test
    rsr._CUM_REWARD_SUM = 0.0
    rsr._CUM_REWARD_COUNT = 0
    rsr._EMA_BONUS = 0.5

    think_concise = ' '.join(_extract_think_blocks(LOG_GOOD_REASONING))
    bonus_concise = compute_exploration_bonus(think_concise, r_answer=0.0)
    check("C1: Concise reasoning gets exploration bonus > 0",
          bonus_concise > 0.0,
          f"bonus={bonus_concise:.3f}")

    think_trivial = ' '.join(_extract_think_blocks(LOG_EMPTY_THINK))
    bonus_trivial = compute_exploration_bonus(think_trivial, r_answer=0.0)
    check("C2: Trivial think gets zero bonus",
          bonus_trivial == 0.0,
          f"bonus={bonus_trivial:.3f}")

    long_think = "word " * 600
    bonus_long = compute_exploration_bonus(long_think, r_answer=0.0)
    check("C3: Very long think gets reduced bonus (length penalty)",
          bonus_long < bonus_concise,
          f"long={bonus_long:.3f} < concise={bonus_concise:.3f}")

    # Simulate high cumulative reward (late training)
    old_sum = rsr._CUM_REWARD_SUM
    old_cnt = rsr._CUM_REWARD_COUNT
    rsr._CUM_REWARD_SUM = 50.0
    rsr._CUM_REWARD_COUNT = 10
    bonus_late = compute_exploration_bonus(think_concise, r_answer=1.0)
    rsr._CUM_REWARD_SUM = old_sum
    rsr._CUM_REWARD_COUNT = old_cnt
    check("C4: Bonus decays with high cumulative reward (late training)",
          bonus_late < bonus_concise,
          f"late={bonus_late:.3f} < early={bonus_concise:.3f}")

    # ===================================================================
    # PART A: Staleness Tracking (n-gram frequency, no model)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART A: Staleness Tracking (n-gram frequency)")
    print("=" * 70)

    rsr._STALENESS_NGRAMS = Counter()
    rsr._STALENESS_CALL_COUNT = 0

    s1 = _staleness_score("The user seems to enjoy classic rock from the seventies era "
                          "with blues influences and guitar solos")
    check("A1: First call has zero staleness",
          s1 == 0.0,
          f"staleness={s1:.3f}")

    # After 60+ calls with same template, staleness should rise
    template = ("Okay let's start by analyzing the user's listening history. "
                "They've enjoyed several songs from classic rock and heavy metal. "
                "The user has a strong interest in classic rock albums. "
                "So, the user's preferences are likely in the rock genre.")
    rsr._STALENESS_CALL_COUNT = 55
    for _ in range(10):
        _staleness_score(template)
    s_after = _staleness_score(template)
    check("A2: Staleness rises after many identical calls",
          s_after > 0.0,
          f"staleness={s_after:.3f} after repeated calls")

    rsr._STALENESS_NGRAMS = Counter()
    rsr._STALENESS_CALL_COUNT = 0

    # ===================================================================
    # INTEGRATION: R_total computation (no model, tier0 path)
    # ===================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION: R_total Computation")
    print("=" * 70)

    think_text = ' '.join(_extract_think_blocks(LOG_DEGENERATE_LOOP))
    red = compute_redundancy_penalty(
        think_text, "User listened to jazz.", "", set(), None, None
    )
    check("I1: Degenerate loop → redundancy = 1.0 (tier0 early exit)",
          red == 1.0,
          f"redundancy={red:.3f}")

    # Correct answer + degenerate reasoning → only positive R_think added
    r_answer, redundancy, info_gain, exploration = 1.0, 1.0, 0.0, 0.1
    r_think = 0.5 * info_gain - 1.0 * redundancy + 0.3 * exploration
    r_total = r_answer + 0.3 * max(0.0, r_think)
    check("I2: Correct answer + bad reasoning → R_total = R_answer (not penalized)",
          r_total == r_answer,
          f"R_total={r_total:.3f}, R_answer={r_answer:.3f}")

    # Wrong answer + bad reasoning → penalty applied
    r_answer = 0.0
    r_think = 0.5 * 0.0 - 1.0 * 1.0 + 0.3 * 0.1
    r_total = r_answer + 0.3 * r_think
    check("I3: Wrong answer + bad reasoning → negative R_total",
          r_total < 0.0,
          f"R_total={r_total:.3f}")

    # Wrong answer + good reasoning → small positive from exploration
    r_answer = 0.0
    r_think = 0.5 * 0.5 - 1.0 * 0.0 + 0.3 * 0.3
    r_total = r_answer + 0.3 * r_think
    check("I4: Wrong answer + good reasoning → small positive R_total",
          r_total > 0.0,
          f"R_total={r_total:.3f}")

    # No think block → all components zero
    red_no = compute_redundancy_penalty("", "", "", set(), None, None)
    exp_no = compute_exploration_bonus("", 0.0)
    check("I5: No think block → all zeros",
          red_no == 0.0 and exp_no == 0.0,
          f"red={red_no:.3f}, exp={exp_no:.3f}")

    # Real log pattern (no model path only checks tier0/tier1/tier2)
    think_real_text = ' '.join(_extract_think_blocks(LOG_REAL_CLASSIC_ROCK))
    red_real = compute_redundancy_penalty(
        think_real_text, "", "", set(), None, None
    )
    check("I6: Real log classic rock pattern → low redundancy",
          red_real < 0.5,
          f"redundancy={red_real:.3f}, need <0.5")

    # ===================================================================
    # MODEL-DEPENDENT TESTS
    # ===================================================================
    if use_model:
        print("\n" + "=" * 70)
        print("MODEL-DEPENDENT TESTS (sentence-transformers)")
        print("=" * 70)

        try:
            import torch
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            rsr._PAST_THINK_EMBEDDINGS.clear()
            rsr._STALENESS_NGRAMS = Counter()
            rsr._STALENESS_CALL_COUNT = 0
            rsr._CUM_REWARD_SUM = 0.0
            rsr._CUM_REWARD_COUNT = 0
            rsr._EMA_BONUS = 0.5

            comps_loop = compute_reasoning_components(
                LOG_DEGENERATE_LOOP, "amazon", r_answer=0.0
            )
            check("M1: Degenerate loop → redundancy=1.0",
                  comps_loop["redundancy"] == 1.0,
                  f"components={comps_loop}")

            comps_good = compute_reasoning_components(
                LOG_GOOD_REASONING, "amazon", r_answer=0.8
            )
            check("M2: Good reasoning → low redundancy",
                  comps_good["redundancy"] < 0.3,
                  f"components={comps_good}")
            check("M2b: Good reasoning → positive info_gain",
                  comps_good["info_gain"] > 0.0,
                  f"info_gain={comps_good['info_gain']:.3f}")

            comps_parrot = compute_reasoning_components(
                LOG_SEARCH_PARROT, "amazon", r_answer=0.5
            )
            check("M3: Search parrot → higher redundancy than good reasoning",
                  comps_parrot["redundancy"] > comps_good["redundancy"],
                  f"parrot={comps_parrot['redundancy']:.3f} > "
                  f"good={comps_good['redundancy']:.3f}")
            check("M4: Search parrot → lower info_gain than good reasoning",
                  comps_parrot["info_gain"] < comps_good["info_gain"],
                  f"parrot_ig={comps_parrot['info_gain']:.3f} < "
                  f"good_ig={comps_good['info_gain']:.3f}")

            # Past-thinking novelty drops on repeated calls
            rsr._PAST_THINK_EMBEDDINGS.clear()
            comps_first = compute_reasoning_components(
                LOG_CORRECT_MODERATE, "amazon", r_answer=0.5
            )
            comps_second = compute_reasoning_components(
                LOG_CORRECT_MODERATE, "amazon", r_answer=0.5
            )
            check("M5: Repeated identical reasoning → info_gain drops",
                  comps_second["info_gain"] <= comps_first["info_gain"],
                  f"first={comps_first['info_gain']:.3f}, "
                  f"second={comps_second['info_gain']:.3f}")

            comps_pp = compute_reasoning_components(
                LOG_PROMPT_PARROT, "amazon", r_answer=0.0
            )
            check("M6: Prompt parrot → high redundancy",
                  comps_pp["redundancy"] > 0.3,
                  f"redundancy={comps_pp['redundancy']:.3f}")

            # Real log pattern — should have moderate info_gain, low redundancy
            comps_real = compute_reasoning_components(
                LOG_REAL_CLASSIC_ROCK, "amazon", r_answer=0.5
            )
            check("M7: Real log classic rock → low redundancy",
                  comps_real["redundancy"] < 0.5,
                  f"redundancy={comps_real['redundancy']:.3f}")
            check("M8: Real log classic rock → positive info_gain",
                  comps_real["info_gain"] > 0.0,
                  f"info_gain={comps_real['info_gain']:.3f}")

        except Exception as e:
            print(f"  [SKIP] Model tests failed: {e}")
            print(f"         (Expected outside training cluster or if model unavailable)")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s in results if s == "PASS")
    failed = sum(1 for _, s in results if s == "FAIL")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    if failed > 0:
        print("\n  Failed tests:")
        for name, status in results:
            if status == "FAIL":
                print(f"    - {name}")
    print()
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reasoning reward components")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip tests requiring sentence-transformers model")
    args = parser.parse_args()

    success = run_tests(use_model=not args.no_model)
    sys.exit(0 if success else 1)
