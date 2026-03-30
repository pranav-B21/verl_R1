"""
Part B: Redundancy Penalty for REC-R1 Reasoning Reward.

Detects and penalizes <think> blocks that copy/parrot the input prompt or
<info> search results rather than synthesizing genuinely new reasoning.

Pipeline (fast → slow, with early exit on Tier 1):
  0. Entity stripping  — remove known item/artist names so referencing
                          "Kind of Blue" is not penalized (RM-R1 principle)
  1. Tier 1 — Exact substring: any 10+ word verbatim run → penalty 1.0
  2. Tier 2 — Weighted n-gram overlap: 0.4 * unigram + 0.6 * bigram ratio
  3. Tier 3 — Sentence-level semantic copy ratio:
               vs prompt sentences (cosine threshold 0.8)
               vs search result sentences (cosine threshold 0.7, stricter)

  redundancy = max(tier1, tier2, tier3)
  floor: redundancy < 0.3 → 0.0 | redundancy >= 0.3 → linear scale to 1.0

Integration:
  compute_score() is the verl entry point, matching the signature of
  reward_SPRec.compute_score(). Returns the penalty in [0.0, 1.0] — used
  as the -beta * redundancy term in:
    R_think = alpha * info_gain - beta * redundancy + gamma * exploration_bonus

References:
  - AEPO (Dong et al., 2026): structural vs degenerate entropy; threshold logic
  - LongPAS (2026): step-level decomposition; cosine sim as RL signal on verl+GRPO
  - RM-R1 (Chen et al., 2025): entity-stripping; evaluate substance over surface
  - REC-R1 framework: stricter search-result threshold (0.7) vs prompt (0.8)
"""

import re
import json
import random
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Module-level caches — loaded once per process, shared across verl workers
# ---------------------------------------------------------------------------

_SENTENCE_MODEL = None
_ENTITY_NAMES: Dict[str, set] = {}


def _get_sentence_model() -> SentenceTransformer:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    return _SENTENCE_MODEL


def _get_entity_names(data_source: str) -> set:
    """Load entity names from name2id.json for the given data source (cached)."""
    global _ENTITY_NAMES
    if data_source in _ENTITY_NAMES:
        return _ENTITY_NAMES[data_source]

    if "amazon" in data_source:
        path = "./data/amazon_data/CDs_and_Vinyl/name2id.json"
    elif "goodreads" in data_source:
        path = "./data/goodreads_data/Goodreads/name2id.json"
    else:
        _ENTITY_NAMES[data_source] = set()
        return _ENTITY_NAMES[data_source]

    try:
        with open(path, 'r') as f:
            _ENTITY_NAMES[data_source] = set(json.load(f).keys())
    except FileNotFoundError:
        _ENTITY_NAMES[data_source] = set()

    return _ENTITY_NAMES[data_source]


# ===========================================================================
# Text extraction helpers
# ===========================================================================

def _extract_think_blocks(solution_str: str) -> List[str]:
    """Return all <think>...</think> block contents."""
    matches = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _extract_info_blocks(solution_str: str) -> List[str]:
    """Return all <info>...</info> search result contents."""
    matches = re.findall(r'<info>(.*?)</info>', solution_str, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _extract_prompt_text(solution_str: str) -> str:
    """Return everything before the first model-output tag (the user prompt)."""
    first_tag = len(solution_str)
    for tag in ['<think>', '<tool_call>', '<search>', '<answer>']:
        idx = solution_str.find(tag)
        if 0 <= idx < first_tag:
            first_tag = idx
    return solution_str[:first_tag].strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences; drop fragments under 3 words (likely filler)."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for r in raw:
        sentences.extend(r.split('\n'))
    return [s.strip() for s in sentences if len(s.strip().split()) >= 3]


# ===========================================================================
# Step 0: Entity stripping
# Ref: RM-R1 — evaluate substance over surface features
# ===========================================================================

def _strip_entities(text: str, entity_names: set) -> str:
    """
    Replace known entity names (album titles, artist names) with [ENT].

    The model should reference "Kind of Blue" — that's reasoning, not copying.
    "Based on the user's listening history including Kind of Blue" is copying
    the prompt's framing. Stripping lets us penalize framing without penalizing
    necessary entity references.
    """
    result = text
    # Longest first so partial matches don't shadow full names
    for name in sorted(entity_names, key=len, reverse=True):
        result = re.sub(re.escape(name), ' [ENT] ', result, flags=re.IGNORECASE)
    result = re.sub(r'(\s*\[ENT\]\s*)+', ' [ENT] ', result)
    return re.sub(r'\s+', ' ', result).strip()


# ===========================================================================
# Tier 1: Exact substring match   O(n) — cheap, run first
# ===========================================================================

def _tier1_substring_match(think_text: str, reference_text: str,
                            min_words: int = 10) -> float:
    """
    Binary penalty: any contiguous run of min_words words from think_text
    that appears verbatim in reference_text → 1.0, else 0.0.

    Catches literal copy-paste that a cosine score would dilute across the
    whole block embedding.
    """
    think_words = think_text.lower().split()
    ref_str = ' '.join(reference_text.lower().split())

    if len(think_words) < min_words:
        return 0.0

    for i in range(len(think_words) - min_words + 1):
        if ' '.join(think_words[i:i + min_words]) in ref_str:
            return 1.0
    return 0.0


# ===========================================================================
# Tier 2: Weighted n-gram overlap   O(n) — cheap, order-insensitive
# ===========================================================================

def _get_ngrams(words: List[str], n: int) -> set:
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _tier2_ngram_overlap(think_text: str, reference_text: str) -> float:
    """
    overlap = 0.4 * unigram_ratio + 0.6 * bigram_ratio

    Bigram weight is higher because shared unigrams are noisy (common words
    like "the", "music", "user" appear in both naturally), but shared bigrams
    like "listening history" or "seems to prefer" are strong copying signals.

    This is ROUGE-1 + ROUGE-2 combined, minus ROUGE-L's order sensitivity —
    catches paraphrased copies that ROUGE-L misses.
    """
    think_words = think_text.lower().split()
    ref_words = reference_text.lower().split()

    if not think_words:
        return 0.0

    think_uni = set(think_words)
    ref_uni = set(ref_words)
    unigram_ratio = len(think_uni & ref_uni) / len(think_uni) if think_uni else 0.0

    think_bi = _get_ngrams(think_words, 2)
    ref_bi = _get_ngrams(ref_words, 2)
    bigram_ratio = len(think_bi & ref_bi) / len(think_bi) if think_bi else 0.0

    return 0.4 * unigram_ratio + 0.6 * bigram_ratio


# ===========================================================================
# Tier 3: Sentence-level semantic copy ratio   O(n*m) — medium cost, most granular
# Ref: LongPAS — step-level decomposition with cosine sim as RL signal
# ===========================================================================

def _tier3_semantic_copy_ratio(
    think_sentences: List[str],
    reference_sentences: List[str],
    threshold: float,
    model: SentenceTransformer,
    device: torch.device,
) -> float:
    """
    For each think sentence, compute max cosine similarity against all
    reference sentences. Sentences above threshold count as "copied".

    Returns: copied_count / total_think_sentences

    Different thresholds for prompt (0.8) vs search results (0.7, stricter):
    after a <tool_call> returns <info>, the <think> block must synthesize,
    not echo — so the model gets less room to parrot retrieved information.
    """
    if not think_sentences or not reference_sentences:
        return 0.0

    think_embs = model.encode(think_sentences, convert_to_tensor=True, device=device)
    ref_embs = model.encode(reference_sentences, convert_to_tensor=True, device=device)

    if think_embs.ndim == 1:
        think_embs = think_embs.unsqueeze(0)
    if ref_embs.ndim == 1:
        ref_embs = ref_embs.unsqueeze(0)

    think_norm = torch.nn.functional.normalize(think_embs, dim=1)
    ref_norm = torch.nn.functional.normalize(ref_embs, dim=1)

    # sim_matrix: (num_think, num_ref) — max per think sentence
    max_sims = torch.mm(think_norm, ref_norm.t()).max(dim=1).values

    return (max_sims > threshold).sum().item() / len(think_sentences)


# ===========================================================================
# Main redundancy penalty
# ===========================================================================

def compute_redundancy_penalty(
    solution_str: str,
    data_source: str,
    prompt_sim_threshold: float = 0.8,
    search_sim_threshold: float = 0.7,
    floor_threshold: float = 0.3,
) -> float:
    """
    Compute the redundancy penalty for a single rollout.

    Returns float in [0.0, 1.0]:
      0.0 = no meaningful redundancy detected
      1.0 = heavy copying / parroting

    Args:
        solution_str:          Full model output (contains <think>, <info>, <answer> etc.)
        data_source:           Dataset identifier ("amazon" or "goodreads")
        prompt_sim_threshold:  Cosine sim threshold for prompt copying (default 0.8)
        search_sim_threshold:  Cosine sim threshold for search result copying (default 0.7)
        floor_threshold:       Raw redundancy below this → 0 penalty (default 0.3)
    """
    think_blocks = _extract_think_blocks(solution_str)
    if not think_blocks:
        return 0.0  # No thinking → nothing to penalize

    think_text = ' '.join(think_blocks)
    prompt_text = _extract_prompt_text(solution_str)
    search_text = ' '.join(_extract_info_blocks(solution_str))

    if not prompt_text and not search_text:
        return 0.0  # Nothing to compare against

    # Step 0: Entity stripping (RM-R1: evaluate substance, not surface)
    entity_names = _get_entity_names(data_source)
    think_stripped = _strip_entities(think_text, entity_names)
    prompt_stripped = _strip_entities(prompt_text, entity_names)
    search_stripped = _strip_entities(search_text, entity_names) if search_text else ""

    # Combined reference for Tiers 1 & 2 (think shouldn't copy either source)
    all_reference = (prompt_stripped + ' ' + search_stripped).strip()

    # ------------------------------------------------------------------
    # Tier 1: Exact substring — free, binary, early exit
    # ------------------------------------------------------------------
    tier1 = _tier1_substring_match(think_stripped, all_reference, min_words=10)
    if tier1 >= 1.0:
        return 1.0

    # ------------------------------------------------------------------
    # Tier 2: Weighted n-gram overlap — cheap, order-insensitive
    # ------------------------------------------------------------------
    tier2 = _tier2_ngram_overlap(think_stripped, all_reference)

    # ------------------------------------------------------------------
    # Tier 3: Sentence-level semantic copy ratio — medium cost
    # Prompt: threshold 0.8 | Search results: threshold 0.7 (stricter)
    # Ref: LongPAS step-level decomposition; REC-R1 search synthesis goal
    # ------------------------------------------------------------------
    model = _get_sentence_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    think_sentences = _split_sentences(think_stripped)

    tier3_prompt = 0.0
    tier3_search = 0.0

    if think_sentences:
        prompt_sentences = _split_sentences(prompt_stripped)
        if prompt_sentences:
            tier3_prompt = _tier3_semantic_copy_ratio(
                think_sentences, prompt_sentences,
                threshold=prompt_sim_threshold, model=model, device=device,
            )

        if search_stripped:
            search_sentences = _split_sentences(search_stripped)
            if search_sentences:
                tier3_search = _tier3_semantic_copy_ratio(
                    think_sentences, search_sentences,
                    threshold=search_sim_threshold, model=model, device=device,
                )

    tier3 = max(tier3_prompt, tier3_search)

    # ------------------------------------------------------------------
    # Combine: redundancy = max across all tiers
    # ------------------------------------------------------------------
    raw_redundancy = max(tier1, tier2, tier3)

    # Floor threshold (AEPO/LongPAS):
    #   Below 0.3 → structural/acceptable overlap, zero penalty
    #   Above 0.3 → linear scale: 0.3 → 0.0, 1.0 → 1.0
    if raw_redundancy < floor_threshold:
        return 0.0
    return min((raw_redundancy - floor_threshold) / (1.0 - floor_threshold), 1.0)


# ===========================================================================
# verl entry point — matches reward_SPRec.compute_score() signature
# ===========================================================================

def compute_score(solution_str: str, ground_truth: dict, data_source: str,
                  method: str = 'strict', format_score: float = 0.,
                  score: float = 1.) -> float:
    """
    verl-compatible entry point. Signature matches reward_SPRec.compute_score.

    Returns the redundancy penalty in [0.0, 1.0].
    Used as the -beta * redundancy term in:
        R_think = alpha * info_gain - beta * redundancy + gamma * exploration_bonus
        R_total = R_answer + lambda * R_think
    """
    do_print = random.randint(1, 64) == 1

    penalty = compute_redundancy_penalty(solution_str, data_source)

    if do_print:
        think_blocks = _extract_think_blocks(solution_str)
        print(f"--------------------------------")
        print(f"[Redundancy] penalty={penalty:.3f}")
        print(f"Think blocks: {think_blocks}")
        print(f"Solution string: {solution_str}")

    return penalty


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    test_prompt = (
        "The user has listened to: Miles Davis - Kind of Blue, "
        "John Coltrane - A Love Supreme, Thelonious Monk - Brilliant Corners. "
        "Based on the user's listening history, recommend a music item."
    )

    # BAD: copies prompt framing verbatim
    test_bad = (
        f"{test_prompt}"
        "<think>Based on the user's listening history including Kind of Blue "
        "and A Love Supreme and Brilliant Corners, the user seems to like jazz. "
        "Based on the user's listening history, I should recommend jazz.</think>"
        "<answer>\"Miles Davis - Bitches Brew\"</answer>"
    )

    # GOOD: introduces new reasoning
    test_good = (
        f"{test_prompt}"
        "<think>This collection spans late-50s to mid-60s modal jazz and hard bop. "
        "The harmonic sophistication suggests the user appreciates complex chord "
        "progressions and improvisational depth. The era clustering around Blue Note "
        "recordings points toward post-bop exploration as a natural next step.</think>"
        "<answer>\"Wayne Shorter - Speak No Evil\"</answer>"
    )

    # PARROT: copies search results
    test_parrot = (
        f"{test_prompt}"
        "<tool_call>users who liked Kind of Blue</tool_call>"
        "<info>Users who enjoyed Kind of Blue also frequently listened to "
        "Herbie Hancock Maiden Voyage and Wayne Shorter Speak No Evil. "
        "These albums share modal jazz characteristics.</info>"
        "<think>Users who enjoyed Kind of Blue also frequently listened to "
        "Herbie Hancock Maiden Voyage and Wayne Shorter Speak No Evil. "
        "These albums share modal jazz characteristics. I will recommend one.</think>"
        "<answer>\"Herbie Hancock - Maiden Voyage\"</answer>"
    )

    print("=== Redundancy Penalty Test ===\n")
    print(f"BAD    (copies prompt):  {compute_redundancy_penalty(test_bad, 'amazon'):.3f}")
    print(f"GOOD   (original):       {compute_redundancy_penalty(test_good, 'amazon'):.3f}")
    print(f"PARROT (copies search):  {compute_redundancy_penalty(test_parrot, 'amazon'):.3f}")
