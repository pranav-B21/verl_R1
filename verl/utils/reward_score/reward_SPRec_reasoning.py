"""
Reasoning Reward Components for REC-R1.

Implements three components that combine into R_think:
  R_think = alpha * info_gain - beta * redundancy + gamma * exploration_bonus

=== Part A: Info Gain ===
  Measures whether <think> introduces genuinely new reasoning.
  1. Prompt copy gate (block-level cosine > 0.85 → 0)
  2. Sentence-level novelty vs prompt+search sentences
  3. Past-thinking novelty vs rollout buffer (catches template collapse)
  4. Staleness tracking via n-gram frequency across calls
  info_gain = min(sentence_score, past_novelty) * (1 - staleness_weight * staleness)
  Refs: Rewarding Progress, LongPAS (Relevance dimension)

=== Part B: Redundancy Penalty ===
  Detects copying, parroting, and self-repetition.
  Tier 0: Self-repetition (intra-think degeneration)
  Tier 1: Exact substring match vs prompt/search
  Tier 2: Frequency-weighted n-gram overlap
  Tier 3: Sentence-level semantic copy ratio
  Refs: AEPO (structural vs degenerate entropy), RM-R1 (entity stripping)

=== Part C: Exploration Bonus ===
  Encourages diverse reasoning early, exploits later.
  1. Sigmoid decay as cumulative reward rises
  2. EMA smoothing prevents oscillations (EPO)
  3. Length penalty for verbose blocks (AEPO)
  Refs: EPO (entropy smoothing), AEPO (length penalty)
"""

import re
import json
import math
import random
import torch
from collections import Counter, deque
from typing import Any, List, Dict, Tuple, Optional


# ===========================================================================
# Module-level caches and persistent state
# ===========================================================================

_SENTENCE_MODEL = None
_ENTITY_NAMES: Dict[str, set] = {}

# Part A: past-thinking buffer and staleness vocabulary
_PAST_THINK_EMBEDDINGS: deque = deque(maxlen=200)
_STALENESS_NGRAMS: Counter = Counter()
_STALENESS_CALL_COUNT: int = 0

# Part C: cumulative reward tracking and EMA
_CUM_REWARD_SUM: float = 0.0
_CUM_REWARD_COUNT: int = 0
_EMA_BONUS: float = 0.5
_EMA_ALPHA: float = 0.1


def _get_sentence_model() -> Any:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer(
            'sentence-transformers/paraphrase-MiniLM-L3-v2'
        )
    return _SENTENCE_MODEL


def _get_entity_names(data_source: str) -> set:
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
# Text helpers
# ===========================================================================

def _extract_think_blocks(solution_str: str) -> List[str]:
    return [m.strip() for m in
            re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)
            if m.strip()]


def _extract_info_blocks(solution_str: str) -> List[str]:
    matches = re.findall(r'<info>(.*?)</info>', solution_str, re.DOTALL)
    matches += re.findall(r'<tool_response>(.*?)</tool_response>',
                          solution_str, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _extract_prompt_text(solution_str: str) -> str:
    first_tag = len(solution_str)
    for tag in ['<think>', '<tool_call>', '<search>', '<answer>']:
        idx = solution_str.find(tag)
        if 0 <= idx < first_tag:
            first_tag = idx
    return solution_str[:first_tag].strip()


def _split_sentences(text: str) -> List[str]:
    sentences = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        for part in re.split(r'(?<=[.!?])\s+', line):
            if len(part.strip().split()) >= 3:
                sentences.append(part.strip())
    return sentences


def _strip_entities(text: str, entity_names: set) -> str:
    result = text
    for name in sorted(entity_names, key=len, reverse=True):
        result = re.sub(re.escape(name), ' [ENT] ', result, flags=re.IGNORECASE)
    result = re.sub(r'(\s*\[ENT\]\s*)+', ' [ENT] ', result)
    return re.sub(r'\s+', ' ', result).strip()


def _get_ngrams(words: List[str], n: int) -> List[Tuple]:
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


# ===========================================================================
# PART A: INFO GAIN
# ===========================================================================

def _prompt_copy_gate(
    think_text: str, prompt_text: str,
    model: Any, device: torch.device,
    threshold: float = 0.85,
) -> bool:
    """Block-level gate: if think ≈ prompt, info_gain = 0."""
    if not think_text or not prompt_text:
        return False
    t_emb = model.encode([think_text], convert_to_tensor=True, device=device)
    p_emb = model.encode([prompt_text], convert_to_tensor=True, device=device)
    sim = torch.nn.functional.cosine_similarity(t_emb, p_emb).item()
    return sim > threshold


def _sentence_level_novelty(
    think_sentences: List[str], ref_sentences: List[str],
    model: Any, device: torch.device,
) -> float:
    """
    Per-sentence novelty: 1 - max_sim against references.
    Short sentences (<5 words) get half weight (likely filler).
    Ref: LongPAS sub-step Relevance.
    """
    if not think_sentences or not ref_sentences:
        return 1.0
    t_embs = model.encode(think_sentences, convert_to_tensor=True, device=device)
    r_embs = model.encode(ref_sentences, convert_to_tensor=True, device=device)
    if t_embs.ndim == 1:
        t_embs = t_embs.unsqueeze(0)
    if r_embs.ndim == 1:
        r_embs = r_embs.unsqueeze(0)
    t_n = torch.nn.functional.normalize(t_embs, dim=1)
    r_n = torch.nn.functional.normalize(r_embs, dim=1)
    max_sims = torch.mm(t_n, r_n.t()).max(dim=1).values

    total_w, weighted_nov = 0.0, 0.0
    for i, sent in enumerate(think_sentences):
        w = 0.5 if len(sent.split()) < 5 else 1.0
        sim = max_sims[i].item()
        nov = max(0.0, 1.0 - sim) if sim < 0.85 else 0.0
        weighted_nov += w * nov
        total_w += w
    return weighted_nov / total_w if total_w > 0 else 0.0


def _past_thinking_novelty(
    think_text: str, model: Any, device: torch.device,
    compare_size: int = 50,
) -> float:
    """
    Compare current think block against buffer of recent rollouts.
    Catches template collapse. Ref: AEPO rollout diversity.
    """
    global _PAST_THINK_EMBEDDINGS
    if not think_text or len(think_text.split()) < 10:
        return 1.0
    t_emb = model.encode([think_text], convert_to_tensor=True, device=device)
    t_emb = torch.nn.functional.normalize(t_emb, dim=1)

    if len(_PAST_THINK_EMBEDDINGS) == 0:
        _PAST_THINK_EMBEDDINGS.append(t_emb.detach().cpu())
        return 1.0

    entries = list(_PAST_THINK_EMBEDDINGS)[-compare_size:]
    past = torch.cat(entries, dim=0).to(device)
    past_n = torch.nn.functional.normalize(past, dim=1)
    max_sim = torch.mm(t_emb, past_n.t()).squeeze(0).max().item()

    _PAST_THINK_EMBEDDINGS.append(t_emb.detach().cpu())
    return max(0.0, 1.0 - max_sim)


def _staleness_score(think_text: str) -> float:
    """
    N-gram frequency across calls. High overlap with accumulated vocabulary
    means template convergence. Ref: EPO temporal window.
    """
    global _STALENESS_NGRAMS, _STALENESS_CALL_COUNT
    words = think_text.lower().split()
    if len(words) < 10:
        return 0.0

    current = set(_get_ngrams(words, 3)) | set(_get_ngrams(words, 4))
    if not current:
        return 0.0

    # Compute overlap (only after enough history)
    staleness = 0.0
    if _STALENESS_CALL_COUNT > 50:
        overlap = sum(1 for ng in current if _STALENESS_NGRAMS[ng] >= 3)
        staleness = overlap / len(current)

    # Update vocabulary
    for ng in current:
        _STALENESS_NGRAMS[ng] += 1
    _STALENESS_CALL_COUNT += 1

    # Periodic decay every 500 calls (simulates epoch refresh)
    if _STALENESS_CALL_COUNT % 500 == 0:
        for key in list(_STALENESS_NGRAMS.keys()):
            _STALENESS_NGRAMS[key] //= 2
            if _STALENESS_NGRAMS[key] == 0:
                del _STALENESS_NGRAMS[key]

    return min(1.0, staleness)


def compute_info_gain(
    think_text: str, prompt_text: str, search_text: str,
    entity_names: set,
    model: Any, device: torch.device,
) -> float:
    """
    Part A. Returns [0, 1]: 0 = adds nothing, 1 = entirely novel.
    """
    if not think_text or len(think_text.split()) < 10:
        return 0.0

    ts = _strip_entities(think_text, entity_names)
    ps = _strip_entities(prompt_text, entity_names) if prompt_text else ""
    ss = _strip_entities(search_text, entity_names) if search_text else ""

    # Gate
    if ps and _prompt_copy_gate(ts, ps, model, device):
        return 0.0

    # Sentence novelty
    t_sents = _split_sentences(ts)
    ref = (ps + ' ' + ss).strip()
    r_sents = _split_sentences(ref) if ref else []
    sent_score = _sentence_level_novelty(t_sents, r_sents, model, device)

    # Past-thinking novelty
    past_nov = _past_thinking_novelty(ts, model, device)

    # Staleness
    stale = _staleness_score(think_text)
    if _STALENESS_CALL_COUNT < 200:
        stale_w = 0.0
    elif _STALENESS_CALL_COUNT < 500:
        stale_w = 0.2
    else:
        stale_w = 0.4

    # Combine: must be novel on BOTH axes
    ig = min(sent_score, past_nov) * (1.0 - stale_w * stale)
    return max(0.0, min(1.0, ig))


# ===========================================================================
# PART B: REDUNDANCY PENALTY
# ===========================================================================

def _tier0_self_repetition(think_text: str, min_words: int = 30) -> float:
    """Detect degenerate looping within a think block."""
    words = think_text.lower().split()
    if len(words) < min_words:
        return 0.0

    total_bi = len(words) - 1
    unique_bi = len(set(zip(words, words[1:])))
    bi_rep = 1.0 - (unique_bi / total_bi) if total_bi > 0 else 0.0

    total_tri = len(words) - 2
    unique_tri = len(set(zip(words, words[1:], words[2:])))
    tri_rep = 1.0 - (unique_tri / total_tri) if total_tri > 0 else 0.0

    ngram_score = 0.3 * bi_rep + 0.7 * tri_rep

    sentences = _split_sentences(think_text)
    sentence_score = 0.0
    if len(sentences) >= 3:
        normalized = [' '.join(s.lower().split()) for s in sentences]
        counts = Counter(normalized)
        top_count = counts.most_common(1)[0][1]
        dup_ratio = sum(c - 1 for c in counts.values()) / len(normalized)
        sentence_score = min(1.0, dup_ratio)
        if top_count >= 5:
            sentence_score = max(sentence_score, 0.9)
        elif top_count >= 3:
            sentence_score = max(sentence_score, 0.6)

    return max(ngram_score, sentence_score)


def compute_redundancy_penalty(
    think_text: str, prompt_text: str, search_text: str,
    entity_names: set,
    model: Optional[Any],
    device: Optional[torch.device],
    prompt_thresh: float = 0.8, search_thresh: float = 0.7,
    floor: float = 0.15,
) -> float:
    """Part B. Returns [0, 1]: 0 = no redundancy, 1 = heavy copying."""
    if not think_text or len(think_text.split()) < 15:
        return 0.0

    tier0 = _tier0_self_repetition(think_text)
    if tier0 >= 0.8:
        return 1.0

    ts = _strip_entities(think_text, entity_names)
    ps = _strip_entities(prompt_text, entity_names) if prompt_text else ""
    ss = _strip_entities(search_text, entity_names) if search_text else ""
    ref = (ps + ' ' + ss).strip()

    if not ref:
        raw = tier0
    else:
        # Tier 1 checks only against the prompt, not search results.
        # Referencing retrieved CF docs in reasoning is expected, not copying.
        tier1 = _tier1_substring_match(ts, ps) if ps else 0.0
        if tier1 >= 1.0:
            return 1.0
        tier2 = _tier2_ngram_overlap(ts, ref)
        tier3 = 0.0
        if model is not None and device is not None:
            t_sents = _split_sentences(ts)
            if t_sents:
                if ps:
                    p_sents = _split_sentences(ps)
                    if p_sents:
                        tier3 = max(tier3, _tier3_semantic_copy_ratio(
                            t_sents, p_sents, prompt_thresh, model, device
                        ))
                if ss:
                    s_sents = _split_sentences(ss)
                    if s_sents:
                        tier3 = max(tier3, _tier3_semantic_copy_ratio(
                            t_sents, s_sents, search_thresh, model, device
                        ))
        raw = max(tier0, tier1, tier2, tier3)

    # Length scaling + floor
    wc = len(think_text.split())
    sweet, cap = 150, 500
    mult = 1.0 if wc <= sweet else (1.0 + 0.5 * min(wc - sweet, cap - sweet) / (cap - sweet))
    scaled = min(1.0, raw * mult)
    if scaled < floor:
        return 0.0
    return min((scaled - floor) / (1.0 - floor), 1.0)


def _tier1_substring_match(think: str, ref: str, min_w: int = 10) -> float:
    tw = think.lower().split()
    rs = ' '.join(ref.lower().split())
    if len(tw) < min_w:
        return 0.0
    for i in range(len(tw) - min_w + 1):
        if ' '.join(tw[i:i + min_w]) in rs:
            return 1.0
    return 0.0


def _tier2_ngram_overlap(think: str, ref: str) -> float:
    tw = think.lower().split()
    rw = ref.lower().split()
    if not tw:
        return 0.0
    r_uni = set(rw)
    uni_r = sum(1 for w in tw if w in r_uni) / len(tw)
    t_bi = _get_ngrams(tw, 2)
    r_bi = set(_get_ngrams(rw, 2))
    bi_r = sum(1 for b in t_bi if b in r_bi) / len(t_bi) if t_bi else 0.0
    return 0.4 * uni_r + 0.6 * bi_r


def _tier3_semantic_copy_ratio(
    t_sents, r_sents, thresh, model, device
) -> float:
    if not t_sents or not r_sents:
        return 0.0
    te = model.encode(t_sents, convert_to_tensor=True, device=device)
    re_ = model.encode(r_sents, convert_to_tensor=True, device=device)
    if te.ndim == 1:
        te = te.unsqueeze(0)
    if re_.ndim == 1:
        re_ = re_.unsqueeze(0)
    tn = torch.nn.functional.normalize(te, dim=1)
    rn = torch.nn.functional.normalize(re_, dim=1)
    ms = torch.mm(tn, rn.t()).max(dim=1).values
    return (ms > thresh).sum().item() / len(t_sents)


# ===========================================================================
# PART C: EXPLORATION BONUS
# ===========================================================================

def compute_exploration_bonus(think_text: str, r_answer: float = 0.0) -> float:
    """
    Part C. Returns [0, ~1]. High early, decays with training progress.
    """
    global _CUM_REWARD_SUM, _CUM_REWARD_COUNT, _EMA_BONUS

    if not think_text or len(think_text.split()) < 10:
        return 0.0

    # Track cumulative reward
    _CUM_REWARD_SUM += r_answer
    _CUM_REWARD_COUNT += 1
    cum_avg = _CUM_REWARD_SUM / _CUM_REWARD_COUNT

    # Sigmoid decay: high when cum_avg is low (early training)
    cum_scaled = cum_avg * 10.0
    sigmoid = 1.0 / (1.0 + math.exp(0.5 * (cum_scaled - 5.0)))

    # EMA smoothing (EPO)
    _EMA_BONUS = _EMA_ALPHA * sigmoid + (1.0 - _EMA_ALPHA) * _EMA_BONUS
    smoothed = _EMA_BONUS

    # Length penalty (AEPO): sweet spot ~100 words, floor 30% at 500+
    wc = len(think_text.split())
    if wc < 10:
        length_factor = 0.0
    elif wc <= 100:
        length_factor = 1.0
    elif wc <= 500:
        length_factor = 1.0 - 0.7 * (wc - 100) / 400
    else:
        length_factor = 0.3

    return smoothed * length_factor


# ===========================================================================
# COMBINED: compute all components
# ===========================================================================

def compute_reasoning_components(
    solution_str: str, data_source: str,
    r_answer: float = 0.0, extra_info: dict = None,
) -> Dict[str, float]:
    """
    Returns dict with info_gain, redundancy, exploration_bonus.
    """
    think_blocks = _extract_think_blocks(solution_str)
    think_text = ' '.join(think_blocks) if think_blocks else ""

    if not think_text or len(think_text.split()) < 10:
        return {"info_gain": 0.0, "redundancy": 0.0, "exploration_bonus": 0.0}

    if extra_info and extra_info.get("prompt_str"):
        prompt_text = extra_info["prompt_str"]
    else:
        prompt_text = _extract_prompt_text(solution_str)
    search_text = ' '.join(_extract_info_blocks(solution_str))
    entity_names = _get_entity_names(data_source)

    model = _get_sentence_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ig = compute_info_gain(
        think_text, prompt_text, search_text, entity_names, model, device
    )
    red = compute_redundancy_penalty(
        think_text, prompt_text, search_text, entity_names, model, device
    )
    exp = compute_exploration_bonus(think_text, r_answer)

    return {"info_gain": ig, "redundancy": red, "exploration_bonus": exp}


# ===========================================================================
# verl entry point (backward compatible)
# ===========================================================================

def compute_score(
    solution_str: str, ground_truth: dict, data_source: str,
    method: str = 'strict', format_score: float = 0.,
    score: float = 1., extra_info: dict = None,
) -> float:
    """Backward-compatible: returns redundancy penalty only."""
    comps = compute_reasoning_components(
        solution_str, data_source, extra_info=extra_info
    )
    do_print = random.randint(1, 64) == 1
    if do_print:
        think_blocks = _extract_think_blocks(solution_str)
        tt = ' '.join(think_blocks) if think_blocks else ""
        t0 = _tier0_self_repetition(tt) if tt else 0.0
        wc = len(tt.split()) if tt else 0
        print(f"--------------------------------")
        print(f"[Reasoning] ig={comps['info_gain']:.3f}  "
              f"red={comps['redundancy']:.3f}  "
              f"exp={comps['exploration_bonus']:.3f}  "
              f"t0={t0:.3f}  wc={wc}")
        print(f"Solution string: {solution_str[:500]}")
    return comps["redundancy"]