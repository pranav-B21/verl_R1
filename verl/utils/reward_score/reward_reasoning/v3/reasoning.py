"""
Reasoning-Reward components — iteration v3 ("evidence-grounded process reward").

These replace the v2 info_gain / redundancy / exploration components. The v2
design produced an R_think that was negative ~100% of the time and whose only
GRPO-surviving intra-group variance (redundancy) was *uncorrelated* with answer
correctness — so it injected misleading gradients into the (majority) all-wrong
groups and the model's answer quality declined. See:
  - ../../../../../../REWARD_REASONING_ANALYSIS.md  (v2 postmortem)
  - ../README.md                                    (iteration history)
  - ./README.md                                     (v3 design rationale)

Design principle
----------------
Under GRPO with std-normalised advantages, a reward component that is constant
across a group cancels; only its intra-group *variance* survives. A process
reward therefore only helps if that surviving variance is a **leading indicator
of correctness**. v3 rewards exactly the behaviours that precede a correct
recommendation in this search-agent setup, all measured *per rollout* from a
single solution string (no module-global state -> resume-safe, GRPO-visible):

  tool_use   — did the agent actually search, get results, and reason after?
  grounding  — is the <answer> supported by the retrieved evidence?
  synthesis  — does the post-retrieval <think> use the evidence AND differ from
               the pre-retrieval plan (i.e. retrieval actually changed it)?
  self_rep   — intra-<think> degenerate looping (a genuine pathology; penalty).

All four are returned in [0, 1]. The orchestrator combines them into R_think.
"""

import re
from typing import Any, Dict, List, Tuple

import torch

# Reuse the stable, cached low-level utilities from the canonical v2 module so
# the (expensive) SentenceTransformer is loaded exactly once per process and so
# v3 does not re-implement battle-tested text helpers.
from ...reward_SPRec_reasoning import (
    _get_sentence_model,
    _split_sentences,
    _extract_think_blocks,
    _extract_info_blocks,
    _tier0_self_repetition,
)

# Cap on how many evidence sentences we embed per rollout (retrieved results are
# truncated to 256 chars each upstream, so this is plenty and bounds cost).
_MAX_EVIDENCE_SENTENCES = 40
# Minimum word counts below which a span is treated as "not present".
_MIN_POST_THINK_WORDS = 8
_MIN_REP_WORDS = 30


# ---------------------------------------------------------------------------
# Embedding / extraction helpers
# ---------------------------------------------------------------------------

def _encode_norm(texts: List[str], model: Any, device: torch.device):
    """Encode `texts` and L2-normalise rows -> (N, D) tensor on `device`."""
    emb = model.encode(texts, convert_to_tensor=True, device=device)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return torch.nn.functional.normalize(emb, dim=1)


def _extract_answer_text(solution_str: str) -> str:
    """Return the predicted title from the last <answer> block, unwrapping a
    leading quoted span if present (mirrors reward_SPRec extraction)."""
    matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if not matches:
        return ""
    ans = matches[-1].strip()
    m = re.search(r'"([^"]+)"', ans)
    return (m.group(1) if m else ans).strip()


def _split_think_by_retrieval(solution_str: str) -> Tuple[str, str]:
    """Split <think> text into (pre_think, post_think) around retrieval.

    pre_think  = all <think> content occurring before the first retrieval result
    post_think = all <think> content occurring after the last retrieval result

    Retrieval results are <tool_response> (sglang multi-turn) or <info> blocks.
    If there is no retrieval result, everything counts as pre_think and
    post_think is empty.
    """
    first_resp = None
    m = re.search(r'<tool_response>|<info>', solution_str)
    if m:
        first_resp = m.start()
    last_resp = -1
    for m in re.finditer(r'</tool_response>|</info>', solution_str):
        last_resp = m.end()

    thinks = [(t.start(), t.group(1).strip())
              for t in re.finditer(r'<think>(.*?)</think>', solution_str, re.DOTALL)
              if t.group(1).strip()]
    if first_resp is None:
        return ' '.join(txt for _, txt in thinks), ""
    pre = ' '.join(txt for pos, txt in thinks if pos < first_resp)
    post = ' '.join(txt for pos, txt in thinks if pos >= last_resp)
    return pre, post


# ---------------------------------------------------------------------------
# Component 1: tool_use (structural)
# ---------------------------------------------------------------------------

def compute_tool_use(solution_str: str) -> float:
    """[0, 1] — did the agent search, obtain results, and reason about them?

    Graded so that merely emitting a query is cheap, while obtaining results and
    actually reasoning afterwards are what carry weight:
        0.3 * issued a search        (<tool_call> / <search>)
      + 0.3 * received results       (non-empty <tool_response> / <info>)
      + 0.4 * substantive post-think (>= 8 words after the last result)
    """
    has_search = bool(re.search(r'<tool_call>|<search>', solution_str))
    has_results = len(_extract_info_blocks(solution_str)) > 0
    _, post = _split_think_by_retrieval(solution_str)
    has_post_think = len(post.split()) >= _MIN_POST_THINK_WORDS
    return 0.3 * has_search + 0.3 * has_results + 0.4 * has_post_think


# ---------------------------------------------------------------------------
# Component 2: grounding (answer <-> evidence)
# ---------------------------------------------------------------------------

def compute_grounding(answer_text: str, evidence_norm, model: Any,
                      device: torch.device) -> float:
    """[0, 1] — is the predicted title supported by the retrieved evidence?

    Max cosine similarity of the <answer> title to any evidence sentence.
    Recommending an item that retrieval actually surfaced is a strong leading
    indicator of correctness, and (unlike v2 redundancy) rewards rather than
    punishes grounding the answer in evidence.
    """
    if evidence_norm is None or not answer_text:
        return 0.0
    a = _encode_norm([answer_text], model, device)
    return max(0.0, min(1.0, torch.mm(a, evidence_norm.t()).max().item()))


# ---------------------------------------------------------------------------
# Component 3: synthesis (post-retrieval reasoning quality)
# ---------------------------------------------------------------------------

def compute_synthesis(pre_think: str, post_think: str, evidence_norm,
                      ev_sents: List[str], model: Any,
                      device: torch.device) -> float:
    """[0, 1] — quality of the post-retrieval reasoning.

    Two equally weighted halves:
      use    = mean over post-think sentences of max cosine to evidence
               (the synthesis actually discusses the retrieved items)
      change = 1 - cosine(pre_think, post_think)
               (retrieval changed the plan, rather than restating it)

    Requiring *both* prevents gaming: copying evidence verbatim scores high on
    `use` but, being unchanged from no real planning, is bounded by `change`;
    diverging randomly scores high on `change` but low on `use`.
    """
    pt_sents = _split_sentences(post_think)
    if not pt_sents:
        return 0.0
    use = 0.0
    if evidence_norm is not None and ev_sents:
        t = _encode_norm(pt_sents, model, device)
        use = torch.mm(t, evidence_norm.t()).max(dim=1).values.mean().item()
    change = 1.0
    if pre_think.strip():
        p = _encode_norm([pre_think], model, device)
        q = _encode_norm([post_think], model, device)
        change = 1.0 - max(0.0, torch.mm(p, q.t()).item())
    return max(0.0, min(1.0, 0.5 * max(0.0, use) + 0.5 * change))


# ---------------------------------------------------------------------------
# Component 4: self_rep (degeneration penalty)
# ---------------------------------------------------------------------------

def compute_self_rep(solution_str: str) -> float:
    """[0, 1] — intra-<think> degenerate repetition (Tier-0 from v2).

    This is the *only* redundancy signal kept from v2: it fires on genuine
    looping/parroting within the model's own reasoning, not on overlap with the
    prompt or retrieved docs (which is legitimate grounding).
    """
    think_text = ' '.join(_extract_think_blocks(solution_str))
    if len(think_text.split()) < _MIN_REP_WORDS:
        return 0.0
    return _tier0_self_repetition(think_text)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_process_components(solution_str: str, data_source: str,
                               prompt_text: str = "",
                               extra_info: dict = None) -> Dict[str, float]:
    """Return {tool_use, grounding, synthesis, self_rep}, each in [0, 1].

    `prompt_text` (the user's listening/reading history) is appended to the
    retrieved results to form the evidence set, since a recommendation grounded
    in the user's own history is also reasonable evidence.
    """
    answer_text = _extract_answer_text(solution_str)
    pre_think, post_think = _split_think_by_retrieval(solution_str)
    info_blocks = _extract_info_blocks(solution_str)

    tool_use = compute_tool_use(solution_str)
    self_rep = compute_self_rep(solution_str)

    # Evidence = retrieved results first, then the user's history.
    ev_sents: List[str] = []
    for blk in info_blocks:
        ev_sents += _split_sentences(blk)
    if prompt_text:
        ev_sents += _split_sentences(prompt_text)
    ev_sents = ev_sents[:_MAX_EVIDENCE_SENTENCES]

    grounding = synthesis = 0.0
    if ev_sents or post_think or answer_text:
        model = _get_sentence_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evidence_norm = _encode_norm(ev_sents, model, device) if ev_sents else None
        grounding = compute_grounding(answer_text, evidence_norm, model, device)
        synthesis = compute_synthesis(
            pre_think, post_think, evidence_norm, ev_sents, model, device
        )

    return {
        "tool_use": float(tool_use),
        "grounding": float(grounding),
        "synthesis": float(synthesis),
        "self_rep": float(self_rep),
    }
