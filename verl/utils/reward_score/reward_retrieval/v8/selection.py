# SPDX-License-Identifier: Apache-2.0
"""v8 selection components — the transferable "continuation" cue.

Why this exists
---------------
v7b fixed retrieval *behavior* (retr% 5% -> 100%) but the funnel measured at
`global_step_200` showed the binding constraint is one step later:

    coverage 2.77%  x  P(pick GT | GT in docs) ~= chance  =  HR@1 ~0.001

Measured on the v7b/baseline step-200 decodes (see the workspace-root
TEST_OUTPUT.md, 2026-07-30):

  * of 57 baseline cases where the GT was in the retrieved docs AND the model
    answered with a retrieved item, it picked the GT **0** times; v7b 2/53.
  * chance, given the ~18-26 candidate items a rollout retrieves, is 3.8-5.4%.

So selection is at chance: nothing in the reward, and nothing the model has
learned, tells it *which* retrieved item is the user's next one. No amount of
extra coverage helps while this stage multiplies by ~0.

The cue this module credits
---------------------------
The retrieved documents are *ordered user histories*: "A user played the
following musics before in sequence: "A", "B", "C", ...". That order is a
collaborative-filtering signal already present in the corpus -- if another user
played X then Y, and the current user just played X, then Y is a continuation
candidate. Nothing needs to be added to or changed in the corpus to expose it,
which keeps this inside the PI's constraint that the corpus stays as-is.

Measured support on the v7b step-200 decodes (3 decodes, 2974 rollouts with
parsed docs):

  * the successor set is non-empty on **60%** of rollouts (so the term is not
    sparse), and holds only **~2.1 items** (vs ~18.5 retrieved items).
  * the model's own answer already falls in that set 7.8% of the time -- low
    but non-zero, so the gradient has somewhere to go from.
  * when the GT is retrievable at all, it is a successor of one of the user's
    own history items in **13%** of cases (27% before excluding GTs that merely
    repeat an item already in the user's history).

PRE-REGISTERED CEILING: 13% is the most this term can buy. Perfect selection
*within* the successor set converts coverage 2.77% x 13% -> HR@1 ~0.0036, about
3x the baseline's ~0.001, and still short of the Table-1 HR@5 target of 0.0102.
This is a real but bounded lever and must not be described as more. See
./README.md for the full decision rule.

Why credit the RULE and not the answer
--------------------------------------
This term rewards a *structural property* of the emitted answer (is it a
continuation of something the user actually played?) rather than rewarding the
GT directly. Rewarding "the answer is the GT" is what the outcome reward
already does, and the train/test forensics show that path memorizes: train
P(hit | GT in docs) reaches 43-55% while test stays at 2-6%
(reward_reasoning/diagnostics/train_test_greedy_diagnostic.md). A structural
rule cannot memorize an item identity, so it is the transfer-safe way to move
selection.
"""

import re
from typing import List, Sequence, Set

# Retrieved docs are built by verl/tools/utils/search_r1_like_utils.py's
# _passages2string as "Doc N (Title: <text>)". Split on the header rather than
# regexing a balanced closing paren: album titles contain parentheses
# ("Greatest Hits (Deluxe Edition)") and a non-greedy \) capture truncates
# them. Same reasoning as audits/audit_a_selection.py::docs_in_rollout.
_DOC_SPLIT = re.compile(r"Doc\s+\d+\s+\(Title:\s*")

# Items inside a doc arrive JSON-escaped (\"Title\") because the whole
# <tool_response> payload is a JSON string; tolerate bare quotes for any
# decode that was not escaped.
_ITEM_ESCAPED = re.compile(r'\\"([^"\\]{1,200})\\"')
_ITEM_BARE = re.compile(r'"([^"\\]{1,200})"')

# The current user's history as rendered into the prompt by the dataset
# builder. Anchored on both ends so it cannot run past the history list into
# the instruction text that follows it.
_PROMPT_HISTORY = re.compile(
    r"The user has played the following musics before:\s*(.*?),?\s*please write", re.S
)

_SEQ_LEAD_IN = "a user played"


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().strip('"').lower()).strip()


def doc_sequences(retrieved_spans: Sequence[str]) -> List[List[str]]:
    """Ordered item lists, one per retrieved document.

    Order is load-bearing here -- the successor relation is the whole point of
    this module, so this must not be turned into a set.
    """
    out = []
    for span in retrieved_spans:
        for body in _DOC_SPLIT.split(span)[1:]:
            items = _ITEM_ESCAPED.findall(body) or _ITEM_BARE.findall(body)
            seq = [norm_title(x) for x in items]
            seq = [s for s in seq if s and not s.startswith(_SEQ_LEAD_IN)]
            if seq:
                out.append(seq)
    return out


def user_history(prompt_str: str) -> Set[str]:
    """Titles the current user already played, parsed from the prompt."""
    if not prompt_str:
        return set()
    m = _PROMPT_HISTORY.search(prompt_str)
    if not m:
        return set()
    return {norm_title(x) for x in _ITEM_BARE.findall(m.group(1))} - {""}


def successor_set(sequences: Sequence[Sequence[str]], history: Set[str]) -> Set[str]:
    """Items that directly follow one of the user's own items in some retrieved
    sequence -- "users who played what you played went on to play this".

    Items the user has already played are excluded: the task asks for a *new*
    item. That costs a little recall (2.6% of held-out targets do repeat an item
    already in the user's history) and is accepted deliberately, because
    crediting "recommend what they just played" would teach a degenerate policy
    that scores on the benchmark without recommending anything.
    """
    if not history:
        return set()
    cands = set()
    for seq in sequences:
        for i, item in enumerate(seq[:-1]):
            if item in history:
                cands.add(seq[i + 1])
    return cands - history


def compute_selection_components(retrieved_spans, prompt_str, answer_title) -> dict:
    """Structural quality of the emitted answer, given what was retrieved.

    Returns independent 0/1 indicators plus diagnostics. The orchestrator, not
    this module, decides how to weight them -- keeping measurement and policy
    separable so an ablation only has to change a weight, never this file.
    """
    seqs = doc_sequences(retrieved_spans)
    hist = user_history(prompt_str)
    succ = successor_set(seqs, hist)
    retrieved_items = {i for s in seqs for i in s}
    pred = norm_title(answer_title) if answer_title else ""

    return {
        # answered a continuation of something the user actually played
        "is_successor": float(bool(pred and pred in succ)),
        # answered *some* retrieved item (grounding; v7b already ~64% here)
        "is_grounded": float(bool(pred and pred in retrieved_items)),
        # answered something the user has already played -- wrong 97.4% of the
        # time, and the single cheapest transferable rule available
        "is_repeat": float(bool(pred and pred in hist)),
        "n_successors": len(succ),
        "n_retrieved_items": len(retrieved_items),
        "has_successors": float(bool(succ)),
    }
