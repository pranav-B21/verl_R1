
import os
import torch
import re
import json
import string
import random
from sentence_transformers import SentenceTransformer



def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def read_json(json_file:str) -> dict:
    f = open(json_file, 'r')
    return json.load(f)


# ---------------------------------------------------------------------------
# Reward-asset cache.
#
# similarity_match() used to construct a SentenceTransformer and re-read
# embeddings.pt (20 MB) + name2id.json on EVERY sample. At train_batch_size=56 x
# rollout.n=8 that is 448 model constructions and 448 file opens per step, all
# against /work (Lustre). On 2026-08-02 (job 883844, v8 resume) the Lustre client
# was evicted mid-run and every subsequent syscall on the mount returned
# ESHUTDOWN -- the traceback showed Errno 108 on both name2id.json and on the
# repo directory itself, i.e. the whole mount was gone. The metadata-op storm
# from this hot path is the most likely trigger, and it also cost real
# wall-clock (~9.5 min/step).
#
# The caches below are pure memoisation: identical objects, identical numerics.
# REC_DATA_ROOT redirects the reads to a node-local / $SCRATCH stage of the
# dataset (see run_in_container_rthink.sh) so the hot path never touches Lustre.
# It defaults to the historical "./data" so standalone callers are unaffected.
# ---------------------------------------------------------------------------
_DATA_ROOT = os.environ.get("REC_DATA_ROOT", "./data")

_MODEL_CACHE = {}
_CATALOG_CACHE = {}

_CATALOG_SUBDIRS = {
    "amazon": "amazon_data/CDs_and_Vinyl",
    "goodreads": "goodreads_data/Goodreads",
}


def _get_model(name: str = 'sentence-transformers/paraphrase-MiniLM-L3-v2'):
    """Construct the sentence-embedding model once per process."""
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _get_catalog(data_source: str, device):
    """Return (embeddings_on_device, name2id) for `data_source`, loaded once.

    The embedding matrix is cached already materialised on `device`, which also
    removes a 20 MB host->device copy per sample.
    """
    for key, subdir in _CATALOG_SUBDIRS.items():
        if key in data_source:
            break
    else:
        raise ValueError(
            f"reward_SPRec: no catalog for data_source={data_source!r} "
            f"(expected one of {sorted(_CATALOG_SUBDIRS)})"
        )

    cache_key = (key, str(device))
    if cache_key not in _CATALOG_CACHE:
        base = os.path.join(_DATA_ROOT, subdir)
        embeddings = torch.load(os.path.join(base, "embeddings.pt"))
        name2id = read_json(os.path.join(base, "name2id.json"))
        _CATALOG_CACHE[cache_key] = (torch.tensor(embeddings, device=device), name2id)
    return _CATALOG_CACHE[cache_key]


def similarity_match(solution_str, ground_truth, data_source, return_rank=False):
    """Tiered outcome reward.

    By default returns the float partial-credit `match` (baseline behaviour,
    unchanged byte-for-byte). When ``return_rank=True`` it instead returns a dict
    ``{"match", "rankId", "N", "target_found"}`` so callers (the v4 dense reward)
    can build a continuous, rank-based signal and apply the F9 fix (a target that
    is absent from ``name2id`` is reported as ``target_found=False`` rather than
    silently falling back to ``target_id=0`` — i.e. item-0's spurious rank-1).
    """
    title = extract_solution(solution_str)
    open_count, close_count = count_answer_tags(solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth["target"].strip().strip('"')}")
        print(f"Extracted answer: {title}")
        print(f"Solution string: {solution_str}")

    rankId = None
    n_items = None
    target_found = False
    if title:
        match = re.search(r'"([^"]*)', title)
        if match:
            text = match.group(1)
        else:
            text = solution_str.split('\n', 1)[0]

        # Identify your sentence-embedding model (cached; see _get_model)
        model = _get_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings, name2id = _get_catalog(data_source, device)

        predict_embedding = torch.tensor(model.encode(text), device=device)
        if predict_embedding.ndim == 1:
            predict_embedding = predict_embedding.unsqueeze(0)
        dist = torch.cdist(predict_embedding, embeddings, p=2).squeeze(0)
        rank = dist.argsort()
        n_items = len(rank)

        target_name = ground_truth["target"].strip().strip('"')
        target_found = target_name in name2id
        if target_found:
            target_id = name2id[target_name]
        else:
            target_id = 0  # baseline fallback (kept so the float `match` is unchanged)

        rank_pos = (rank == target_id).nonzero(as_tuple=False)
        rankId = rank_pos.item() + 1 if rank_pos.numel() > 0 else len(rank) + 1
        if rankId == 1:
            match = 1.0
        elif rankId <= 5:
            match = 0.8
        elif rankId <= 10:
            match = 0.5
        elif rankId <= 100:
            match = 0.1
        elif rankId <= 500:
            match = 0.001
        else:
            match = 0.0
        if open_count > 1 or close_count > 1:  # prevent output a lot of </answer>
            match = match / 4
            if match  == 0:
                match = -0.5
    else:
        match = 0.0

    if return_rank:
        return {
            "match": match,
            "rankId": rankId,
            "N": n_items,
            "target_found": target_found,
        }
    return match


def compute_score(solution_str, ground_truth, data_source, method='strict', format_score=0., score=1., return_rank=False):
    return similarity_match(solution_str, ground_truth, data_source, return_rank=return_rank)
