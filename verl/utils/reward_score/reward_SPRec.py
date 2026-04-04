
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


def similarity_match(solution_str, ground_truth, data_source):
    title = extract_solution(solution_str)
    open_count, close_count = count_answer_tags(solution_str)

    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth["target"].strip().strip('"')}")
        print(f"Extracted answer: {title}")
        print(f"Solution string: {solution_str}")

    if title:
        match = re.search(r'"([^"]*)', title)
        if match:
            text = match.group(1)
        else:
            text = solution_str.split('\n', 1)[0]
        
        # Identify your sentence-embedding model
        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        if "amazon" in data_source:
            embeddings = torch.load(f"./data/amazon_data/CDs_and_Vinyl/embeddings.pt")
            name2id = read_json(f"./data/amazon_data/CDs_and_Vinyl/name2id.json")
        if "goodreads" in data_source:
            embeddings = torch.load(f"./data/goodreads_data/Goodreads/embeddings.pt")
            name2id = read_json(f"./data/goodreads_data/Goodreads/name2id.json")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = torch.tensor(embeddings, device=device)
        
        predict_embedding = torch.tensor(model.encode(text), device=device)
        if predict_embedding.ndim == 1:
            predict_embedding = predict_embedding.unsqueeze(0)
        dist = torch.cdist(predict_embedding, embeddings, p=2).squeeze(0)
        rank = dist.argsort()

        target_name = ground_truth["target"].strip().strip('"')
        if target_name in name2id:
            target_id = name2id[target_name]
        else:
            target_id = 0

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
    return match


def compute_score(solution_str, ground_truth, data_source, method='strict', format_score=0., score=1.):
    match_score = similarity_match(solution_str, ground_truth, data_source)
<<<<<<< HEAD
    return match_score
=======
    return match_score
>>>>>>> upstream/vista
