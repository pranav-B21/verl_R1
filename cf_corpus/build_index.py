# SPDX-License-Identifier: Apache-2.0
"""Encode a corpus JSONL and build a FAISS Flat index (step 4, re-index).

CRITICAL: this must reproduce the ONLINE document encoding in
examples/.../local_dense_retriever/retrieval_server.py exactly, or the offline
coverage sweep and the served retriever will disagree:
  * model  : intfloat/e5-base-v2
  * prefix : documents get "passage: " (queries get "query: " at serve time)
  * pooling: mean over tokens, masked by attention_mask
  * norm   : L2-normalize (so inner-product == cosine)
  * dtype  : fp16 on GPU (matches use_fp16=True); index stores fp32 vectors
  * index  : IndexFlatIP, 768-dim  (matches the existing e5_Flat.index:
             memmap size 85302*768*4 == 262,047,744 bytes)

Line order of the corpus JSONL becomes the vector/row order of the index, which
is how retrieval_server.load_docs maps a hit back to a doc -- so index and
corpus MUST be built from the same file (use assemble_corpus.py first).

Run in the retriever conda env (has torch/transformers/faiss). GPU strongly
recommended but CPU works for ~25k docs.

Usage:
  conda activate retriever
  python cf_corpus/build_index.py \
      --corpus data/amazon_data/cf/corpora_cf_meta.jsonl \
      --index  data/amazon_data/cf/e5_Flat_cf_meta.index
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_contents(path: str) -> list[str]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line)["contents"])
    return docs


def mean_pool(last_hidden_state, attention_mask):
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def encode_docs(docs, model_path, max_length, batch_size, use_fp16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    if use_fp16 and device == "cuda":
        model = model.half()

    out = np.empty((len(docs), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(docs), batch_size):
        # is_query=False -> "passage: " prefix, matching retrieval_server.Encoder
        batch = [f"passage: {d}" for d in docs[i : i + batch_size]]
        inp = tok(batch, max_length=max_length, padding=True, truncation=True,
                  return_tensors="pt").to(device)
        emb = mean_pool(model(**inp, return_dict=True).last_hidden_state,
                        inp["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        out[i : i + len(batch)] = emb.float().cpu().numpy()
        if (i // batch_size) % 20 == 0:
            print(f"  encoded {min(i + batch_size, len(docs))}/{len(docs)}", flush=True)
    return np.ascontiguousarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="match retrieval_query_max_length in retrieval_server.py")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--save-memmap", action="store_true",
                    help="also dump raw fp32 embeddings alongside the index")
    args = ap.parse_args()

    import faiss

    print(f"loading corpus {args.corpus} ...")
    docs = load_contents(args.corpus)
    print(f"  {len(docs)} docs")

    print("encoding (e5 'passage: ', mean-pool, L2-norm)...")
    emb = encode_docs(docs, args.model, args.max_length, args.batch_size, not args.no_fp16)
    dim = emb.shape[1]
    print(f"  embeddings {emb.shape} dtype={emb.dtype}")

    index = faiss.IndexFlatIP(dim)  # normalized vectors -> IP == cosine
    index.add(emb)
    assert index.ntotal == len(docs), (index.ntotal, len(docs))
    faiss.write_index(index, args.index)
    print(f"wrote FAISS IndexFlatIP ({index.ntotal} x {dim}) -> {args.index}")

    if args.save_memmap:
        mm_path = args.index.replace(".index", ".memmap")
        mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=emb.shape)
        mm[:] = emb[:]
        mm.flush()
        print(f"wrote embeddings memmap -> {mm_path}")

    print("\nServe with retrieval_server.py by pointing --index_path/--corpus_path "
          "at this index and its source corpus (same file used for build_index).")


if __name__ == "__main__":
    main()
