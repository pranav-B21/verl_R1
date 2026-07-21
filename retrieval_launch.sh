module load gcc/15 cuda/12.6 nccl/12.4 nvidia_math

file_path='data/amazon_data'
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/corpora.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

export FAISS_DISABLE_GPU_STACK=1
export FAISS_ENABLE_GPU_MALLOC=1

# NOTE: --topk here is a FALLBACK ONLY and does NOT control retrieval depth.
# verl/tools/search_tool.py:178 sets topk = config.get("topk", 3) and sends it in the
# request payload; retrieval_server.py:370-371 only falls back to this value when the
# request omits topk, which never happens. Effective top-k is therefore 3, and it is set
# in examples/sglang_multiturn/config/tool_config/search_tool_config.yaml (which does not
# currently declare a `topk:` key, hence the search_tool.py default of 3).
# Confirmed empirically: real rollouts recover exactly 3.03 docs/turn.
# ==> To widen top-k (roadmap M2, PI-endorsed exploration), add `topk: 20` to
#     search_tool_config.yaml. Changing the flag below will do nothing.
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 1 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu