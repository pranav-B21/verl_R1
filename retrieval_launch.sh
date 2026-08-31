module load gcc/15 cuda/12.6 nccl/12.4 nvidia_math

file_path='data/amazon_data'

# --- corpus/index pair -------------------------------------------------------
# 2026-08-24: switched to the REPAIRED metadata corpus (SalesRank 0.0%->99.5%,
# Categories 0.0%->100.0%).  The originals read the Amazon *2014* keys
# salesRank/categories against the *2018* dump keyed rank/category, so both
# lookups missed silently and all 13,111 metadata docs shipped as
# "SalesRank: None, ..., Categories: ".  Rebuilt by data/regenerate_metadata.py
# + cf_corpus/build_index.py (fp32, --max-length 256).
#
# INDEX AND CORPUS ARE A MATCHED PAIR.  Corpus line order IS index row order
# (retrieval_server.load_docs maps a hit back by row), so a new index served
# against the old corpus returns WRONG DOCUMENTS SILENTLY -- no error, no
# mismatch warning, just the wrong text.  Override both or neither.
#
# To reproduce a pre-2026-08-24 run, serve the frozen pair:
#   CORPUS_TAG='' bash retrieval_launch.sh
CORPUS_TAG="${CORPUS_TAG-_v2}"
index_file=$file_path/e5_Flat${CORPUS_TAG}.index
corpus_file=$file_path/corpora${CORPUS_TAG}.jsonl

for f in "$index_file" "$corpus_file"; do
    [ -f "$f" ] || { echo "[retrieval_launch] FATAL: missing $f" >&2; exit 1; }
done
# rows in the FAISS index must equal lines in the corpus, or the pair is mismatched
n_docs=$(wc -l < "$corpus_file")
n_rows=$(python -c "import faiss,sys; print(faiss.read_index(sys.argv[1]).ntotal)" "$index_file")
[ "$n_docs" = "$n_rows" ] || {
    echo "[retrieval_launch] FATAL: mismatched pair -- $corpus_file has $n_docs docs but" >&2
    echo "                   $index_file has $n_rows rows. Serving these together returns" >&2
    echo "                   wrong documents silently. Refusing to start." >&2
    exit 1
}
echo "[retrieval_launch] serving $corpus_file ($n_docs docs) + $index_file"

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