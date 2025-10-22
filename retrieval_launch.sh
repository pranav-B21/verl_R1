module load gcc/15 cuda/12.6 nccl/12.4 nvidia_math

file_path='data/amazon_data'
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/corpora.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
