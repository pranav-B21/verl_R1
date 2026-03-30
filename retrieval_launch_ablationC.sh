module load gcc/15 cuda/12.6 nccl/12.4 nvidia_math

file_path='data/amazon_data'
index_file=$file_path/ablations/ablation_c_no_reasoning/e5_c_Flat.index
corpus_file=$file_path/ablations/ablation_c_no_reasoning/corpora_c.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

export FAISS_DISABLE_GPU_STACK=1
export FAISS_ENABLE_GPU_MALLOC=1

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 1 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
