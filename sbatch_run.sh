#!/bin/bash

#SBATCH -p gh -N 1 -n 1 -t 10:00:00 -o output_1.log

source ~/.bashrc

cleanup() {
  if [[ -n "${retrieval_pid:-}" ]] && kill -0 "$retrieval_pid" 2>/dev/null; then
    kill "$retrieval_pid" 2>/dev/null || true
    wait "$retrieval_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

conda activate retriever
bash retrieval_launch.sh &
retrieval_pid=$!
conda deactivate

sleep 120  # wait 2 minutes so the retriever job fully settles

bash run_in_container.sh
