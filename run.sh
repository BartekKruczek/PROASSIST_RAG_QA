#!/bin/bash

MAX_JOBS=8

# check if nvcc is installed
cuda_installed=0
if nvcc --version
then
    echo "CUDA is installed."
    cuda_installed=1
else
    echo "CUDA is not installed."
fi

export CUDA_INSTALLED=$cuda_installed

# export repo folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# install dependencies based on CUDA installation
if [ $cuda_installed -eq 1 ]; then
    uv sync --extra gpu
    pip install flash-attn --no-build-isolation --verbose
    export FAISS_ENABLE_GPU=ON FAISS_OPT_LEVEL=avx512
    pip install --no-binary :all: faiss-cpu --verbose
else
    uv sync
fi

# run script
uv run raq_qa/main.py