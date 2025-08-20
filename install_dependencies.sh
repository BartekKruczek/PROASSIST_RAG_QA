#!/bin/bash

MAX_JOBS=16

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

# install dependencies based on CUDA installation
if [ $cuda_installed -eq 1 ]; then
    uv sync --extra gpu
    echo "UV sync with GPU support"

    # pip install flash-attn --no-build-isolation
    # echo "Flash attention installed"

    # export FAISS_ENABLE_GPU=ON FAISS_OPT_LEVEL=avx512 FAISS_ENABLE_CUVS=ON
    # pip install --no-binary :all: faiss-cpu
    # echo "FAISS with GPU support installed"
else
    uv sync
    echo "UV sync without GPU support"
fi