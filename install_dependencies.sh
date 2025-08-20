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

# install dependencies 
uv sync