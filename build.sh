#!/bin/bash

# Save the current working directory
original_dir=$(pwd)

# Default values
COMPUTE_TYPE="CUDA"
CUDA_ARCH_SM="sm_$(nvidia-smi -i=0 --query-gpu=compute_cap --format=csv,noheader | sed "s/\.//")"

# Create and navigate to the build directory
mkdir -p build
cd build || exit

cmake ..
make

# Return to the original working directory
cd "$original_dir" || exit
