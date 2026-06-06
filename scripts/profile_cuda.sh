#!/usr/bin/env bash
# scripts/profile_cuda.sh
#
# Profile one generation pass with NVIDIA Nsight Systems.
# Requires: nsys (Nsight Systems CLI) and a CUDA build.
#
# Usage:
#   chmod +x scripts/profile_cuda.sh
#   ./scripts/profile_cuda.sh [model_path] [report_name]

set -euo pipefail

MODEL="${1:-./model}"
REPORT="${2:-bitnet_profile}"
PROMPT="The fundamental limit of 1-bit neural networks is"

echo "============================================"
echo " BitNet Engine — CUDA Profiler"
echo "============================================"
echo " Model:   $MODEL"
echo " Report:  $REPORT.nsys-rep"
echo "============================================"

# build CUDA release binary
echo ""
echo "[1/3] Building CUDA release binary..."
CUDA_ARCH="${CUDA_ARCH:-sm_80}" \
    cargo build --release --features cuda

# check nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys not found."
    echo "Install Nsight Systems from https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# run profiler
echo ""
echo "[2/3] Running Nsight Systems profile..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="${REPORT}" \
    --force-overwrite=true \
    ./target/release/bitnet-cli generate \
        --model   "${MODEL}" \
        --prompt  "${PROMPT}" \
        --max-tokens 50 \
        --quant   ternary

echo ""
echo "[3/3] Done."
echo ""
echo "Report written to: ${REPORT}.nsys-rep"
echo "Open with:         nsys-ui ${REPORT}.nsys-rep"
