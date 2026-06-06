#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <assert.h>

#define WARP_SIZE       32
#define WARP_MASK       0xFFFFFFFFu
#define TILE_K          256
#define TILE_M          4
#define THREADS_PER_CTA 128
#define BITS_PER_WORD   64

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            printf("[CUDA ERROR] %s:%d — %s\n",                               \
                   __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            __trap();                                                          \
        }                                                                      \
    } while (0)

__device__ __forceinline__ int popcount32(uint32_t x) {
    return __popc(x);
}

__device__ __forceinline__ int popcount64(uint64_t x) {
    return __popcll(x);
}

__device__ __forceinline__ uint32_t xnor32(uint32_t a, uint32_t b) {
    return ~(a ^ b);
}

__device__ __forceinline__ uint64_t xnor64(uint64_t a, uint64_t b) {
    return ~(a ^ b);
}

__device__ __forceinline__ int warp_reduce_sum_int(int val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(WARP_MASK, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_float(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(WARP_MASK, val, offset);
    return val;
}

__host__ __device__ __forceinline__
void ternary_encode(int8_t w, uint8_t& mag, uint8_t& sign_bit) {
    mag      = (w != 0) ? 1u : 0u;
    sign_bit = (w  < 0) ? 1u : 0u;
}

template<int ALIGN>
__host__ __device__ __forceinline__ size_t align_up(size_t n) {
    return (n + ALIGN - 1) & ~(size_t)(ALIGN - 1);
}
