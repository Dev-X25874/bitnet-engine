#include "bitnet_common.cuh"
#include <cuda_fp16.h>

__global__ void pack_activations_fp16(
    const __half* __restrict__ x,
    uint64_t*     __restrict__ x_pack,
    int K)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int word_idx = tid;
    int base     = word_idx * 64;
    if (base >= K) return;

    uint64_t word = 0ULL;
#pragma unroll
    for (int b = 0; b < 64 && (base + b) < K; ++b) {
        uint16_t raw;
        __half v = x[base + b];
        memcpy(&raw, &v, sizeof(raw));
        uint64_t sign = (raw >> 15) & 1ULL;
        word |= (sign << b);
    }
    x_pack[word_idx] = word;
}

extern "C"
__global__ void bitgemv_1bit(
    const uint64_t* __restrict__ W_packed,
    const uint64_t* __restrict__ x_packed,
    float*          __restrict__ y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = blockIdx.x * TILE_M + warp_id;

    if (row >= N) return;

    const uint64_t* w_row = W_packed + (size_t)row * K_words;

    int accum = 0;
#pragma unroll 4
    for (int w = lane; w < K_words; w += WARP_SIZE) {
        uint64_t ww = w_row[w];
        uint64_t xw = x_packed[w];
        accum += popcount64(xnor64(ww, xw));
    }

    accum = warp_reduce_sum_int(accum);

    if (lane == 0) {
        int K   = K_words * 64;
        float dot = (float)(2 * accum - K);
        atomicAdd(&y[row], w_scale * x_scale * dot);
    }
}

extern "C"
__global__ void bitgemv_ternary(
    const uint64_t* __restrict__ W_mag,
    const uint64_t* __restrict__ W_sign,
    const uint64_t* __restrict__ x_packed,
    float*          __restrict__ y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = blockIdx.x * TILE_M + warp_id;

    if (row >= N) return;

    const uint64_t* wm = W_mag  + (size_t)row * K_words;
    const uint64_t* ws = W_sign + (size_t)row * K_words;

    int dot = 0;
#pragma unroll 4
    for (int w = lane; w < K_words; w += WARP_SIZE) {
        uint64_t mag  = wm[w];
        uint64_t sign = ws[w];
        uint64_t xw   = x_packed[w];
        dot += popcount64(mag & xnor64(sign, xw));
        dot -= popcount64(mag & (sign ^ xw));
    }

    dot = warp_reduce_sum_int(dot);

    if (lane == 0)
        atomicAdd(&y[row], w_scale * x_scale * (float)dot);
}

extern "C"
__global__ void add_bias_relu(
    float*       __restrict__ y,
    const float* __restrict__ bias,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = y[i] + bias[i];
        y[i] = v > 0.f ? v : 0.f;
    }
}

extern "C" void launch_bitgemv_1bit(
    const uint64_t* W_packed,
    const uint64_t* x_packed,
    float*          y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words,
    cudaStream_t    stream)
{
    dim3 grid((N + TILE_M - 1) / TILE_M);
    dim3 block(THREADS_PER_CTA);
    bitgemv_1bit<<<grid, block, 0, stream>>>(
        W_packed, x_packed, y, w_scale, x_scale, N, K_words);
}

extern "C" void launch_bitgemv_ternary(
    const uint64_t* W_mag,
    const uint64_t* W_sign,
    const uint64_t* x_packed,
    float*          y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words,
    cudaStream_t    stream)
{
    dim3 grid((N + TILE_M - 1) / TILE_M);
    dim3 block(THREADS_PER_CTA);
    bitgemv_ternary<<<grid, block, 0, stream>>>(
        W_mag, W_sign, x_packed, y, w_scale, x_scale, N, K_words);
}
