#include "bitnet_common.cuh"
#include <cuda_fp16.h>
#include <math.h>

#define EPS 1e-6f

extern "C"
__global__ void rmsnorm_fp16(
    const __half* __restrict__ x,
    __half*       __restrict__ y,
    const __half* __restrict__ w,
    int           dim)
{
    int row = blockIdx.x;
    const __half* x_row = x + (size_t)row * dim;
    __half*       y_row = y + (size_t)row * dim;

    float ms = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        ms += v * v;
    }
    ms = warp_reduce_sum_float(ms);

    __shared__ float smem[32];
    int lane    = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) smem[warp_id] = ms;
    __syncthreads();

    if (warp_id == 0) {
        ms = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? smem[lane] : 0.f;
        ms = warp_reduce_sum_float(ms);
        if (lane == 0) smem[0] = ms;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / (float)dim + EPS);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]) * rms_inv;
        y_row[i] = __float2half(v * __half2float(w[i]));
    }
}

extern "C" void launch_rmsnorm_fp16(
    const __half* x,
    __half*       y,
    const __half* w,
    int           rows,
    int           dim,
    cudaStream_t  stream)
{
    int threads = min(dim, 1024);
    rmsnorm_fp16<<<rows, threads, 0, stream>>>(x, y, w, dim);
}
