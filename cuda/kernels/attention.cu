#include "bitnet_common.cuh"
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#define HEAD_DIM_BITS  512
#define HEAD_DIM_WORDS (HEAD_DIM_BITS / 64)
#define SEQ_BLOCK      64

extern "C"
__global__ void bitnet_attention(
    const uint64_t* __restrict__ Q_pack,
    const uint64_t* __restrict__ K_pack,
    const __half*   __restrict__ V,
    __half*         __restrict__ O,
    float    scale,
    int      T,
    int      H,
    int      B)
{
    int b   = blockIdx.z;
    int h   = blockIdx.y;
    int q_t = blockIdx.x;

    if (b >= B || h >= H || q_t >= T) return;

    int lane    = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    const uint64_t* q_row = Q_pack
        + ((size_t)b * H + h) * T * HEAD_DIM_WORDS
        + (size_t)q_t * HEAD_DIM_WORDS;

    extern __shared__ char smem[];
    uint64_t* sk = reinterpret_cast<uint64_t*>(smem);
    __half*   sv = reinterpret_cast<__half*>(sk + SEQ_BLOCK * HEAD_DIM_WORDS);
    float*    ss = reinterpret_cast<float*>(sv + SEQ_BLOCK * (HEAD_DIM_BITS / 8));

    float m = -FLT_MAX;
    float d = 0.f;
    float o_acc[HEAD_DIM_BITS / 8 / sizeof(float)] = {};

    for (int tile_start = 0; tile_start < T; tile_start += SEQ_BLOCK) {
        int tile_end = min(tile_start + SEQ_BLOCK, T);
        int tile_len = tile_end - tile_start;

        for (int i = threadIdx.x; i < tile_len * HEAD_DIM_WORDS; i += blockDim.x) {
            int k_t  = tile_start + i / HEAD_DIM_WORDS;
            int word = i % HEAD_DIM_WORDS;
            sk[i] = K_pack[((size_t)b * H + h) * T * HEAD_DIM_WORDS
                           + (size_t)k_t * HEAD_DIM_WORDS + word];
        }

        for (int i = threadIdx.x; i < tile_len * (HEAD_DIM_BITS / 8); i += blockDim.x) {
            int k_t = tile_start + i / (HEAD_DIM_BITS / 8);
            int dim  = i % (HEAD_DIM_BITS / 8);
            sv[i] = V[((size_t)b * H + h) * T * (HEAD_DIM_BITS / 8)
                      + (size_t)k_t * (HEAD_DIM_BITS / 8) + dim];
        }
        __syncthreads();

        for (int ki = 0; ki < tile_len; ++ki) {
            int dot = 0;
            const uint64_t* k_row = sk + ki * HEAD_DIM_WORDS;
#pragma unroll
            for (int w = warp_id; w < HEAD_DIM_WORDS; w += (blockDim.x / WARP_SIZE))
                dot += popcount64(xnor64(q_row[w], k_row[w]));

            dot = warp_reduce_sum_int(dot);

            float score = (float)(2 * dot - HEAD_DIM_BITS) * scale;
            if (lane == 0 && warp_id == 0)
                ss[ki] = score;
        }
        __syncthreads();

        if (warp_id == 0) {
            float m_new = m;
            for (int ki = 0; ki < tile_len; ++ki)
                m_new = fmaxf(m_new, ss[ki]);

            float d_new = d * expf(m - m_new);
            for (int ki = 0; ki < tile_len; ++ki) {
                float p = expf(ss[ki] - m_new);
                ss[ki]  = p;
                d_new  += p;
            }

            float rescale = expf(m - m_new);
            for (int dim = lane; dim < HEAD_DIM_BITS / 8; dim += WARP_SIZE)
                o_acc[dim] *= rescale;

            m = m_new;
            d = d_new;
        }
        __syncthreads();

        for (int ki = 0; ki < tile_len; ++ki) {
            float p = ss[ki];
            const __half* v_row = sv + ki * (HEAD_DIM_BITS / 8);
            for (int dim = threadIdx.x; dim < HEAD_DIM_BITS / 8; dim += blockDim.x)
                o_acc[dim] += p * __half2float(v_row[dim]);
        }
        __syncthreads();
    }

    __half* out_row = O + ((size_t)b * H + h) * T * (HEAD_DIM_BITS / 8)
                        + (size_t)q_t * (HEAD_DIM_BITS / 8);

    for (int dim = threadIdx.x; dim < HEAD_DIM_BITS / 8; dim += blockDim.x)
        out_row[dim] = __float2half(o_acc[dim] / d);
}
