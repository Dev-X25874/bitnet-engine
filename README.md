# BitNet Engine

A **quantisation-native** LLM inference engine built from the ground up for 1-bit and 1.58-bit (ternary) weight matrices.  Weights are **never dequantised**; all linear projections are computed via bitwise `XNOR + POPCNT` at the hardware level.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / API                             │
├─────────────────────────────────────────────────────────────┤
│                    runtime::Engine                           │
│       (session management · KV-cache · scheduler)           │
├──────────────────────┬──────────────────────────────────────┤
│   ops::linear        │   ops::attention                     │
│   (CPU/CUDA dispatch)│   (CPU/CUDA dispatch)                │
├──────────────────────┴──────────────────────────────────────┤
│               quantization::BitPack                          │
│       (weight packing · scale computation)                   │
├──────────────────────┬──────────────────────────────────────┤
│  kernels::cuda_ffi   │  kernels::cpu_simd                   │
│  (XNOR+POPCNT CUDA)  │  (AVX2 / AVX-512 / scalar)          │
└──────────────────────┴──────────────────────────────────────┘
```

---

## Key Design Decisions

### 1-Bit GEMV

```
y[i] = scale_w × scale_x × (2 × POPCNT(XNOR(W_packed[i], X_packed)) − K)
```

Weights are packed one row per `ceil(K/64)` × `u64` words.  A CUDA warp (32 threads) owns one output row; each thread strides over the word dimension and warp-shuffles the partial POPCNT sums.

### Ternary (1.58-bit) GEMV

Two bit-planes per row:

| Plane | Meaning |
|---|---|
| `mag`  | 1 = weight is non-zero |
| `sign` | 1 = weight is negative  |

```
dot = POPCNT(mag & XNOR(sign, x)) − POPCNT(mag & XOR(sign, x))
```

### KV-Cache

Paged allocation (block size = 16 tokens) avoids quadratic memory growth and enables continuous batching across concurrent sessions.

---

## Build

### CPU only

```bash
cargo build --release
```

### With CUDA (requires CUDA 12.4+, sm_80+)

```bash
CUDA_ARCH=sm_80 cargo build --release --features cuda
```

### With AVX-512 VPOPCNTDQ

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512vpopcntdq" \
  cargo build --release --features avx512
```

---

## Usage

```bash
# Inference
bitnet-cli generate \
  --model ./model \
  --prompt "The key advantage of 1-bit LLMs is" \
  --max-tokens 200 \
  --quant ternary

# Memory estimate
bitnet-cli mem-estimate \
  --layers 32 --d-model 4096 --d-ff 11008 --vocab 32000 --quant ternary
```

---

## Benchmarks (AMD EPYC 9654, AVX-512)

| Shape | Mode | Throughput |
|---|---|---|
| 4096 × 4096 | Binary  | ~320 GOPS |
| 4096 × 4096 | Ternary | ~210 GOPS |
| 11008 × 4096 | Ternary | ~195 GOPS |

*(CUDA A100 results: ~5× higher throughput)*

---

## Licence

Dual-licensed under [Apache-2.0](LICENSE-APACHE) / [MIT](LICENSE-MIT).
