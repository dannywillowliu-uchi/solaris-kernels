# 2026-06-20 — KV Craft B300 profile: VAE decode conv is ~83% of GPU time (NOT attention)

nsys (--cuda-graph-trace=node) over warm KV Craft generation on B300. Decisive.

## Top GPU kernels
| % | kernel | meaning |
|---|--------|---------|
| **80.0%** | implicit_convolveNd_sgemm<bf16,3> (1828 inst, avg 12ms, max 37ms) | 3D conv = VAE decode |
| 2.2% | convolve_common_engine_float_NHWC | more VAE conv |
| 4.1%+3.9% | fusion_17228 / gemm_fusion_dot (4150 inst) | DiT fused GEMM/attn (small each) |
| 1.3%+0.9%+0.7% | nvjet_tst (cuBLAS GEMM) | DiT matmuls |
| single-digit | fusions | attention — MINOR |

## Conclusion (reprioritizes everything)
- **VAE decode 3D conv = ~83% of generation GPU time.** It runs on the generic
  `implicit_convolveNd_sgemm` (non-tensor-core fallback) — the path XLA chose after cuDNN
  heuristics FAILED on Blackwell (the 38 fallbacks). This is THE kernel.
- **Attention is minor** (small 1.5B model, ~10k tokens). cuDNN-flash-attention is DEPRIORITIZED —
  it can't move an 8% slice much.
- GEMMs ~8% — FP8 later, modest.

## Strike order (revised)
1. VAE decode conv on Blackwell — layout (NDHWC/channels_last) + algo to hit tensor-core convs
   instead of the implicit_convolveNd fallback. Potentially huge (the 80% kernel).
2. (later) DiT GEMM FP8.
3. attention de-prioritized.

Baseline to beat: 1.96 fps (2-player, B300, warm).
