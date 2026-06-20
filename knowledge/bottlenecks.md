# KV Craft B300 bottlenecks (measured 2026-06-20)

Baseline: 1.96 fps (2-player, warm cache) = ~510 ms/frame. Target ~20 fps (50 ms/frame) => ~10x.

## Per-frame GPU time (from nsys --cuda-graph-trace)
| component | % | ~ms/frame | efficient? |
|-----------|---|-----------|------------|
| **VAE decode 3D conv** | **~83%** | **~420** | NO — non-tensor-core fallback |
| DiT (GEMMs + attention) | ~13% | ~66 | yes (cuBLAS nvjet tensor-core) |
| memory/layout glue (pad/concat/slice/reduce) | ~3% | ~15 | minor overhead |

## #1 bottleneck: VAE decode 3D convolution (the gate to realtime)
- Runs as `implicit_convolveNd_sgemm<bf16,3>` = cuDNN's GENERIC implicit-GEMM conv, NOT the Blackwell
  tensor-core conv path.
- Root cause: cuDNN heuristics found NO sm_100 algorithm for these 3D conv shapes (38x "None of the
  algorithms worked") -> XLA fell back to the generic kernel. Likely also NCDHW (bad) layout.
- Even if the rest were free, ~420 ms VAE alone caps at ~2.4 fps. Nothing else matters until this moves.
- Fix space: scoped NDHWC (VAE only), pinned cuDNN algo, cuDNN version bump, or a Pallas-GPU conv.
- Plausible 3-5x if it reaches a real tensor-core path -> overall ~5-8 fps, then FP8/glue close more.

## Not bottlenecks
- DiT GEMMs/attention (~13%): efficient, small model (1.5B) + few tokens (~10k). FP8 = low ROI.
- Memory glue (~3%): RoPE / KV concat / per-player channel concat-split / conv pad. Fusable later.

## Why this shape
few-step distillation (DiT only ~4 steps -> VAE dominates) x Blackwell conv immaturity (heuristic
fallback) => VAE share inflated from a normal ~30-40% to ~83%. NOT the usual "attention-bound".

## Dead-ends
- global `--xla_gpu_force_conv_nhwc` -> 0.72 fps (2.7x WORSE), reverted (DiT transposes; NDHWC conv
  still 33 fallbacks on Blackwell).
