# Research sources

## Models
- Open-Oasis (prototype): https://github.com/etched-ai/open-oasis
- Oasis-500M checkpoint mirror: https://huggingface.co/camenduru/oasis-500m
  - `oasis500m.pt` (ST-DiT), `vit-l-20.pt` (ViT-VAE)
- Wan2.1 (architecture family of the 14B ship model): video DiT + VAE

## Hopper kernel references
- FlashAttention-3 (TMA + wgmma + warp-spec, FP8 attention): tridao/flash-attention v3
- CUTLASS 3.x Hopper GEMM / collective builder (wgmma, TMA, pipeline)
- NVIDIA Hopper tuning guide; Nsight Compute roofline workflow
- Triton on Hopper (tl.dot wgmma lowering, TMA descriptors)
- Transformer Engine FP8 (E4M3/E5M2 recipes, scaling) — for the aggressive tier

## Internal
- ../amd-kernel-forge — the pattern this harness mimics (worst-shape-first + knowledge stores)
- ../kernel-forge, ../autonomous-kernel — prior autonomous kernel loops
