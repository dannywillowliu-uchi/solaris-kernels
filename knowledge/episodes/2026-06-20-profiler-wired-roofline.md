# 2026-06-20 — Profiler wired; both easy wins are near-roofline at real shapes

Question raised: are we wasting perf by optimizing on wall-clock with no profiler? Wired
`prof.py` (torch.profiler busy% + launch count) into the harness and used ncu on the H100.
Answer: NO big bubbles left at the workload shapes — the structural wins already captured them.

## Profiler reads (H100, bf16)

| kernel / shape | busy% | launches | ncu | verdict |
|----------------|-------|----------|-----|---------|
| attn banded / stream (real) | **100%** | 7 (3 sdpa+cat+memset) | — | bubble-free; fully saturates GPU |
| attn banded / frame (small) | 80% | 7 | — | ~20% launch-gap bubble (small shapes only) |
| attn SDPA kernel / chunk | — | — | **SM 74.8%**, DRAM 7% | compute-bound at FA2-class MFU |
| adaln fused / full (real) | **99%** | **1** | — | compile fused whole chain to 1 Triton kernel |
| adaln fused / frame | 92% | 1 | — | tiny bubble |

AdaLN HBM check: full shape moves ~2.26 GB, measured 0.871 ms => ~78% of H100 HBM peak (3.35 TB/s).

## What this means (saved a night of wasted grinding)

- **No launch-bubbles to pop at the real shapes.** Batching the 3 banded SDPA calls would gain
  ~0% at stream (already 100% busy); only helps tiny shapes that don't matter.
- **Attention headroom = ~10-20%**, only via a better SDPA kernel (FA3 lifts 75% -> ~85-90% MFU).
  Hard (source build) for modest gain. Not the night's priority.
- **AdaLN headroom = ~22%** to HBM peak via hand-Triton. Modest.
- So the easy structural wins (banding, compile-fusion) already sit near their ceilings. The
  productive directions are NEW kernels + the FP8 lever, not squeezing these two.

## Loop direction set
1. FFN/QKV GEMM problem (FP8 vs bf16) — unexplored, compute-bound, the real aggressive lever.
2. VAE decode (conv3d) — unexplored, likely real headroom.
3. Then: FP8 attention (drift-gated), AdaLN hand-Triton (modest).
