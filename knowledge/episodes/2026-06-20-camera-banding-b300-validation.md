# 2026-06-20 — Camera-banding validated on B300 (~3x at ship shape)

First real-hardware run of the harness. Box: datacrunch fin-03, 2x B300 SXM6 (sm_100, cap
10.3), torch 2.11.0+cu128, env on NFS (root disk was 95% full). Single GPU, bf16.

## Result: the camera-banding lever is real at the kernel level

`problems/stdit_attention` (naive banded = 3 SDPA calls, block-diagonal over 3 cameras) vs
dense SDPA, synthetic inputs at ship shapes:

| shape | q × k tokens | correct | dense | banded | speedup |
|-------|--------------|---------|-------|--------|---------|
| frame | 4,608 × 4,608 | exact (0) | 0.248 ms | 0.147 ms | 1.69x |
| chunk | 13,824 × 13,824 | exact (0) | 2.555 ms | 0.973 ms | 2.63x |
| stream | 13,824 × 110,592 | exact (0) | 20.96 ms | 7.15 ms | **2.93x** |

Speedup → 3.0x FLOP-ideal as shape grows (small shapes: launch overhead of 3 kernels; large
shapes: O(n^2) FLOP cut dominates). At the realistic streaming shape it's 2.93x — nearly the
theoretical max, from the *structure* alone, with an unoptimized 3-call kernel.

## What this validates and what it does NOT

VALIDATED:
- Harness runs end-to-end on real silicon; correctness path exact (max_err 0).
- The roofline's `camera_band=1/3` assumption translates to real wall-clock (~3x), not just FLOPs.
- The highest-value attention win is *cheap to realize* — banding structure, not a fancy kernel.

NOT yet validated (open risks):
- **Semantic validity of banding.** Inputs are synthetic random. Whether the REAL model's
  cross-camera attention weight is low enough that banding doesn't hurt output quality is
  unproven — needs harvest + the trajectory drift gate. The kernel is correct *as a banded
  kernel*; whether banding is *allowed* for this model is the real open question.
- **Target arch.** This is B300 (sm_100), the SOURCE/reference. The 2.93x ratio is a FLOP
  effect and should largely transfer, but absolute times + that flash backends behave the same
  need an H100 (sm_90).
- **Headroom past naive.** A fused varlen / FlexAttention block-diagonal kernel could claw back
  the small-shape launch overhead and the last ~0.07x to 3.0x; modest vs the structural win.

## Next
1. Get real model data on a box → measure cross-camera attention weight → confirm banding is legal.
2. H100 to confirm the ratio on the port target.
3. Same treatment for vae_decode (measure the placeholder) + adaln.
