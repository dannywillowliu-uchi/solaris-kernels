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

## AdaLN fusion — the correctness-preserving win (no banding caveat)

`problems/adaln_modulation`: eager AdaLN chain (LN -> modulate -> gated residual, x2) vs
`torch.compile` candidate. Same math, fused HBM passes. B300, bf16:

| shape | S × D | correct | eager | fused | speedup |
|-------|-------|---------|-------|-------|---------|
| frame | 4,608 × 5120 | exact* | 0.439 ms | 0.171 ms | 2.57x |
| chunk | 13,824 × 5120 | exact* | 1.319 ms | 0.479 ms | 2.75x |
| full | 36,864 × 5120 | exact* | 3.402 ms | 1.243 ms | 2.74x |

*max_err ~6e-2 = bf16 rounding (compiled uses different reduction order); allclose passes.
2.74x at the full ship shape from torch.compile alone, no hand-tuning, ALWAYS legal (unlike
banding). A hand-Triton fused kernel would push further toward the HBM roofline.

## Cross-SM port: H100 (sm_90) — PROVEN

Same problems, same harness, retargeted to H100 (temp-h100, 2x H100 80GB, torch 2.12+cu130,
cap 9.0). Reused an existing venv. bf16, single GPU.

camera-banded attention (speedup vs dense):
| shape | B300 sm_100 | H100 sm_90 |
|-------|-------------|------------|
| frame | 1.69x | 2.06x |
| chunk | 2.63x | 2.84x |
| stream | 2.93x | **2.97x** |

AdaLN fusion (speedup vs eager, correctness-preserving):
| shape | B300 sm_100 | H100 sm_90 |
|-------|-------------|------------|
| frame | 2.57x | 3.10x |
| chunk | 2.75x | 3.01x |
| full | 2.74x | **3.02x** |

Ratios transferred (as predicted: banding=FLOP effect, AdaLN=HBM effect — both arch-independent).
AdaLN is *better* on H100 (3.02 vs 2.74) because H100 is relatively more bandwidth-starved, so
killing HBM passes helps more. Absolute times: H100 dense-stream 50.8ms vs B300 20.96ms (~2.4x
slower) — H100 has less raw throughput, so the frame budget is tighter and kernels matter MORE
on the actual serving target.

## Bottom line — both claims PROVEN

1. Harness speeds up this model's kernels: banding ~2.9-3.0x + AdaLN ~2.7-3.0x.
2. Harness ports kernels across SM (sm_100 -> sm_90): same harness, speedups hold on both.

## Next
1. Get real model data -> measure cross-camera attention weight -> confirm banding is legal.
2. vae_decode (measure the placeholder) + wire the autonomous candidate-generation loop.
3. Push past the easy wins: hand-written FA3 / fused Triton with per-arch tuning.
