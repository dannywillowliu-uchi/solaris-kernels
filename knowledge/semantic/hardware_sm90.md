# Target: H100 / H200 (Hopper, sm_90)

Per-user serving GPU. The port must compile and run here.

## Compute (dense, no sparsity)

| dtype | H100 SXM | notes |
|-------|----------|-------|
| FP16 / BF16 | ~990 TFLOPS | structural-port math |
| FP8 (E4M3/E5M2) | ~1979 TFLOPS | aggressive-port lever (2x) |
| TF32 | ~495 TFLOPS | |

- H100 SXM: **132 SMs**, 228 KB smem/SM, HBM3 **~3.35 TB/s**.
- H200 SXM: same compute, HBM3e **~4.8 TB/s** (memory-bound kernels gain here).

## Hopper features the agent may use

- **TMA** (Tensor Memory Accelerator) — bulk async global<->shared copies, descriptor-based.
- **wgmma** — warpgroup async MMA (4th-gen tensor cores). m64 x n{8..256} x k16 for FP16.
- **Thread-block clusters + distributed shared memory** — multi-CTA cooperation.
- **Async barriers / mbarrier** — producer/consumer warp specialization (FA3 pattern).
- Native **FP8** tensor cores (E4M3/E5M2) — aggressive tier only.

## What does NOT exist on sm_90 (do not emit)

- **FP4 / MXFP4** tensor-core paths (Blackwell sm_100 only).
- **tcgen05** 5th-gen tensor core instructions.
- 2-SM "tcgen05" MMA, Blackwell tensor-memory.

A kernel using any of the above won't compile for sm_90. Porting B300->H100 means lowering
these to wgmma + TMA + (optionally) FP8.

## Source for reference: B300 (Blackwell, sm_100)

FP16 ~2250 TFLOPS, HBM ~8 TB/s, 148 SMs. The reference kernel times come from here; the
roofline ceiling on the target is lower (less BW, fewer SMs) so a "good" port is judged
against the **H100** roofline, not the B300 absolute time.
