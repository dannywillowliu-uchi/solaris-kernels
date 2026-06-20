# Overnight kernel-optimization loop — state

Autonomous loop started 2026-06-20 02:35 PDT. **Stop at 10:00 PDT 2026-06-20.** Hourly time check.
On each wake: `date`; if >= 10:00 stop + write final summary episode (do NOT schedule again).

## Boxes (personal project — authorized for use)

| arch | host | python | files |
|------|------|--------|-------|
| H100 (sm_90, TARGET) | root@31.56.109.71 | `/root/mpk-venv/bin/python` (torch 2.12+cu130) | `/root/oasis-port/{stdit_attention,adaln}` |
| B300 (sm_100, SOURCE) | root@95.133.253.31 | `/mnt/SFS-nc15dnf9/oasis-port/venv/bin/python` | `/mnt/SFS-nc15dnf9/oasis-port/{stdit_attention,adaln}` |

H100 has full profilers: `ncu` (PATH /usr/local/cuda-13.0/bin), `nsys`, `torch.profiler`.
Both boxes: root disk tight; H100 reuses existing mpk-venv, B300 env on NFS.

## Methodology — PROFILE FIRST (the whole point)

Do NOT optimize on wall-clock alone. Per kernel: `prof.py` (busy% + launch count) → if busy<90%
or many launches, pop the bubble; if busy~100% on one kernel, use `ncu --section SpeedOfLight`
to check roofline headroom — only write a faster kernel if there's real headroom. Then eval.py
(correctness + speedup) → record → commit.

## Current bests (speedup vs baseline, correct)

| kernel | B300 sm_100 | H100 sm_90 | profiler verdict |
|--------|-------------|------------|------------------|
| attention (camera-banded) | 2.93x | 2.97x | stream shape **100% busy, bubble-free**; cudnn flash SDPA. ncu headroom TBD. |
| adaln (compile fusion) | 2.74x | 3.02x | full shape **99% busy, 1 fused Triton kernel**, ~78% of HBM roofline. |

Key finding: both easy wins are ALREADY bubble-free / near-roofline at the real (large) shapes.
Launch-bubbles only exist at small shapes (attn frame 80% busy) which matter least. So further
gains are NOT bubble-popping — they're better kernels (FA3/FP8) or new kernels.

## Backlog (profiler-reprioritized, highest headroom first)

1. [DONE] ncu attention SDPA = 74.8% SM throughput (compute-bound). FA3 headroom only ~10-20%, hard source-build. De-prioritized.
2. [NEXT] NEW: FFN/QKV GEMM problem — FP8 vs bf16 (the real aggressive lever; compute-bound, less drift-prone). Likely more headroom than the already-optimized attn/adaln.
3. [ ] NEW: VAE decode problem — conv3d (Wan-VAE) at ship shape, cuDNN channels_last_3d vs default.
4. [ ] AdaLN hand-Triton to close the ~22% gap to HBM roofline (modest).
5. [ ] FP8 banded attention (aggressive tier) — speed + correctness vs bf16 golden; flag for drift gate.
6. [ ] FA3 attention (only if 1-5 exhausted; ~10-20%, source build).

## Log (append each experiment)
- 02:35 baseline recorded; profiler wired; attn/adaln confirmed bubble-free at real shapes.
- 02:40 ncu: attn SDPA 74.8% SM (compute-bound) -> FA3 only ~10-20%. adaln ~78% HBM peak. Easy wins near ceiling; pivot to NEW kernels (FFN-FP8, VAE) + FP8 lever.
