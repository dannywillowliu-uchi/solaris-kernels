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

## MAJOR PIVOT (03:30) — real target is SOLARIS (open JAX model)

Target model is now **Solaris** (github.com/solaris-wm/solaris, HF nyu-visionx/solaris) — a
multiplayer Minecraft world model, DiT on MatrixGame 2.0, **JAX implementation** (not PyTorch),
Self-Forcing distilled (few-step AR), rolling KV cache 6 latent frames, multiplayer joint
self-attention (players concatenated in tokens), per-player FFN/cross-attn, frozen MatrixGame VAE.
Runs on GPU via XLA-CUDA (README supports GPU inference, >=48GB).

CONSEQUENCES:
- The PyTorch synthetic kernels (attn/adaln/ffn) were a METHODOLOGY DEMO; do NOT keep grinding them.
- Kernel optimization is now JAX-NATIVE: (1) XLA cuDNN flash-attn flag, (2) Pallas-GPU kernels,
  (3) JAX FFI to FA3/CUTLASS. Profiling (nsys/ncu) is unchanged and still applies.
- Splash Attention (their fast kernel) is TPU/Pallas — does NOT port to GPU; XLA falls back. That
  fallback is the headroom.

CURRENT TASK (overnight): bring Solaris up on B300 (root@95.133.253.31), then profile.
- Setup running in tmux 'solaris' on B300; env on NFS /mnt/SFS-nc15dnf9/oasis-port/solaris-run.
- After setup (DONE_SETUP in setup.log): run `CUDA_VISIBLE_DEVICES=0 venv/bin/python src/inference.py
  experiment_name=solaris device.eval_num_samples=1`; confirm it generates a video; then nsys/ncu
  profile to find the real GPU hot kernels AND check whether XLA uses cuDNN flash or naive fallback.
- Next sessions/wakeups: CONTINUE SOLARIS (bring-up -> profile -> JAX kernel opt), not the old FFN task.

## Backlog (profiler-reprioritized, highest headroom first) — SUPERSEDED by Solaris pivot above

1. [DONE] ncu attention SDPA = 74.8% SM throughput (compute-bound). FA3 headroom only ~10-20%, hard source-build. De-prioritized.
2. [NEXT] NEW: FFN/QKV GEMM problem — FP8 vs bf16 (the real aggressive lever; compute-bound, less drift-prone). Likely more headroom than the already-optimized attn/adaln.
3. [ ] NEW: VAE decode problem — conv3d (Wan-VAE) at ship shape, cuDNN channels_last_3d vs default.
4. [ ] AdaLN hand-Triton to close the ~22% gap to HBM roofline (modest).
5. [ ] FP8 banded attention (aggressive tier) — speed + correctness vs bf16 golden; flag for drift gate.
6. [ ] FA3 attention (only if 1-5 exhausted; ~10-20%, source build).

## Log (append each experiment)
- 02:35 baseline recorded; profiler wired; attn/adaln confirmed bubble-free at real shapes.
- 02:40 ncu: attn SDPA 74.8% SM (compute-bound) -> FA3 only ~10-20%. adaln ~78% HBM peak. Easy wins near ceiling; pivot to NEW kernels (FFN-FP8, VAE) + FP8 lever.
- 03:10 ARCHITECTURE CORRECTION: real model = CS2 4-POV JOINT, views CHANNEL-CONCAT (4x16=64ch)
  over a SINGLE spatial grid (NOT 3-cam spatial tiling). => camera/view-BANDING DOES NOT APPLY
  (no per-view token separation; every token carries all views in channels). PARK banding work.
  All synthetic shapes (attn seq len, adaln, vae) are now SUSPECT — pending real serving config
  from model author (prompt sent). Keep arch-AGNOSTIC kernels running (FFN-FP8, VAE conv3d,
  adaln-Triton) but treat shapes as provisional; recompute once real config lands.
  Real conditioning: AdaLN adaln_embed_dim=32 (4 players x 8 actions), patch-embed in_channels=64.
- 19:59 MILESTONE: Solaris generated 2-player video end-to-end on B300 (output/solaris/eval_structure/video_0_side_by_side.mp4). JAX works on Blackwell. First-run ~6min incl conv-autotune compile (not fair fps). Compile cache warm -> doing clean timed re-run for steady-state fps.
- 20:03 BASELINE fps measured: 1.96 fps (257 frames 2-player / 131s, warm cache, 0 conv-fallbacks). ~510 ms/frame. Target ~20fps => ~10x gap. Recorded in results/gains.csv. Next: cuDNN-flash attention (lever #1).
- 20:17 PROFILE (nsys graph-trace): VAE decode 3D conv = ~83% GPU time (implicit_convolveNd_sgemm fallback). Attention MINOR. STRIKE = VAE conv layout/algo on Blackwell. cuDNN-flash deprioritized.
- 20:28 OPT-ATTEMPT #1 (VAE conv): global XLA --xla_gpu_force_conv_nhwc => 0.72 fps (vs 1.96 baseline) = 2.7x WORSE. REVERTED. Dead-end: forces DiT transposes + Blackwell NDHWC conv still 33 fallbacks. Next VAE-conv candidates: scoped-NDHWC(VAE only) / pinned cuDNN algo / cuDNN bump / Pallas-GPU conv.
- 21:01 WIN #1: cuDNN 9.10->9.23 => VAE conv 38 fallbacks->0, fps 1.96->4.14 (2.11x), warm-vs-warm. New baseline 4.14 fps. Recorded gains.csv. NEXT: re-profile (VAE conv should shrink) + launch autonomous loop.
