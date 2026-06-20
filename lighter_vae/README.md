# Lighter VAE decoder (distillation) — the multi-× frames-out lever

The frozen Wan-VAE decoder is ~83% of frames-out GPU time and ~13× too slow for 20-30 fps.
Kernels (cuDNN-9.23 done, FP8 conv ~2×) top out ~7 fps frames-out. The multi-× win is a
**lighter decoder distilled to match the frozen one in the SAME latent space** — a drop-in
faster decoder. Server (DiT) latents and the client contract are UNCHANGED.

## Why this is a config+training problem, not a new arch
KV Craft's decoder is `Decoder3d(dim=128, dim_mult=[1,2,4,4], num_res_blocks=2, z_dim=...)`.
The student is the SAME class with a lighter config:
- `dim` 128 → 64 (half the channels everywhere)
- `dim_mult` [1,2,4,4] → [1,2,2,2] (cap the high-res channel blowup — the 512-ch stages are the FLOP killers)
- `num_res_blocks` 2 → 1
- (optional) depthwise-separable convs in CausalConv3d for the high-res stages
Same `z_dim` and VAE_SCALE → identical latent interface.

## Distillation recipe
- **Teacher:** frozen Wan-VAE (vae.pt), `teacher.decode(latent)`.
- **Student:** light `Decoder3d`, `student.decode(latent)`.
- **Data:** latents from real frames — `latent = teacher.encode(eval_frames)` — plus latents the
  DiT actually produces (rollout latents) so the student matches the serving distribution.
- **Loss:** L1 + LPIPS (perceptual) on decoded RGB; teacher RGB is the target. Optionally a
  small GAN term for sharpness (Flash-VAED/LeanVAE style).
- **Gate:** student-vs-teacher PSNR/SSIM/FID on held-out frames MUST stay high; AND measure the
  student decode fps (the win). Target: multi-× faster at >~0.97 SSIM vs teacher.

## Files
- `student.py`  — light Decoder3d config + builder (reuses KV Craft src.models.wan_vae).
- `distill.py`  — load teacher, build student, distill on latents, validate quality + speed, save.

## Status
Harness written; needs GPU training (load vae.pt teacher + train student). Runs in the KV Craft
env. This is the real frames-out lever; FP8 conv kernels are the parallel ~2× margin.

## Drop-in
Once distilled + quality-gated, swap the student decoder into the serving VAE-decode path
(same `decode(latent, scale=VAE_SCALE)` signature). No DiT/client changes.
