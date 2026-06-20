# 2026-06-20 — KV Craft real architecture (from open code) + the cuDNN-flash lever

Read the open KV Craft repo ((base WM repo, internal)). Replaces ALL prior synthetic-shape
guesses with real numbers, and surfaced the #1 GPU kernel lever before running anything.

## Real architecture (config + src/models/transformer.py)

- **DiT backbone:** hidden **1536**, **30 layers**, **12 heads**, **head_dim 128**, ffn **8960**,
  patch_size **[1,2,2]**, qk_norm on. (~1.5-2B params, NOT 14B.)
- **obs_resolution 360x640**; **880 patches per frame per player**; num_frames_context **33**.
- **Multiplayer:** `the MP model class`, `multiplayer_method: concat_c`. Causal **block mask**,
  `block_size = spatial_size * num_players` (= 880*2 = 1760 tokens/frame for 2 players). Per-player
  cross-attn/FFN; joint multiplayer self-attention.
- **Action module:** keyboard_dim_in 23 + mouse_dim_in 2, 16 heads, hidden 128 (separate from DiT).
- **Serving:** Self-Forcing distilled, autoregressive, rolling KV cache **6 latent frames**.
- **VAE:** frozen MatrixGame-2.0 Wan VAE (src/models/wan_vae.py).
- **Framework:** JAX. TPU uses Splash Attention (Pallas, does NOT port to GPU).

## #1 LEVER (found by reading, to confirm by profiling)

`src/models/transformer.py:240` and `:261`:
```
x = jax.nn.dot_product_attention(q, k, v, is_causal=False)   # NO implementation= arg
```
=> uses XLA's DEFAULT attention lowering on GPU, **not cuDNN FlashAttention.** `jax.nn.dot_product_attention`
accepts `implementation='cudnn'` to dispatch to cuDNN's fused flash kernel (the sm_90/sm_100
`sdpa_*_flash` kernel, ~75% MFU, the same one we profiled). Switching GPU inference to cudnn should
be a large speedup with identical numerics. CAVEAT: the multiplayer causal *block* mask must be
expressible to cuDNN (causal yes; arbitrary block mask may need handling / Pallas fallback).

## Plan (in scope = kernels)

1. Run inference on B300 (GPU 0), confirm it generates a video (validates JAX on Blackwell sm_100).
2. nsys/ncu profile -> confirm default attention is the bottleneck and how far below cuDNN it is.
3. Flip GPU attention to `implementation='cudnn'` (single-player path first; handle MP block mask).
   Measure speedup + numeric match. This is the first real KV Craft kernel win.
4. Then: per-player FFN/attn batching, VAE decode, Pallas-GPU for what cuDNN can't cover.

## Token-count reality
Sliding window 6 frames x 1760 tokens (2p) ~= 10.5k attention tokens at serving — MUCH smaller than
the ~37k I'd synthesized. The whole model is smaller/cheaper than assumed; realtime is more reachable.
