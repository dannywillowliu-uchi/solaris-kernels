# Model facts (verified from source / cited)

Two models. They diverge MORE than the shared "Oasis" name suggests — read the divergence
table before assuming a kernel proven on the prototype transfers to the ship model.

## Prototype: Open-Oasis 500M  (etched-ai/open-oasis, camenduru/oasis-500m)

Verified by reading the cloned source (`dit.py`, `attention.py`, `vae.py`, `generate.py`).

- **DiT**: class `DiT`, blocks `SpatioTemporalDiTBlock` in `DiT.blocks` (ModuleList).
  Built via `DiT_S_2()` → **16 blocks** (NOT the signature default of 12), hidden **1024**,
  **16 heads**, **head_dim 64**, patch 2, latent ch 16, latent grid 18×32 → token grid
  9×16 = **144 spatial tokens/frame**, max **32 frames**, action dim **25** (additive into
  the AdaLN conditioning `c`).
- **Attention is AXIAL + torch SDPA** (`F.scaled_dot_product_attention`), NOT flash-attn,
  NOT custom, NOT full-3D:
  - `s_attn` — spatial, **bidirectional**, 144 keys, over B·T batch (the wide one).
  - `t_attn` — temporal, **causal**, ≤32 keys, over B·144 batch (the autoregressive axis).
- **AdaLN**: `*_adaLN_modulation = Sequential(SiLU, Linear(1024, 6*1024))`; split order
  `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`; `modulate = x*(1+scale)+shift`.
- **VAE is a ViT TRANSFORMER VAE** (`AutoencoderKL`, ViT-L patch-20), NOT 3D-conv: encode/decode
  are stacks of SDPA `AttentionBlock`s (enc 6 / dec 12, dim 1024, 16 heads). Per-frame 2D, no
  temporal compression. Decode path = method `AutoencoderKL.decode` (post_quant_conv → decoder
  blocks → dec_norm → predictor → unpatchify). Latent scale 1/12.75.
- **Sampler: DDIM, default 10 steps**, v-prediction, Diffusion-Forcing per-frame noise.
  **No DMD / distillation / few-step path exists in the open repo.**

## Ship: internal OASIS DMD driving model — 14B Wan2.1-style, DMD-distilled

THE actual ship target (operator-provided config). A multi-camera driving world model being
repurposed as the per-user multiplayer realtime renderer. Generic Wan2.1-14B numbers below are
cited; the OASIS-specific overrides are operator-provided and take precedence.

- **Implementation class: `WanTransformer3DModel` (HF diffusers).** => call sites for harvest live
  in diffusers' `modeling_wan` (WanAttnProcessor / WanTransformerBlock), NOT the original Wan repo.
  The serving stack is a diffusers-based fork. Scheduler is **FlowMatch + DMD few-step**.
- **DiT (dense, no MoE)**: dim **5120**, **40 layers**, **40 heads**, **head_dim 128**, FFN ~13824,
  patch **[1,2,2]**, **full 3D self-attention**. AdaLN/action conditioning, **action dim 10**
  (driving controls — NOT 25; the 500M's 25 is Minecraft).
- **3-CAMERA geometry (drives the token + attention shape):** RGB **512×2304** = 3 cameras tiled
  horizontally (left|front|right), each 512×768. VAE: 16 latent ch, ~8× spatial / ~4× temporal.
  Latent spatial **64×288** (288 = 3×96, one ~96-wide block per camera). Per 32-frame chunk:
  latent T≈8; tokens/latent-timestep = 32×144 = **4608**; **total ≈ 36,864 video tokens**. One
  latent frame ≈ 590 KB bf16; a full chunk latent ≈ 4.5 MB bf16.
- **DMD 4-step**, FlowMatch sigmas **[1, 0.9375, 0.8333, 0.625, 0]** (5 sigmas = 4 steps), iterative
  (denoise→re-noise). No teacher / CFG / critic at inference.
- **Streaming = block-causal**: bidirectional within a chunk, causal across chunks. KV computed ONCE
  per chunk at the clean pass, reused across all 4 denoising steps. Rolling window (sink + recent).
- **PROJECT-SPECIFIC KERNEL LEVER:** the spatial token grid (64×288) is **3 weakly-coupled camera
  blocks** (96 cols each). Left/front/right share little content, so cross-camera spatial attention
  is a candidate for a **block-sparse / camera-banded mask** — a targeted FLOP cut that falls
  directly out of the 3-camera tiling and is invisible to a generic dense attention port. Verify the
  cross-camera attention weight is actually low before pruning; if so, this is a high-value kernel.

## Divergence table (prototype vs ship) — what transfers and what does NOT

| Kernel | Open-Oasis 500M | 14B Wan ship | Transfers? |
|--------|-----------------|--------------|------------|
| Attention math | **axial** (s+t separate), SDPA, **head_dim 64**, ≤144 keys | **full 3D**, **head_dim 128**, **~36,864 keys** (3-cam, banded?) | **NO — different in kind.** 500M proves the harness, not the attention kernel. |
| AdaLN | SiLU+Linear→6·hidden, additive cond | same shape, larger hidden | **YES — same op, just wider.** Best transfer target. |
| VAE decode | **ViT transformer** (SDPA blocks) | **3D causal conv** (Wan-VAE) | **NO — different in kind.** ViT-VAE port ≠ conv3d port. |
| Sampler | DDIM 10-step | DMD 4-step iterative | harness only |
| Streaming/KV | temporal-causal axial, 32-frame | block-causal chunked, KV-per-chunk | concept transfers, shapes don't |

### Consequence
Open-Oasis 500M is a faithful prototype for the **harness, gate, ledger, and AdaLN port**. It is
**NOT** a kernel-transfer proxy for attention (axial hd64 vs full-3D hd128) or VAE (ViT vs conv3d).
For those, the right open prototype is **Wan2.1 itself** — and specifically the distilled streaming
variants **CausVid / Self-Forcing on Wan2.1-1.3B** (4-step, causal, KV-cached, ~17 FPS single H100),
which match the ship model's architecture AND its few-step autoregressive regime. See README
"Prototype choice".

### Few-step regime shifts the bottleneck (cited)
Per DiT forward, attention is 77-85% of the pass. But because the distilled model runs the DiT only
~4× while VAE decode is step-count-independent, **VAE decode rises to 30-40% of end-to-end latency**
in the few-step regime (Flash-VAED: 2.3% @50 steps → 31-41% @4 steps). So attention and VAE decode
are **co-priority** for the ship model, not attention-then-everything-else. Amortize across **chunks**
(KV reuse), not across the 4 steps — step-level diffusion caching (DeepCache) fails at 4 steps.
