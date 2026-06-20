# Port strategy: B300/sm_100/FP16 -> H100/sm_90

Grounded in the Hopper-kernel research (FA3, FlexAttention/FA4, CUTLASS, Triton, cuDNN) and the
model facts in `model_facts.md`. Read both before authoring.

## Two tiers, always structural first

1. **Structural (FP16->FP16).** Re-express in Hopper primitives (TMA, wgmma, warp-spec). No
   precision change; tier-1 correctness is binary. Lands first, always.
2. **Aggressive (FP16->FP8).** Only when structural is already roofline-bound AND drift budget
   has headroom. See the FP8 placement rule below — it is NOT "FP8 everywhere".

## Toolchain hierarchy (recommendation tree)

```
ATTENTION
  standard mask, seqlen <= ~2k  -> cuDNN 9 SDPA (PyTorch 2.5+ default on H100)
  standard mask, seqlen >  2k   -> FlashAttention-3  (source-only beta in flash-attention/hopper/,
                                   NOT the flash-attn PyPI wheel; needs CUDA>=12.3)
  STRUCTURED / spatiotemporal mask:
     per-frame spatial (block-diagonal) -> flash_attn_varlen_func + cu_seqlens at frame bounds
                                           (block-diagonal by construction, zero mask overhead)
     spatial + causal-in-time / mixed   -> FlexAttention (PyTorch 2.6+) w/ FA4 backend
                                           (mask_mod predicate; MUST run under torch.compile)
     3D full-attention video DiT         -> Sliding Tile Attention (STA) or Sparse VideoGen
PLAIN GEMM (FFN/QKV)
  large batch         -> cuBLASLt via nn.Linear (already near-optimal)
  small-M FP8 / epilogue fusion -> Triton SplitK or CUTLASS 3.x ping-pong (NOT Triton-TMA: its
                                   descriptor H2D path is ~3000x slower for small-M)
FUSED ELEMENTWISE (AdaLN) -> Triton, always (80-90% peak BW)
CONV3D (Wan-VAE decode)  -> cuDNN with channels_last_3d (NDHWC) + cudnn.benchmark=True
                            (NDHWC is critical; NCHW inserts transposes that kill Tensor Cores)
```

## Per-kernel plan

### Attention
- **Ship (14B): full-3D, head_dim 128, ~33-76k tokens.** head_dim 128 is FA3's most-optimized
  path — dense ports cleanly. The real decision is the MASK: it's block-causal (bidirectional in
  a chunk, causal across chunks) with KV cache. Don't feed a dense mask to bare FA3 (wastes FLOPs).
  Use varlen for the spatial blocks and/or FlexAttention+FA4 for the block-causal structure; for
  full-3D consider STA/SVG sparsity (HunyuanVideo-class wins 1.6-10x over dense FA3).
- **Prototype (500M): axial SDPA, head_dim 64** — proves the harness, not the ship kernel. Don't
  over-invest tuning hd64 axial; it won't transfer to hd128 full-3D.
- **KV caching is across CHUNKS, not steps.** Historical K/V computed once per chunk at the clean
  pass, reused across all 4 denoising steps. Only the current chunk's K/V recompute per step.

### AdaLN  (best transfer target — same op on both models)
- Firmly **HBM-bandwidth-bound** (AI < 1 FLOP/byte). The win is collapsing 6-9 HBM passes (LN
  read-twice + write, separate scale/shift, separate residual) into **1 read + 1 write**.
- **Triton fused kernel**: residual -> LN (mean+var one pass, FP32 accumulate) -> affine ->
  `(1+scale)*x+shift` -> gate -> store. Template: flash-attn's `ops/triton/layer_norm.py`. This
  exact fusion is a validated **3.2-3.4x**. `torch.compile` gets ~10-15% free but breaks fusion at
  the conditioning-MLP GEMM. Apex FusedLayerNorm does NOT fuse modulation (wrong tool).
- FP8 buys ~nothing on compute here but halves bytes moved on a BW-bound op — only worth it if the
  pipeline is already FP8 end-to-end. **H200 runs this ~43% faster than H100 from BW alone.**

### VAE decode  (co-priority with attention in the few-step regime)
- **Ship (Wan-VAE): 3D causal conv, compute-bound.** cuDNN channels_last_3d first; verify via
  Nsight that `TensorOp` kernels fire (Hopper heuristics may skip TC for small GEMMs — benchmark=True
  works around). CUTLASS implicit-GEMM conv3d (stable since 3.6) only if cuDNN leaves >20% on the
  table. Note: an *architectural* swap (depthwise-separable conv3d) can beat any kernel work 5-6x —
  out of scope for a port loop, but flag it to the model team.
- **Prototype (500M): ViT transformer VAE (SDPA), NOT conv3d** — does not exercise the conv path.

## FP8 placement rule (critical for an autoregressive model)

FA3-FP8 gives ~1.6x real attention speedup but RMSE ~48x worse than FP16 **even with block-quant +
incoherent processing** — and those errors compound across denoising steps + frames. Empirically
naive FP8 dropped a video model's VBench 0.802->0.633.

- **Keep attention in FP16/BF16.** (Llama-3 405B also skips FP8 in self-attention.) If you must,
  only at the schedule ends (early/late sigma), never the mid-schedule steps, and gate hard on the
  trajectory rollout.
- **Spend FP8 on the dense FFN/QKV GEMMs** (compute-bound, far less drift-sensitive). This is the
  high-value, low-risk aggressive-tier target — not attention.

## The drift budget
`drift(port) <= drift_budget` over an N-frame rollout, pinned to "no worse than the shipped B300
FP16 deployment" once that trajectory exists. Structural ports should clear tier-2 trivially; the
gate exists to catch (a) bugs allclose missed on the captured shape and (b) FP8 drift on the GEMMs.

## Dead ends to seed
- _none yet — record FP8-attention drift, Triton-TMA small-M regressions, NCHW-conv3d transposes here._
