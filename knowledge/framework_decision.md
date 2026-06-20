# Framework decision (recommendation, pending sign-off)

Q: keep optimizing in JAX (Solaris's impl) or reimplement the serving DiT in PyTorch/CUDA?

## Recommendation: PyTorch/CUDA for the production serving stack
- JAX kernel opt = fighting the framework: Pallas (< Triton maturity), JAX-FFI (fiddly),
  XLA-on-Blackwell rough (38 conv fallbacks, GEMM-autotune precision warnings). FA4 / FP8 /
  NVFP4 / fused CuTe kernels are all non-native -> FFI or Pallas reimpl each.
- The mature, portable kernel ecosystem (FA4, CUTLASS/CuTe-DSL FP8+NVFP4 GEMM, TransformerEngine,
  Triton, torch.compile) is PyTorch/CUDA-native, and it's where the team's kernel expertise is.
- Port is BOUNDED: DiT is 1.5B, Wan2.1 architecture has reference PyTorch. Reimplement DiT forward
  + 3/4-step sampler + load JAX weights. Solaris-JAX -> reference/weights/validation only.
- Cost: reimplementation + weight-loading + numeric validation (days). Beyond "kernels", but the
  interactive harness is being built fresh anyway -> build it PyTorch-native from day one.

## Plan if approved
- Short-term: keep JAX-Solaris to PROVE levers fast (no reimpl).
- Medium-term: PyTorch DiT inference; drop in FA4 (attn), FP8/NVFP4 GEMM (FFN/QKV), fused CuTe AdaLN.
- AdaLN is memory-bound + small DiT share -> modest; the big server levers are FP8/NVFP4 GEMM + FA4.

## Done so far
- kernels/adaln_fused_cute.py : fused AdaLN (LN+modulate+gate+residual) CuTe DSL kernel + torch ref.
  Needs box compile + validation. Targets the PyTorch path (FFI to use from JAX).
