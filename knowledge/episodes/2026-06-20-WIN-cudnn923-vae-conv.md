# 2026-06-20 — WIN #1: cuDNN 9.10->9.23 fixes Blackwell VAE conv (2.11x)

Top bottleneck was VAE 3D conv (~83% GPU time) on cuDNN's generic fallback because cuDNN 9.10
heuristics found NO sm_100 conv algo (38 "None of the algorithms" warnings).

Fix: `uv pip install --python <venv> "nvidia-cudnn-cu12>=9.11"` -> 9.23.2.1 (ABI-compatible within
cuDNN 9; jax 0.6.2 unchanged). Clear/repoint JAX compile cache to force re-autotune.

Result (warm-vs-warm, eval_structure 257f, 2-player, B300 GPU0):
- conv fallbacks: 38 -> 0
- fps: 1.96 -> 4.145  =  2.11x   (gen 131s -> 62s)
- correctness: cuDNN algorithm change = same conv math (formal SSIM gate vs golden = TODO).

Transfers to B200 (sm_100, primary Blackwell target in cuDNN — likely even cleaner). NOT to H100 (Hopper).
Reproduce: bump cuDNN in the venv + fresh JAX_COMPILATION_CACHE_DIR.
