# 2026-06-20 — KV Craft first run on B300: JAX works, Blackwell XLA pain points = our targets

Ran `src/inference.py experiment_name=solaris device.eval_num_samples=1` on B300 GPU 0 (JAX
2.12/cuda13 path, env on NFS). Confirms the model runs on Blackwell AND reveals exactly where
XLA leaves perf on the table on sm_100 — which is exactly what we optimize.

## Confirmed
- **JAX/KV Craft runs on Blackwell B300.** Loaded clip.pt, vae.pt (245 MB), solaris.pt (6.7 GB);
  loaded eval datasets (structure/rotation/both_look_away/one_looks_away; 257 frames each);
  entered `rollout_func` generation. No fatal error. Only harmless int64->int32 dtype warnings.
- **The long silence is COMPILE, not a hang:** GPU0 util 0% while 206 GB allocated (JAX preallocs
  ~75% of the 275 GB). XLA autotuning is CPU-bound; the main compute hasn't run yet.

## Blackwell XLA pain points (our optimization targets, by where XLA struggles)
1. **VAE decode convs — cuDNN heuristics FAIL on sm_100.** 38x `conv_algorithm_picker: None of
   the algorithms provided by cuDNN heuristics worked; trying fallback algorithms` for the VAE
   upsampling convs (e.g. bf16[2,384,1,88,160] 3x3x3; bf16[2,352,640,96] 3x3; layouts bf012 / b01f).
   => cuDNN lacks good Blackwell conv algos -> exhaustive fallback search (slow compile) and likely
   slow fallback kernels at runtime. **VAE-decode conv is a confirmed B300 kernel target.** Likely
   needs NHWC/channels-last layout, forced algo, or a Pallas-GPU conv.
2. **GEMM fusion autotuner precision mismatch** — 3x `gemm_fusion_autotuner: Results do not match
   the reference` on Blackwell (XLA rejecting bad GEMM candidates). Watch GEMM correctness on sm_100.
3. **Attention** (from code, src/models/transformer.py): `jax.nn.dot_product_attention` with NO
   `implementation='cudnn'` -> default XLA lowering, not cuDNN flash. #1 lever.

## So the optimization shortlist (B300, profiler-to-confirm)
- Attention: default XLA -> `implementation='cudnn'` (cuDNN flash). Identical numerics.
- VAE decode conv: fix the Blackwell conv path (layout / algo / Pallas). XLA literally can't find a
  good cuDNN algo by heuristic today.
- GEMM (FFN/QKV): verify correctness, then FP8.

## Pending
- First video / DONE_INFER (long watch armed -> FIRST_VIDEO_AT marker). Then: warm-compile-cache
  re-run to measure clean steady-state **fps** (compile excluded) for the gains chart.
