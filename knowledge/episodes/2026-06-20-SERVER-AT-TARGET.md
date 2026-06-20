# 2026-06-20 — SERVER latent-gen is AT TARGET (25.7 fps, 3-step)

Definitive measurement: instrumented base_mp_runner.py to skip VAE decode (client-side) and
time the latent rollout (warmup-then-timed, compile excluded). B300 GPU0, 2-player, 3 steps.

  LATENT_GEN_SECONDS = 9.9953 for 257 frames  ->  **25.71 fps**   [target 20-30]  AT TARGET.

## Why this reframes everything
- The full-pipeline 1.96 fps that drove the "10x off" urgency was ~83% **VAE decode = CLIENT-side**.
- Server's real job (DiT latent-gen) was always ~17% -> 25.7 fps. We were optimizing the client's bottleneck.
- The **3-step reduction (operator call) crossed the line**: 3-step 25.7 fps; 4-step ~=19 fps (just under).

## Implications
- Server baseline target is essentially MET for latent generation on one B300.
- Remaining DiT kernel work (Pallas flash attn, FP8 FFN GEMM) = MARGIN + player-scaling headroom, NOT a blocker.
- cuDNN one-liner attention flash failed in JAX (UnexpectedTracerError); Pallas flash is the clean lever if more DiT headroom is wanted.
- Open: 3-step QUALITY (model distilled for 4 steps) — validate video quality at 3 steps (model-quality question, separate from kernels).
- VAE (client) already helped by cuDNN 9.23 (38->0 conv fallbacks, 2.11x) for whoever runs decode.
