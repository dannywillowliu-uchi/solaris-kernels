# KV Craft — serving-kernel optimization

Make the **KV Craft** multiplayer world model stream in real time on NVIDIA **Blackwell** (B300/B200).

We profile the model's hot path on a B300, optimize the GPU kernels (JAX/XLA), and accumulate the
gains over time into a chart. **Scope = the kernels + the serving VAE**; collaborators own the
interactive harness, netcode, and systems design.

## Headline results

| metric | result |
|--------|--------|
| Server latent-gen (DiT, 3-step) | **25.7 fps** — real-time |
| VAE 3D conv on Blackwell (cuDNN 9.10 → 9.23) | **2.11×** (0 fallbacks, tensor-core) |
| Current bottleneck | **VAE decode → pixels** (~83% of each frame) |

The server's job (generating latents) is already real-time; the heavy part is the VAE decode, which
is being attacked three ways: a **distilled lighter decoder**, **decode-small + upscale**, and a
**client-side Metal decoder**. Full writeup + the gains chart: **[`DEMO_SUMMARY.md`](DEMO_SUMMARY.md)**.

## Layout

| path | what |
|------|------|
| `harness/` | autonomous **profile → measure → gate** loop (`profile.sh`, `measure.sh`) |
| `agents/agent_prompt.md` | the optimization agent's methodology + rules |
| `scripts/loop.sh` | the autonomous loop driver |
| `lighter_vae/` | slim VAE decoder **distillation** (the multi-× frames-out lever) |
| `edge/` | macOS **Metal** VAE decoder (client-side decode) |
| `src/oasis_forge/` | split-serving protocol bridge (`streaming.py`, `cli.py`) |
| `knowledge/` | findings (`episodes/`) + verified model facts (`semantic/`) |
| `results/` | gains ledger (`gains.csv`) + improvement chart (`plot_gains.py`) |
| `DEMO_SUMMARY.md` | the demo brief — read this first |

## Use (on the B300 box)

```bash
GPU=0 bash harness/profile.sh                     # rank GPU kernels (nsys)
GPU=0 bash harness/measure.sh                     # frames-out fps + SSIM quality gate
python results/plot_gains.py                      # render the improvement chart
```

## Hardware / stack

B300 (sm_100) and B200 — Blackwell. JAX 0.6.2 + **cuDNN 9.23** (9.10 lacks Blackwell 3D-conv algos).
Optimizations transfer B200 ↔ B300 (same family); not to H100 (Hopper, different conv path).
