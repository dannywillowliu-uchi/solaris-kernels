# Solaris autokernel harness

Autonomous profile → patch → measure → gate → record loop that speeds up Solaris generation on
the B300 without degrading video quality. Ported from `amd-kernel-forge`, adapted to Solaris/JAX.
The gains-over-time chart is the demo artifact; it grows from `results/gains.csv` automatically.

## Pieces

| file | role |
|------|------|
| `profile_solaris.sh` | (on box) nsys `--cuda-graph-trace=node` over warm generation → ranked GPU kernels. Finds the slowness. |
| `measure_solaris.sh` | (on box) warm-cache generation → **fps** + **SSIM quality gate** vs golden video. Did the patch help without breaking quality? |
| `../agents/solaris_agent_prompt.md` | the optimization agent's methodology + hard rules + patch space. |
| `../scripts/solaris_loop.sh` | the autonomous loop: profile → launch agent with fresh profile + ledger → agent patches/measures/records → re-launch. Walltime + stop-file. |
| `../results/gains.csv` + `plot_gains.py` | ledger of KEPT wins → the improvement chart. |

## Run it

```bash
# one-shot tools (on the box, GPU 0)
GPU=0 bash profile_solaris.sh                       # rank kernels
GPU=0 GOLDEN=<baseline.mp4> bash measure_solaris.sh  # fps + quality

# autonomous loop (local; launches claude -p agents, pushes each win)
./scripts/solaris_loop.sh 12        # 12h; touch .solaris_stop to halt
python results/plot_gains.py --target solaris   # the chart
```

## Discipline (enforced by the agent prompt)

- Measure **warm** (compile excluded). - **Quality-gate** every change (SSIM vs golden). - **Revert
regressions**, log dead-ends. - **One variable per run.** - **GPU 0 only.** - Keep attention BF16.
- **Push every kept win** (commit + `gains.csv` row) so the chart grows live.

## Current state (2026-06-20)

- Baseline: **1.96 fps** (2-player, B300, warm).
- Profile: **VAE 3D conv ≈ 83% of GPU time** (cuDNN heuristic fallback on Blackwell, non-tensor-core).
- Dead-end: global `--xla_gpu_force_conv_nhwc` → **0.72 fps (reverted)** — forces DiT transposes,
  Blackwell NDHWC conv no better.
- Next candidates for the VAE conv: scoped NDHWC (VAE only), pinned cuDNN algo, cuDNN bump, Pallas-GPU conv.
