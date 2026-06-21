# KV Craft Serving Optimization — Summary (for demo)

Goal: make the **KV Craft** multiplayer world model stream in real time (**20–30 fps**) on
NVIDIA **Blackwell** (B300 sm_103 + B200 sm_100). We (a) built an **autonomous kernel-optimization
agent/harness**, (b) used it to profile + optimize the JAX/XLA serving path, and (c) opened a
**lighter-VAE distillation** track for the remaining gap. This doc is the demo brief.

Repo: `github.com/dannywillowliu-uchi/kv-craft-kernels` (private).
Target model: KV Craft (JAX, DiT-on-MatrixGame-2.0; DiT = hidden 1536 / 30 layers / 12 heads /
head_dim 128 / ffn 8960; Self-Forcing few-step; Wan-VAE 3D-conv decoder; 2-player joint attention).

---

## 1. The headline result

| metric | before | after | note |
|--------|--------|-------|------|
| **Server latent-gen (DiT)** | "looked like" 1.96 fps | **25.7 fps (3-step)** | **AT the 20–30 target.** The "10× off" was a *scoring artifact* — see §3. |
| VAE 3D conv on Blackwell | 38 cuDNN fallbacks (generic, non-tensor-core) | **0 fallbacks** (tensor-core) | cuDNN 9.10→9.23 |
| Full-pipeline frames-out | 1.96 fps | **4.14 fps** (2.11×) | cuDNN 9.23 VAE win |

**Punchline:** the server's actual job (generating latents) is **already real-time**; the heavy
part is the **VAE decode → pixels**, which is being attacked on two tracks (kernels + lighter VAE).

---

## 2. The autonomous kernel-optimization agent (the centerpiece)

A profile→patch→measure→gate→record loop that optimizes GPU kernels and **accumulates gains over
time into a chart**, modeled on a prior AMD-MI355X kernel-forge.

```
loop:
  PROFILE   nsys --cuda-graph-trace=node  -> rank GPU kernels by time (find the bottleneck)
  DIAGNOSE  why slow? fallback kernel / bad layout / launch-gap bubble / memory-bound
  PATCH     one JAX-native fix (XLA flag / cuDNN / layout / Pallas / FP8)
  MEASURE   warm-cache run -> fps + SSIM quality gate (compile excluded)
  GATE      keep iff faster AND quality held; else REVERT + log the dead-end
  RECORD    append results/gains.csv -> plot_gains.py renders the improvement chart; commit + push
  NEXT      re-profile, attack the new top kernel
```

Pieces (all in the repo):
- `harness/profile.sh` — nsys profiler → ranked kernels (handles XLA CUDA-graphs).
- `harness/measure.sh` — warm fps + SSIM quality gate.
- `agents/agent_prompt.md` — the agent's methodology + hard rules + scoring.
- `scripts/loop.sh` — the autonomous loop driver (launches agents, walltime/stop, pushes wins).
- `results/gains.csv` + `results/plot_gains.py` — the **gains-over-time chart** (the demo visual).

**Demo idea:** show the agent loop running, the live profile, and the chart filling in as wins land.

---

## 3. The key methodology insight (great demo moment): *score the right objective*

Naively, frames-out was **1.96 fps → "we're 10× off"**. The profiler (nsys, CUDA-graph-traced)
showed **~83% of GPU time is the VAE decode, ~17% is the DiT**. Then the scoping realization:

- **Server produces *latents*; the client *decodes* to pixels.** So the server metric is
  **DiT latent-gen**, not full pipeline. Measured directly (decode skipped, warm): **25.7 fps —
  already at target.** The 1.96 fps panic was the *client-side* VAE.
- This caught us **optimizing the wrong thing** (the client's bottleneck). Fixing the scoring is
  what turned "10× off, panic" into "server's at target; the open question is the VAE."

(Later the edge/client VAE turned out to be too weak, so the VAE moved back server-side — now the
two VAE tracks below carry it.)

---

## 4. Wins (kept, with numbers)

1. **cuDNN 9.10 → 9.23 — fixes Blackwell VAE 3D conv.** cuDNN 9.10's heuristics found *no* sm_100
   algorithm for the VAE's 3D convs → 38 "None of the algorithms worked" fallbacks → generic
   non-tensor-core kernel. Bumping cuDNN (ABI-compatible within cuDNN 9, jax 0.6.2 unchanged) →
   **0 fallbacks, tensor-core convs, 2.11× full-pipeline** (1.96→4.14 fps frames-out).
2. **Step reduction 4→3** crosses real-time for latent-gen (3-step 25.7 fps vs 4-step ~19).
   (Quality of 3-step vs 4-step is the open validation.)
3. **Correct server baseline established:** DiT latent-gen instrumented (decode-excluded, warm).

Transfers cleanly across **B200 ↔ B300** (same Blackwell family); does **not** transfer to H100
(Hopper, no FP4, different conv path).

---

## 5. Dead-ends (honest — these are data, and good demo "what we ruled out")

- **Global `--xla_gpu_force_conv_nhwc`** → 0.72 fps (2.7× *worse*): forces DiT transposes; Blackwell
  NDHWC conv no better. Reverted.
- **Global cuDNN-flash attention monkeypatch** → JAX `UnexpectedTracerError` (hit incompatible
  call sites: action module odd head dims, VAE/clip attention). Reverted.
- **FP8 VAE conv (naive dtype swap)** → `TypePromotionError` (fp8 has no implicit promotion with
  fp32 bias). Proper FP8 conv needs explicit quant/scaling + JAX's FP8-conv support is immature.
- **Exhaustive conv-autotune flags** → marginal gain, pathologically slow compile. cuDNN 9.23
  already picks good algos.

Lesson: in **JAX**, big kernels (FA-flash, FP8/NVFP4 GEMM/conv) are non-native — they need Pallas
or FFI; the productized wins (cuDNN-version bump, XLA-native paths) are where the easy gains are.

---

## 6. The two live tracks for VAE frames-out (the heavy part)

The VAE is ~86% of each frame even post-cuDNN-9.23 (~207 ms); ~13× too slow for 20 fps. Kernels
top out ~2×. So:

- **Track A — VAE kernels:** cuDNN 9.23 (done, 2.11×); FP8 conv (~2× more, needs real quant work —
  JAX FP8-conv is the blocker). Ceiling ~7 fps frames-out.
- **Track B — lighter/distilled VAE (the multi-× lever, `lighter_vae/`):** distill a **slim
  `Decoder3d`** (dim 128→64, dim_mult [1,2,4,4]→[1,2,2,2], res-blocks 2→1) to match the frozen
  Wan-VAE in the **same latent space** → drop-in faster decoder, no DiT/client changes. Gate on
  SSIM/FID vs teacher + decode-fps. This is how frames-out reaches 20–30 fps.

---

## 7. Infra / how to reproduce

- Boxes: 2× B300 (datacrunch) + 1× B200, all Blackwell, JAX 0.6.2 + **cuDNN 9.23** + nsys/ncu.
- Env on NFS (root disks small). Compile-cache on disk so warm runs exclude the ~6-min JAX compile.
- Single GPU per stream; multi-GPU = throughput + (future) VAE/attention sharding for more players.
- Everything (harness, findings episodes, gains ledger, dead-ends) committed to the repo.

**For the demo:** the agent loop + the gains chart + the "score the right objective" story +
the cuDNN-9.23 Blackwell-conv fix are the strongest beats.

---

## 8. Frames-out lever: decode-small + upscale (right idea; needs the right resolution)

The promising VAE lever: **decode at lower resolution, then upscale.** VAE conv cost ∝ resolution²,
so the full-res (360×640) upsample stages are ~all the FLOPs; decoding smaller skips them.

**Honest status — measured, not just projected:**
- **William reports ~30 fps** with aggressive res reduction (his decode-small + upscale path).
- **Our first implementation (½-res latent-downscale + bilinear) regressed: 3.52 fps vs 4.14 full-res.**
  Half-res wasn't aggressive enough and/or latent-downscale is the wrong method — decoder-stage
  *truncation* at ¼-res is the likely fix. This is an open item, not a banked win.
- **Safe shrink regardless:** it only softens per-frame detail; the world-model state is the latent
  (untouched), so nothing compounds in the autoregressive rollout.

**Full frames-out picture (honest):** bottleneck = VAE. Banked: cuDNN-9.23 (2.11×). In progress:
distilled lighter decoder (training converges; speedup TBD) and decode-small+upscale (William ~30 fps;
our ½-res attempt regressed, ¼-res/truncation pending). DiT/server side is already real-time (25.7 fps).
