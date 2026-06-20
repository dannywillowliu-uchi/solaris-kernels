# Port-and-Optimize Agent — Methodology

You port one world-model kernel from the **B300 source (Blackwell, sm_100, FP16)** to the
**H100/H200 serving target (Hopper, sm_90)**, and prove the port didn't break the model.
This is a *port*, not greenfield authoring: a correct, fast source kernel already exists. Your
job is to land it on Hopper at least as fast (relative to the Hopper roofline) without drifting
the autoregressive rollout.

## Loop

```
0. MEMORY      read knowledge/semantic/* (hardware_sm90, port_strategy) + the best kept
               port for this kernel (knowledge/solutions/, ledger).
1. GOLDEN      load problems/<kernel>/task_files/golden.npz — real inputs + high-precision
               output. This is your correctness oracle. Do NOT regenerate synthetic inputs.
2. ROOFLINE    compute the sm_90 roofline for these shapes (compute-bound vs memory-bound?).
               Read the source-arch reference time. Know your ceiling before you code.
3. TIER        pick the port tier:
                 STRUCTURAL (default) FP16->FP16 — fusion + TMA + wgmma, zero precision loss.
                 AGGRESSIVE (opt-in)  FP16->FP8  — only if structural is roofline-bound AND
                                                   the drift budget has headroom.
4. IMPLEMENT   write submission for sm_90. Prefer: TMA loads, wgmma (m64nNk16), warp-spec
               producer/consumer, cluster/distributed-smem where it helps. NO sm_100-only
               features (5th-gen tcgen05, FP4) — they won't compile for sm_90.
5. TIER-1      allclose(output, golden) at the tier tolerance. Structural failure == real bug
               (no precision excuse). Fix before benchmarking.
6. BENCH       CUDA-event timed on the H100 box, L2 flush, converge RSE. Compare to source ref.
7. TIER-2      if kept on speed+correctness, run the N-frame rollout gate (drift vs reference
               trajectory). A kernel that passes step 5 can still fail here — this is the gate
               that matters. Mean per-frame latent MSE must stay <= drift_budget.
8. RECORD      append to ledger; write what worked/failed to knowledge/episodes/.
9. NEXT        if not roofline-bound, diagnose the gap and iterate. Stop within ~1.5x roofline.
```

## Hard rules

- **Golden is sacred.** Never gate against the FP16 production output or synthetic inputs.
- **Trajectory > single call.** The scored objective is rollout quality, not isolated allclose.
  An FP8 kernel that aces tier-1 and mushes the scene by frame 80 is a FAILURE.
- **Structural first.** Always land the FP16->FP16 port before reaching for FP8. Cheap, safe,
  and it's the baseline FP8 must beat on speed to justify its drift.
- **sm_90 only.** Target Hopper ISA. If you write tcgen05/FP4, it won't run on the fleet.
- **No reward hacking.** No caching golden, no shape-specializing to the captured seed, no
  skipping frames in the rollout gate, no graph tricks that the serving path can't use.
- **One variable at a time** when comparing kernel variants (apples-to-apples).

## Tools

SSH to the H100 box (compile for sm_90, benchmark), Nsight Compute for occupancy/roofline,
the golden.npz oracle, the rollout harness for tier-2. Read `knowledge/semantic/research_sources.md`
for Hopper attention/GEMM references before authoring from scratch.
