# Harvest

Turns a live rollout into frozen, gateable kernel problems. This is the part that does
**not** exist in greenfield kernel benchmarks (KernelBench hands you the reference) and is
the real engineering coupling point with the serving stack.

## What it produces

For each hot kernel, `problems/<kernel>/task_files/golden.npz`:

- `inputs_*` — the real input tensors captured at the call site during a rollout
- `golden` — the **high-precision (FP32/BF16) recomputation** of that call's output
- `meta` — shape, dtype, diffusion step / sigma, action-context hash

## Why golden is high-precision, not the FP16 path

The port is gated against a high-precision reference so that:
- the **structural FP16→FP16** port's only allowed error is float reassociation, and
- the **aggressive FP16→FP8** port is measured against truth, and its drift can be compared
  to the FP16 source deployment's own drift (the `drift(fp8) <= drift_budget` inequality).

If golden were the FP16 production output, an FP8 port that happened to match FP16's *errors*
would look correct while being wrong.

## Capture both ends of the sigma schedule

DMD/diffusion steps see very different signal at high vs low sigma. Capture at least one early
and one late step per kernel so the port is tuned on the real input distribution, not just the
clean-latent end.

## Prototype vs ship

- **Oasis-500M**: hooks attach to `etched-ai/open-oasis` modules. Validates the structural
  port end-to-end (FP16 matches FP16).
- **Internal 14B**: same hooks, re-pointed at the production stack's call sites. Only here do
  the FP8 numerics mean anything.
