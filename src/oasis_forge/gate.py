"""Two-tier drift gate.

The crux of the whole harness. A ported kernel must clear both tiers before it can be
kept, because for an autoregressive world model a kernel can pass an isolated allclose
and still drift the scene to mush over a long rollout.

  tier 1 (per-kernel):  allclose(port_output, golden) at the tier's tolerance.
                        cheap; runs in the inner loop on every candidate.
  tier 2 (trajectory):  run an N-frame autoregressive rollout with the ported kernel
                        swapped in, compare the latent trajectory against a reference
                        rollout. mean per-frame latent MSE must stay <= drift_budget.

Arrays may be numpy arrays or torch tensors; anything `.detach().cpu().numpy()`-able or
already numpy works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import PortConfig, PortTier


def _to_np(x: Any) -> np.ndarray:
	if isinstance(x, np.ndarray):
		return x
	# torch tensor duck-typing — avoid importing torch on CPU-only boxes.
	if hasattr(x, "detach"):
		return x.detach().cpu().float().numpy()
	return np.asarray(x, dtype=np.float32)


@dataclass
class Tier1Result:
	passed: bool
	max_abs_err: float
	max_rel_err: float


@dataclass
class Tier2Result:
	passed: bool
	mean_latent_mse: float
	per_frame_mse: list[float]
	perceptual: float | None


def tier1_correctness(
	port_output: Any,
	golden: Any,
	tier: PortTier,
	cfg: PortConfig,
) -> Tier1Result:
	"""Per-kernel allclose vs the high-precision golden output."""
	p = _to_np(port_output).astype(np.float64)
	g = _to_np(golden).astype(np.float64)
	if p.shape != g.shape:
		return Tier1Result(False, float("inf"), float("inf"))
	rtol, atol = cfg.tolerances(tier)
	abs_err = np.abs(p - g)
	max_abs = float(abs_err.max()) if abs_err.size else 0.0
	denom = np.maximum(np.abs(g), 1e-12)
	max_rel = float((abs_err / denom).max()) if abs_err.size else 0.0
	passed = bool(np.allclose(p, g, rtol=rtol, atol=atol))
	return Tier1Result(passed, max_abs, max_rel)


def tier2_trajectory(
	port_latents: Sequence[Any],
	reference_latents: Sequence[Any],
	cfg: PortConfig,
	perceptual: float | None = None,
) -> Tier2Result:
	"""Compounding-drift gate over an N-frame autoregressive rollout.

	port_latents / reference_latents: per-frame latent tensors from two rollouts driven
	by the SAME action timeline — one with the ported kernel, one with the reference. The
	only difference between the trajectories must be the kernel under test.
	"""
	n = min(len(port_latents), len(reference_latents))
	per_frame: list[float] = []
	for i in range(n):
		a = _to_np(port_latents[i]).astype(np.float64)
		b = _to_np(reference_latents[i]).astype(np.float64)
		per_frame.append(float(np.mean((a - b) ** 2)))
	mean_mse = float(np.mean(per_frame)) if per_frame else float("inf")

	passed = mean_mse <= cfg.drift_budget_latent_mse
	if cfg.drift_budget_perceptual is not None and perceptual is not None:
		passed = passed and (perceptual <= cfg.drift_budget_perceptual)
	return Tier2Result(passed, mean_mse, per_frame, perceptual)
