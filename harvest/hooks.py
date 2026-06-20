"""Call-site instrumentation — STUB until an Oasis-500M rollout runs on a box.

The harvest layer is what makes the port real-shape, real-distribution instead of
synthetic: it captures the actual tensors flowing through the three hot kernels during a
rollout, plus a high-precision golden output to gate against.

Plan once a rollout exists:

  1. Load Oasis-500M (etched-ai/open-oasis + camenduru/oasis-500m): DiT + ViT-VAE.
  2. Register forward hooks on the three call sites:
       - ST-DiT attention module(s)
       - the AdaLN modulation op inside each DiT block
       - the VAE decoder forward
  3. Run a short rollout with a fixed action timeline + seed. At representative DMD/diffusion
     steps (capture both an early and a late sigma — they see very different signal),
     record (args, kwargs) inputs.
  4. Recompute each captured call in FP32/BF16 to get the GOLDEN output (the port is gated
     against this high-precision result, NOT the FP16 production path).
  5. Save per problem to problems/<kernel>/task_files/golden.npz:
       inputs_*  : the real input tensors
       golden    : high-precision output
       meta      : shape, dtype, step/sigma, action-context hash

The reference + ledger code already consume golden.npz; only this capture is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Real call sites, verified against the cloned etched-ai/open-oasis source (see
# knowledge/semantic/model_facts.md). i = block index 0..15 (DiT_S_2 -> 16 blocks).
# NOTE: 500M attention is AXIAL (s_attn + t_attn separate) + torch SDPA, head_dim 64; the
# 14B ship model is full-3D head_dim 128 — different kernel. VAE here is a ViT transformer
# (SDPA), NOT the Wan-VAE conv3d. Capture proves the harness + AdaLN port, not attn/VAE transfer.
CALL_SITES: dict[str, list[str]] = {
	# Two axial attentions per block. s_attn = spatial/bidirectional/144 keys; t_attn = temporal/causal/<=32 keys.
	"stdit_attention": ["DiT.blocks[i].s_attn", "DiT.blocks[i].t_attn"],
	# SiLU+Linear(hidden, 6*hidden); hook the Linear to get the raw 6*hidden pre-split tensor.
	"adaln_modulation": ["DiT.blocks[i].s_adaLN_modulation", "DiT.blocks[i].t_adaLN_modulation"],
	# decode is a METHOD (AutoencoderKL.decode), not an nn.Module.forward — wrap it. Sub-modules:
	# VAE.post_quant_conv, VAE.decoder[j].attn (SDPA), VAE.predictor.
	"vae_decode": ["AutoencoderKL.decode"],
}

# Sampler reminder: open-oasis is DDIM 10-step (no DMD). Capture at an EARLY and a LATE step;
# they see very different signal. The ship model is DMD 4-step iterative [~999,749,499,249].


@dataclass
class Capture:
	problem: str
	inputs: dict[str, Any] = field(default_factory=dict)
	golden: Any = None
	meta: dict[str, Any] = field(default_factory=dict)


def register_hooks(model: Any, problems: list[str]) -> Any:
	"""Attach forward hooks for the given problems. Returns a handle to collect captures."""
	raise NotImplementedError(
		"Implement against a loaded Oasis-500M. See module docstring for the 5-step plan. "
		"Resolve CALL_SITES from etched-ai/open-oasis first."
	)


def save_golden(capture: Capture, out_path: str) -> None:
	"""Write a Capture to problems/<kernel>/task_files/golden.npz."""
	raise NotImplementedError("np.savez(out_path, **inputs, golden=..., meta=...)")
