"""AdaLN modulation — reference + synthetic inputs (correctness-preserving fusion target).

Unlike banding, this changes NOTHING semantically: it's the exact AdaLN-block elementwise
chain (LayerNorm -> modulate -> gated residual, twice per block), just a candidate that fuses
the HBM round-trips. Same math, always legal, real speedup. Bandwidth-bound, so it's the
cleanest cross-SM port story too (H200 > H100 > B300 by HBM ratio).

ref_eager is the golden (multiple HBM passes). The candidate (submission) must match it and
beat its time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BENCH = [
	dict(name="tiny", B=1, S=256, D=128, seed=0),
	dict(name="frame", B=1, S=4608, D=5120, seed=1),
	dict(name="chunk", B=1, S=13824, D=5120, seed=2),
	dict(name="full", B=1, S=36864, D=5120, seed=3),  # full ship chunk tokens
]


def generate_input(B, S, D, seed, device="cuda", dtype=torch.bfloat16):
	g = torch.Generator(device=device).manual_seed(seed)

	def r(*shp):
		return torch.randn(*shp, device=device, dtype=dtype, generator=g)

	x = r(B, S, D)
	mod = [r(B, 1, D) * 0.1 for _ in range(6)]  # shift/scale/gate for msa + mlp
	y_attn, y_mlp = r(B, S, D), r(B, S, D)  # sublayer outputs for the gated residuals
	return (x, *mod, y_attn, y_mlp)


def ref_eager(x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, y_attn, y_mlp):
	d = x.shape[-1]
	m1 = F.layer_norm(x, (d,)) * (1 + scale_msa) + shift_msa
	x = x + gate_msa * y_attn
	m2 = F.layer_norm(x, (d,)) * (1 + scale_mlp) + shift_mlp
	x = x + gate_mlp * y_mlp
	return x, m1, m2
