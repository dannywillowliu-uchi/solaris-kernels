"""Camera-banded sparse attention — reference + synthetic inputs (model-independent).

The ship model's spatial grid is 64x288 = 3 horizontally-tiled cameras (left|front|right),
96 cols each. The hypothesis (to validate on real golden) is that cross-camera attention
weight is low, so attention can be pruned to block-diagonal over the 3 camera groups — a ~1/3
FLOP cut that a generic dense FA3 port can't see. This file lets us develop + benchmark that
kernel NOW, on synthetic tensors at ship shapes, before any model is available.

Two references:
  ref_dense  -- full all-to-all SDPA. The SPEED BASELINE (what you'd run without banding) and
                the thing banding approximates.
  ref_banded -- block-diagonal SDPA over camera groups. The CORRECTNESS GOLDEN the ported
                kernel must match. Default submission == this, unoptimized.

Camera tokens are laid out contiguously here (3 clean blocks = best case). Real layout is
interleaved by row, so a production kernel needs a gather/reorder — noted, not modeled in v1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N_CAM = 3

# Streaming shapes: query = current chunk; key = cached window + chunk. tiny is CPU-checkable.
BENCH = [
	dict(name="tiny", B=1, H=2, Sq=96, Sk=96, D=16, seed=0),
	dict(name="frame", B=1, H=40, Sq=4608, Sk=4608, D=128, seed=1),  # 1 latent frame
	dict(name="chunk", B=1, H=40, Sq=13824, Sk=13824, D=128, seed=2),  # 3-frame chunk self-attn
	dict(name="stream", B=1, H=40, Sq=13824, Sk=110592, D=128, seed=3),  # chunk vs 21+3 window
]


def camera_groups(seq: int, n_cam: int = N_CAM) -> list[int]:
	"""Contiguous, near-equal camera group sizes summing to seq."""
	base = seq // n_cam
	sizes = [base] * n_cam
	sizes[-1] += seq - base * n_cam
	return sizes


def generate_input(B, H, Sq, Sk, D, seed, device="cuda", dtype=torch.bfloat16):
	g = torch.Generator(device=device).manual_seed(seed)
	scale = (D ** -0.25)  # keep logits sane at large D
	q = torch.randn(B, H, Sq, D, device=device, dtype=dtype, generator=g) * scale
	k = torch.randn(B, H, Sk, D, device=device, dtype=dtype, generator=g) * scale
	v = torch.randn(B, H, Sk, D, device=device, dtype=dtype, generator=g)
	return q, k, v, camera_groups(Sq), camera_groups(Sk)


def ref_dense(q, k, v, gq=None, gk=None):
	"""Full all-to-all attention. Speed baseline."""
	return F.scaled_dot_product_attention(q, k, v)


def ref_banded(q, k, v, gq, gk):
	"""Block-diagonal attention over camera groups. Correctness golden."""
	outs = []
	aq = ak = 0
	for cq, ck in zip(gq, gk):
		outs.append(F.scaled_dot_product_attention(
			q[..., aq:aq + cq, :], k[..., ak:ak + ck, :], v[..., ak:ak + ck, :]))
		aq += cq
		ak += ck
	return torch.cat(outs, dim=-2)
