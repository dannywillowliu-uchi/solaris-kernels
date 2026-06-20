"""Candidate camera-banded attention kernel. THIS is the file the agent optimizes.

Default == naive banded (a Python loop of SDPA per camera group). Correct but unoptimized:
3 separate kernel launches, no batching, no fusion. The agent's job is to make this fast on
the target arch while still matching ref_banded — e.g.:
  - batch the 3 groups into one varlen / cu_seqlens FlashAttention call
  - FlexAttention with a camera block_mask (BACKEND=FLASH) under torch.compile
  - a single block-diagonal kernel that never computes cross-camera scores

Must keep the signature: custom_kernel(q, k, v, gq, gk) -> out  [B, H, Sq, D].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def custom_kernel(q, k, v, gq, gk):
	outs = []
	aq = ak = 0
	for cq, ck in zip(gq, gk):
		outs.append(F.scaled_dot_product_attention(
			q[..., aq:aq + cq, :], k[..., ak:ak + ck, :], v[..., ak:ak + ck, :]))
		aq += cq
		ak += ck
	return torch.cat(outs, dim=-2)
