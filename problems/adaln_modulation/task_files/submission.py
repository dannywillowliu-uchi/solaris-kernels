"""Candidate AdaLN kernel. THIS is what the agent optimizes.

Default == torch.compile (Inductor) of the eager chain: a legal, zero-effort baseline that
fuses the pointwise ops + LayerNorm into far fewer HBM passes. The agent can push further with
a hand-written Triton fused kernel (flash-attn ops/triton/layer_norm.py template) targeting the
sm_90 HBM roofline. Must match ref_eager.
"""

from __future__ import annotations

import torch

from reference import ref_eager

custom_kernel = torch.compile(ref_eager)
