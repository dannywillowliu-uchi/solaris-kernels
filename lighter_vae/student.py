"""Light VAE decoder = Solaris Decoder3d with a slim config (same latent space => drop-in).

The heavy teacher is Decoder3d(dim=128, dim_mult=[1,2,4,4], num_res_blocks=2). The student keeps
the SAME z_dim / VAE_SCALE (identical latent interface) but cuts the FLOPs:
  dim 128->64, dim_mult [1,2,4,4]->[1,2,2,2] (kill the 512-ch high-res stages), num_res_blocks 2->1.
Run inside the Solaris repo env (imports src.models.wan_vae).
"""

from __future__ import annotations

from src.models.wan_vae import Decoder3d  # reuse the exact arch; only the config is slimmed

# Tunable student configs, lightest first. Pick by the quality/speed Pareto in distill.py.
STUDENT_CONFIGS = {
	"s1_half": dict(dim=64, dim_mult=[1, 2, 2, 2], num_res_blocks=1, attn_scales=[]),
	"s2_mid": dict(dim=96, dim_mult=[1, 2, 2, 2], num_res_blocks=1, attn_scales=[]),
	"s0_min": dict(dim=48, dim_mult=[1, 2, 2], num_res_blocks=1, attn_scales=[]),
}


def build_student(z_dim, rngs, config="s1_half"):
	"""Instantiate a slim Decoder3d student. z_dim MUST match the teacher (latent interface)."""
	c = STUDENT_CONFIGS[config]
	# NOTE: confirm Decoder3d's exact arg order against src/models/wan_vae.py:481 before training.
	return Decoder3d(
		dim=c["dim"],
		z_dim=z_dim,
		dim_mult=c["dim_mult"],
		num_res_blocks=c["num_res_blocks"],
		attn_scales=c["attn_scales"],
		rngs=rngs,
	)
