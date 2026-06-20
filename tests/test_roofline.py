"""Sanity checks for the budget model (not a profiler — just guards the arithmetic)."""

from oasis_forge.roofline import OASIS_DMD, budget


def test_more_gpus_faster():
	assert budget(n_gpus=8).fps > budget(n_gpus=1).fps


def test_camera_band_helps_and_attention_binds():
	dense = budget(camera_band=1.0)
	banded = budget(camera_band=1 / 3)
	assert banded.fps > dense.fps
	# attention is the binding kernel at ship shapes
	assert max(dense.kernels, key=lambda k: k.roofline_s).name == "attention"


def test_smaller_window_faster():
	assert budget(window_frames=8).fps > budget(window_frames=21).fps


def test_single_h100_cannot_hit_realtime_even_optimized():
	# fully-levered single H100 still misses 30fps -> per-user cluster must be multi-GPU
	r = budget(n_gpus=1, camera_band=1 / 3, fp8_ffn=True, window_frames=8)
	assert r.fps < 30.0
