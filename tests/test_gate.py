"""CPU-testable checks for the parts that don't need hardware: gate + ledger."""

import numpy as np

from oasis_forge.config import PortConfig, PortTier
from oasis_forge.gate import tier1_correctness, tier2_trajectory
from oasis_forge.ledger import Attempt, Ledger


def test_tier1_structural_passes_on_reassociation_noise():
	cfg = PortConfig()
	g = np.random.randn(128, 256).astype(np.float32)
	port = g + np.random.randn(*g.shape).astype(np.float32) * 1e-4  # reassociation-scale
	r = tier1_correctness(port, g, PortTier.STRUCTURAL, cfg)
	assert r.passed


def test_tier1_structural_fails_on_real_bug():
	cfg = PortConfig()
	g = np.random.randn(64, 64).astype(np.float32)
	port = g.copy()
	port[0, 0] += 5.0  # a real wrong element
	r = tier1_correctness(port, g, PortTier.STRUCTURAL, cfg)
	assert not r.passed


def test_tier1_aggressive_tolerates_fp8_rounding():
	cfg = PortConfig()
	g = np.random.randn(64, 64).astype(np.float32)
	port = g * (1.0 + np.random.randn(*g.shape).astype(np.float32) * 1e-2)  # ~fp8 rel error
	strict = tier1_correctness(port, g, PortTier.STRUCTURAL, cfg)
	loose = tier1_correctness(port, g, PortTier.AGGRESSIVE, cfg)
	assert loose.passed and not strict.passed


def test_tier2_catches_compounding_drift():
	cfg = PortConfig()
	ref = [np.random.randn(16, 32).astype(np.float32) for _ in range(cfg.rollout_frames)]
	# tight port: tiny per-frame error -> passes
	tight = [x + np.random.randn(*x.shape).astype(np.float32) * 1e-3 for x in ref]
	# drifting port: error grows with frame index -> fails
	drift = [x + np.random.randn(*x.shape).astype(np.float32) * 1e-2 * (i + 1)
		for i, x in enumerate(ref)]
	assert tier2_trajectory(tight, ref, cfg).passed
	assert not tier2_trajectory(drift, ref, cfg).passed


def test_ledger_roundtrip_and_best(tmp_path):
	led = Ledger(tmp_path / "ledger.jsonl")
	led.append(Attempt("attn", "structural", True, True, True, 40.0, 60.0, 1.5,
		1e-4, 1e-5, approach="fused"))
	led.append(Attempt("attn", "aggressive", True, True, True, 30.0, 60.0, 2.0,
		2e-2, 5e-4, approach="fp8"))
	led.append(Attempt("attn", "structural", True, False, None, None, 60.0, None,
		9.9, None, approach="buggy"))
	best = led.best("attn")
	assert best is not None and best.speedup == 2.0
	assert len(led.all()) == 3
