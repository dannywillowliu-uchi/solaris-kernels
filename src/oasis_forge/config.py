"""Hardware + port configuration.

Source arch is the B300 serving deployment (Blackwell, sm_100, FP16). Target is the
per-user serving GPU H100/H200 (Hopper, sm_90). The port loop reads these to inform
codegen targets and roofline math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PortTier(str, Enum):
	"""How aggressive the port is allowed to be for a given kernel."""

	# FP16 -> FP16. No precision loss; correctness is binary (allclose at reassociation
	# tolerance). Pure structural win: fusion, TMA, wgmma. Default, low risk.
	STRUCTURAL = "structural"

	# FP16 -> FP8. Opt-in. Bigger speedup, real numeric drift. Must clear the trajectory
	# gate (drift(fp8) <= drift_budget). Only where the budget allows.
	AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class HardwareConfig:
	name: str
	sm: str
	# Peak throughput (TFLOPS, dense, no sparsity) — roofline numerator.
	bf16_tflops: float
	fp16_tflops: float
	fp8_tflops: float | None  # None == no native FP8 tensor cores
	hbm_tbps: float
	smem_kb_per_sm: int
	num_sms: int
	# Hopper-class codegen features available to the agent.
	has_tma: bool
	has_wgmma: bool
	has_tb_clusters: bool


# --- Source: B300 serving (Blackwell). Reference timings come from here. ---
B300 = HardwareConfig(
	name="B300 (Blackwell)",
	sm="sm_100",
	bf16_tflops=2250.0,
	fp16_tflops=2250.0,
	fp8_tflops=4500.0,
	hbm_tbps=8.0,
	smem_kb_per_sm=228,
	num_sms=148,
	has_tma=True,
	has_wgmma=True,
	has_tb_clusters=True,
)

# --- Port target: H100 SXM (Hopper). Benchmarks run here. ---
# H200 is the same compute, ~4.8 TB/s HBM3e — override hbm_tbps when targeting H200.
H100_SXM = HardwareConfig(
	name="H100 SXM (Hopper)",
	sm="sm_90",
	bf16_tflops=990.0,
	fp16_tflops=990.0,
	fp8_tflops=1979.0,
	hbm_tbps=3.35,
	smem_kb_per_sm=228,
	num_sms=132,
	has_tma=True,
	has_wgmma=True,
	has_tb_clusters=True,
)

H200_SXM = HardwareConfig(
	name="H200 SXM (Hopper)",
	sm="sm_90",
	bf16_tflops=990.0,
	fp16_tflops=990.0,
	fp8_tflops=1979.0,
	hbm_tbps=4.8,
	smem_kb_per_sm=228,
	num_sms=132,
	has_tma=True,
	has_wgmma=True,
	has_tb_clusters=True,
)


@dataclass
class PortConfig:
	source: HardwareConfig = B300
	target: HardwareConfig = H100_SXM

	# Tier-1 (per-kernel) correctness tolerances vs golden high-precision reference.
	# Structural: only float reassociation error is allowed. Aggressive: FP8 rounding.
	structural_rtol: float = 2e-3
	structural_atol: float = 2e-3
	aggressive_rtol: float = 3e-2
	aggressive_atol: float = 3e-2

	# Tier-2 (trajectory) drift budget over an N-frame autoregressive rollout.
	# Pinned to "no worse than the shipped B300 deployment" once that number exists;
	# until then this is a placeholder ceiling on mean per-frame latent MSE.
	rollout_frames: int = 64
	drift_budget_latent_mse: float = 1e-3
	# Optional perceptual ceiling on decoded RGB (LPIPS or similar). None == skip.
	drift_budget_perceptual: float | None = None

	# A kept port must beat the source-arch reference time by at least this factor on
	# the target. (Porting that regresses isn't a port.)
	min_speedup: float = 1.0

	knowledge_dir: str = "knowledge"
	ledger_path: str = "knowledge/ledger.jsonl"

	def tolerances(self, tier: PortTier) -> tuple[float, float]:
		if tier is PortTier.STRUCTURAL:
			return self.structural_rtol, self.structural_atol
		return self.aggressive_rtol, self.aggressive_atol
