"""Roofline / frame-budget calculator for the streaming few-step video DiT.

Answers, before any GPU time: which kernel binds, how far from realtime we are on N×H100,
and how much the real levers (KV window, camera-banding, FP8 FFN, chunk size) buy.

All outputs are ESTIMATES from first-order FLOP/byte counts + the H100 roofline. They model
the per-DiT-forward cost of generating ONE chunk in a streaming, KV-cached rollout:
query = the new chunk's tokens; keys = cached window + current chunk. Overlap of compute and
HBM is assumed ideal (true time is >= max(compute, memory)). VAE decode is a labelled
placeholder until measured. Treat as an order-of-magnitude planning tool, not a profiler.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import H100_SXM, HardwareConfig


@dataclass(frozen=True)
class ModelShape:
	name: str
	d_model: int
	n_layers: int
	n_heads: int
	head_dim: int
	ffn_dim: int
	params_b: float  # billions, for weight-read bytes
	tokens_per_latent_frame: int
	rgb_per_latent_frame: int  # VAE temporal downsample
	dmd_steps: int


# The operator-provided OASIS DMD driving config.
OASIS_DMD = ModelShape(
	name="OASIS-DMD-14B (3-cam driving)",
	d_model=5120,
	n_layers=40,
	n_heads=40,
	head_dim=128,
	ffn_dim=13824,
	params_b=14.0,
	tokens_per_latent_frame=4608,  # 32x144 (latent 64x288, patch 2x2)
	rgb_per_latent_frame=4,  # ~4x temporal downsample
	dmd_steps=4,  # sigmas [1, .9375, .8333, .625, 0]
)


@dataclass
class KernelCost:
	name: str
	flops: float  # per full forward (all layers)
	bytes_hbm: float  # per full forward
	compute_s: float
	memory_s: float

	@property
	def roofline_s(self) -> float:
		return max(self.compute_s, self.memory_s)

	@property
	def bound(self) -> str:
		return "compute" if self.compute_s >= self.memory_s else "memory"


@dataclass
class BudgetResult:
	kernels: list[KernelCost]
	per_forward_s: float
	per_chunk_s: float
	per_rgb_frame_s: float
	fps: float
	target_fps: float
	chunk_frames: int
	window_frames: int
	n_gpus: int
	dtype: str
	camera_band: float
	fp8_ffn: bool

	@property
	def gap(self) -> float:
		"""How many x too slow (or headroom if <1)."""
		return self.target_fps / self.fps if self.fps else float("inf")


def budget(
	model: ModelShape = OASIS_DMD,
	hw: HardwareConfig = H100_SXM,
	chunk_frames: int = 3,
	window_frames: int = 21,
	n_gpus: int = 1,
	target_fps: float = 30.0,
	camera_band: float = 1.0,  # 1.0=dense; 1/3 if cross-camera attention pruned to own camera
	fp8_ffn: bool = False,  # FP8 on FFN/QKV GEMMs (2x compute, less drift-sensitive)
	vae_decode_ms_per_latent_frame: float = 8.0,  # PLACEHOLDER until measured
) -> BudgetResult:
	d = model.d_model
	tpf = model.tokens_per_latent_frame
	q = chunk_frames * tpf  # query tokens (the new chunk)
	k = (window_frames + chunk_frames) * tpf  # key tokens (window + current)
	L = model.n_layers

	gemm_tflops = (hw.fp8_tflops if (fp8_ffn and hw.fp8_tflops) else hw.bf16_tflops) * n_gpus
	attn_tflops = hw.bf16_tflops * n_gpus  # keep attention in BF16 (FP8-attn drift)
	hbm = hw.hbm_tbps * 1e12 * n_gpus
	dtype = "fp8-ffn/bf16-attn" if fp8_ffn else "bf16"

	# ---- per-forward FLOPs (all layers) ----
	qkv = L * 2 * q * d * (3 * d)
	attn = L * 4 * q * k * d * camera_band  # QK^T + AV; camera_band prunes cross-camera keys
	outproj = L * 2 * q * d * d
	ffn = L * 4 * q * d * model.ffn_dim

	# ---- HBM bytes: weights read once per forward (the big memory term), sharded across GPUs ----
	w_bytes = model.params_b * 1e9 * 2.0  # bf16 weights
	# attention also streams Q/K/V (flash recompute) — small vs weights, included for completeness.
	attn_bytes = L * (q + 2 * k) * d * 2.0

	def cost(name: str, flops: float, tflops: float, byts: float) -> KernelCost:
		return KernelCost(name, flops, byts, flops / (tflops * 1e12), byts / hbm)

	kernels = [
		cost("attention", attn, attn_tflops, attn_bytes),
		cost("ffn", ffn, gemm_tflops, w_bytes * (ffn / (qkv + outproj + ffn))),
		cost("qkv_proj", qkv, gemm_tflops, w_bytes * (qkv / (qkv + outproj + ffn))),
		cost("out_proj", outproj, gemm_tflops, w_bytes * (outproj / (qkv + outproj + ffn))),
	]

	# per-forward roofline: compute and memory overlap -> max of the two totals.
	tot_compute = sum(k_.compute_s for k_ in kernels)
	tot_memory = sum(k_.memory_s for k_ in kernels)
	per_forward = max(tot_compute, tot_memory)

	vae_s = (chunk_frames * vae_decode_ms_per_latent_frame) / 1000.0
	per_chunk = model.dmd_steps * per_forward + vae_s
	rgb_per_chunk = chunk_frames * model.rgb_per_latent_frame
	per_rgb_frame = per_chunk / rgb_per_chunk
	fps = 1.0 / per_rgb_frame if per_rgb_frame else 0.0

	return BudgetResult(
		kernels, per_forward, per_chunk, per_rgb_frame, fps, target_fps,
		chunk_frames, window_frames, n_gpus, dtype, camera_band, fp8_ffn,
	)
