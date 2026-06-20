"""Flat JSONL attempt ledger — no DB (forge-v4 style).

One line per port attempt. The agent and post-run hooks append; the prompt builder and
monitor read. Append-only so it's crash-safe and greppable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Attempt:
	problem: str  # kernel id, e.g. "stdit_attention"
	tier: str  # PortTier value
	# Outcome
	compiled: bool
	tier1_pass: bool  # per-kernel allclose vs golden
	tier2_pass: bool | None  # trajectory drift gate; None if not run
	# Timing on the TARGET arch (us). reference_us is the source-arch kernel time.
	target_us: float | None
	reference_us: float | None
	speedup: float | None
	# Numerics
	max_abs_err: float | None
	rollout_latent_mse: float | None
	# Provenance
	approach: str = ""  # one-line: "fused QK + TMA loads, wgmma m64n128k16"
	notes: str = ""
	submission_sha: str = ""
	ts: float = field(default_factory=time.time)


class Ledger:
	def __init__(self, path: str | Path):
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)

	def append(self, attempt: Attempt) -> None:
		with self.path.open("a") as f:
			f.write(json.dumps(asdict(attempt)) + "\n")

	def all(self) -> list[Attempt]:
		if not self.path.exists():
			return []
		out: list[Attempt] = []
		for line in self.path.read_text().splitlines():
			line = line.strip()
			if line:
				out.append(Attempt(**json.loads(line)))
		return out

	def best(self, problem: str) -> Attempt | None:
		"""Best kept attempt for a problem: passed both gates, max speedup."""
		kept = [
			a
			for a in self.all()
			if a.problem == problem
			and a.tier1_pass
			and (a.tier2_pass is True)
			and a.speedup is not None
		]
		if not kept:
			return None
		return max(kept, key=lambda a: a.speedup or 0.0)
