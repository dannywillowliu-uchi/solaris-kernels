"""H100 SSH executor — STUB until a target box exists.

Mirrors amd-kernel-forge/remote.py: an asyncssh wrapper to compile + benchmark candidate
kernels on the Hopper target. Filled in once we have a box. Kept as a typed interface so
the loop driver and cli can import against it now.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RemoteResult:
	ok: bool
	stdout: str
	stderr: str
	returncode: int


class H100Remote:
	"""SSH executor for the Hopper benchmark box."""

	def __init__(self, host: str, user: str, gpu: int, workspace: str, key: str | None = None):
		self.host = host
		self.user = user
		self.gpu = gpu
		self.workspace = workspace
		self.key = key

	async def run(self, cmd: str, timeout: float = 600.0) -> RemoteResult:
		raise NotImplementedError(
			"H100Remote is a stub. Wire to asyncssh once a Hopper box is assigned. "
			"See ../amd-kernel-forge/src/amd_forge/remote.py for the reference impl."
		)

	async def compile(self, submission_path: str, sm: str = "sm_90") -> RemoteResult:
		raise NotImplementedError("stub: nvcc/triton compile for sm_90 on the target box")

	async def benchmark(self, problem: str, submission_path: str) -> RemoteResult:
		raise NotImplementedError("stub: CUDA-event timed benchmark with L2 flush, RSE convergence")
