"""Profiling harness — where does the kernel's time go? Finds bubbles before we optimize blind.

  python prof.py --shape stream

Reports per iteration: wall time, summed GPU-kernel time, GPU-busy% (wall vs kernel-sum ==
the BUBBLE gap), kernel launch count, and the top kernels. Low busy% or high launch count =
inter-kernel/launch bubbles worth popping. High busy% on a single kernel = already near
roofline; go deeper with ncu (occupancy/TC-util/memory) not more wall-clock guessing:

  ncu --set full -k 'regex:.*' -c 5 <venv>/python prof.py --shape stream     # per-kernel roofline
  nsys profile -o /tmp/tr <venv>/python prof.py --shape stream               # timeline / gaps

Generic over problems: works for any (reference.generate_input, submission.custom_kernel) pair.
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from reference import BENCH, generate_input
from submission import custom_kernel


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--shape", default="stream")
	ap.add_argument("--dtype", default="bfloat16")
	ap.add_argument("--iters", type=int, default=20)
	args = ap.parse_args()

	dtype = getattr(torch, args.dtype)
	sh = next(s for s in BENCH if s["name"] == args.shape)
	kw = {k: v for k, v in sh.items() if k != "name"}
	inp = generate_input(**kw, device="cuda", dtype=dtype)

	for _ in range(10):  # warmup (compile, autotune, allocator)
		custom_kernel(*inp)
	torch.cuda.synchronize()

	with profile(activities=[ProfilerActivity.CUDA]) as prof:
		s = torch.cuda.Event(enable_timing=True)
		e = torch.cuda.Event(enable_timing=True)
		s.record()
		for _ in range(args.iters):
			custom_kernel(*inp)
		e.record()
		torch.cuda.synchronize()
	wall_ms = s.elapsed_time(e) / args.iters

	rows = sorted(
		(k for k in prof.key_averages() if k.self_device_time_total > 0),
		key=lambda k: -k.self_device_time_total,
	)
	kernel_ms = sum(k.self_device_time_total for k in rows) / args.iters / 1000.0
	launches = sum(k.count for k in rows) / args.iters
	busy = 100.0 * kernel_ms / wall_ms if wall_ms else 0.0

	print(f"shape={args.shape}  device={torch.cuda.get_device_name(0)}  dtype={args.dtype}")
	print(f"wall={wall_ms:.3f}ms  gpu_kernel={kernel_ms:.3f}ms  busy={busy:.0f}%  "
		f"launches/iter={launches:.0f}   (low busy% / many launches => bubbles)")
	print(f"{'kernel':52s} {'self_ms':>9s} {'count':>6s}")
	for k in rows[:8]:
		print(f"{k.key[:52]:52s} {k.self_device_time_total / args.iters / 1000:9.4f} "
			f"{k.count / args.iters:6.0f}")


if __name__ == "__main__":
	main()
