"""Eval harness for AdaLN modulation. Correctness-preserving speedup (candidate vs eager golden).

  python eval.py            # all shapes the device can run
  python eval.py --dtype float16
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from reference import BENCH, generate_input, ref_eager
from submission import custom_kernel


def _bench_cuda(fn, args, warmup=10, iters=50):
	flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
	for _ in range(warmup):
		fn(*args)
	torch.cuda.synchronize()
	times = []
	for _ in range(iters):
		flush.zero_()
		s = torch.cuda.Event(enable_timing=True)
		e = torch.cuda.Event(enable_timing=True)
		s.record()
		fn(*args)
		e.record()
		torch.cuda.synchronize()
		times.append(s.elapsed_time(e))
	return statistics.median(times)


def _bench_cpu(fn, args, warmup=2, iters=5):
	for _ in range(warmup):
		fn(*args)
	times = []
	for _ in range(iters):
		t0 = time.perf_counter()
		fn(*args)
		times.append((time.perf_counter() - t0) * 1000.0)
	return statistics.median(times)


def _allclose(a, b, rtol, atol):
	return all(torch.allclose(x.float(), y.float(), rtol=rtol, atol=atol) for x, y in zip(a, b))


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--dtype", default="bfloat16")
	ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--rtol", type=float, default=2e-2)
	ap.add_argument("--atol", type=float, default=2e-2)
	args = ap.parse_args()

	dtype = getattr(torch, args.dtype)
	dev = args.device
	bench = _bench_cuda if dev == "cuda" else _bench_cpu
	name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
	print(f"device: {name}  dtype: {args.dtype}")

	results = []
	all_ok = True
	for sh in BENCH:
		if dev == "cpu" and sh["D"] > 512:
			print(f"{sh['name']:8s} skipped on CPU")
			continue
		inp = generate_input(sh["B"], sh["S"], sh["D"], sh["seed"], device=dev, dtype=dtype)
		out = custom_kernel(*inp)
		gold = ref_eager(*inp)
		ok = _allclose(out, gold, args.rtol, args.atol)
		all_ok = all_ok and ok
		max_err = max((o.float() - g.float()).abs().max().item() for o, g in zip(out, gold))

		t_eager = bench(ref_eager, inp)
		t_cand = bench(custom_kernel, inp)
		speedup = t_eager / t_cand if t_cand else 0.0
		print(f"{sh['name']:8s} correct={str(ok):5s} max_err={max_err:.2e}  "
			f"eager={t_eager:7.3f}ms  fused={t_cand:7.3f}ms  speedup={speedup:5.2f}x")
		results.append(dict(shape=sh["name"], correct=ok, max_err=max_err,
			eager_ms=t_eager, fused_ms=t_cand, speedup=speedup))

	Path("results.json").write_text(json.dumps(results, indent=2))
	print(f"\nALL CORRECT: {all_ok}   ({len(results)} shapes)")
	return 0 if all_ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
