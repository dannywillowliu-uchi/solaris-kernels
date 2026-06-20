"""Eval harness for camera-banded attention. Runs anywhere torch runs.

  python eval.py                 # all shapes that fit the device (CPU skips the big ones)
  python eval.py --shapes tiny   # one shape
  python eval.py --dtype float16

Checks: (1) candidate matches ref_banded (correctness golden), (2) candidate speedup vs
ref_dense (the un-banded baseline), reported against the ~3x FLOP-ideal of camera banding.
On CUDA: CUDA-event timing, L2 flush, warmup, median of iters. On CPU: perf_counter, big
shapes skipped. Writes results.json for the ledger.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from reference import BENCH, generate_input, ref_banded, ref_dense
from submission import custom_kernel

CPU_MAX_ELEMS = 4_000_000  # skip shapes whose attn is too big to run on CPU in reasonable time


def _bench_cuda(fn, args, warmup=10, iters=50):
	flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")  # ~256MB > L2
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


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--dtype", default="bfloat16")
	ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--shapes", nargs="*", default=None)
	ap.add_argument("--rtol", type=float, default=2e-2)
	ap.add_argument("--atol", type=float, default=2e-2)
	args = ap.parse_args()

	dtype = getattr(torch, args.dtype)
	dev = args.device
	bench = _bench_cuda if dev == "cuda" else _bench_cpu
	shapes = [s for s in BENCH if args.shapes is None or s["name"] in args.shapes]

	if dev == "cuda":
		print(f"device: {torch.cuda.get_device_name(0)}  dtype: {args.dtype}")
	else:
		print(f"device: CPU  dtype: {args.dtype}  (big shapes skipped)")

	results = []
	all_ok = True
	for sh in shapes:
		name = sh["name"]
		if dev == "cpu" and sh["Sq"] * sh["Sk"] > CPU_MAX_ELEMS:
			print(f"{name:8s} skipped on CPU")
			continue
		q, k, v, gq, gk = generate_input(
			sh["B"], sh["H"], sh["Sq"], sh["Sk"], sh["D"], sh["seed"], device=dev, dtype=dtype)
		out = custom_kernel(q, k, v, gq, gk)
		gold = ref_banded(q, k, v, gq, gk)
		ok = torch.allclose(out.float(), gold.float(), rtol=args.rtol, atol=args.atol)
		all_ok = all_ok and ok
		max_err = (out.float() - gold.float()).abs().max().item()

		t_dense = bench(ref_dense, (q, k, v, gq, gk))
		t_cand = bench(custom_kernel, (q, k, v, gq, gk))
		speedup = t_dense / t_cand if t_cand else 0.0
		print(f"{name:8s} correct={str(ok):5s} max_err={max_err:.2e}  "
			f"dense={t_dense:8.3f}ms  candidate={t_cand:8.3f}ms  speedup={speedup:5.2f}x "
			f"(banding FLOP-ideal ~3.00x)")
		results.append(dict(shape=name, correct=ok, max_err=max_err,
			dense_ms=t_dense, candidate_ms=t_cand, speedup=speedup))

	Path("results.json").write_text(json.dumps(results, indent=2))
	print(f"\nALL CORRECT: {all_ok}   ({len(results)} shapes)  -> results.json")
	return 0 if all_ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
