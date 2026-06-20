#!/usr/bin/env bash
# Profile Solaris generation on the GPU box -> ranked GPU kernels by total time.
# The "find the slowness / pop the bubble" step. Runs ON the box (invoked via ssh by the loop).
#
#   GPU=0 ./profile_solaris.sh [out_file]
#
# Uses nsys with --cuda-graph-trace=node (REQUIRED — XLA wraps the rollout in CUDA graphs;
# without it nsys reports "does not contain CUDA kernel data"). Captures a warm-cache window
# of steady-state generation (delay past model load), then ranks kernels.
set -euo pipefail

B=${SOLARIS_RUN:-/mnt/SFS-nc15dnf9/oasis-port/solaris-run}
OUT=${1:-$B/profile_kernels.txt}
export HF_HOME=$B/hf JAX_COMPILATION_CACHE_DIR=${CACHE:-$B/jaxcache} CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -n "${XLA:-}" ] && export XLA_FLAGS="$XLA"
NSYS=/usr/local/bin/nsys
NSYS_STATS=/usr/local/cuda-13.0/bin/nsys

cd "$B/solaris"
"$NSYS" profile --trace=cuda,cudnn,cublas --cuda-graph-trace=node \
	--delay="${DELAY:-55}" --duration="${DURATION:-70}" --force-overwrite true -o "$B/trace_prof" \
	"$B/venv/bin/python" src/inference.py experiment_name=solaris device.eval_num_samples=1 >/dev/null 2>&1 || true

echo "# Solaris GPU kernel ranking ($(date -u +%FT%TZ))  XLA=${XLA:-default}"
"$NSYS_STATS" stats --report cuda_gpu_kern_sum --format table "$B/trace_prof.nsys-rep" 2>/dev/null \
	| grep -vE "Generating|Processing|SQLite|^$" | head -30 | tee "$OUT"
