#!/usr/bin/env bash
# Measure Solaris steady-state generation fps (warm cache) + quality gate vs a golden video.
# The "did the patch actually help, without breaking quality" step. Runs ON the box.
#
#   GPU=0 [XLA="<flags>"] [CACHE=<dir>] [GOLDEN=<mp4>] ./measure_solaris.sh
#
# Prints: fps, gen_seconds, and (if GOLDEN set + ffmpeg present) SSIM vs golden.
# fps = 257 frames / (video_write_time - "Running eval" time)  -> compile excluded if CACHE warm.
set -euo pipefail

B=${SOLARIS_RUN:-/mnt/SFS-nc15dnf9/oasis-port/solaris-run}
export HF_HOME=$B/hf JAX_COMPILATION_CACHE_DIR=${CACHE:-$B/jaxcache} CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -n "${XLA:-}" ] && export XLA_FLAGS="$XLA"
LOG=$B/measure.log
VID=$B/solaris/output/solaris/eval_structure/video_0_side_by_side.mp4

cd "$B/solaris" && rm -rf output
"$B/venv/bin/python" src/inference.py experiment_name=solaris device.eval_num_samples=1 > "$LOG" 2>&1 || true

if [ ! -f "$VID" ]; then echo "FAIL: no video produced"; tail -5 "$LOG"; exit 1; fi
RUNEVAL=$(grep "Running eval on eval_structure" "$LOG" | head -1 | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
VT=$(stat -c %Y "$VID"); RE=$(date -u -d "$RUNEVAL" +%s 2>/dev/null); GEN=$((VT-RE))
FPS=$(awk -v g="$GEN" 'BEGIN{ if(g>0) printf "%.3f", 257/g; else print "0" }')
echo "fps=$FPS gen_seconds=$GEN frames=257 xla=${XLA:-default}"

# Quality gate: SSIM vs golden (1.0 = identical). A real patch must keep SSIM high (e.g. >=0.98
# for a numerically-identical change; lower only for accepted precision trades like FP8/FP4).
if [ -n "${GOLDEN:-}" ] && [ -f "$GOLDEN" ] && command -v ffmpeg >/dev/null 2>&1; then
	SSIM=$(ffmpeg -i "$VID" -i "$GOLDEN" -lavfi ssim -f null - 2>&1 | grep -oE "All:[0-9.]+" | tail -1)
	echo "quality_ssim_vs_golden=$SSIM  (golden=$GOLDEN)"
else
	echo "quality_ssim_vs_golden=skipped (set GOLDEN to a baseline mp4; needs ffmpeg)"
fi
