#!/usr/bin/env bash
# Autonomous Solaris kernel-optimization loop (ported from amd-kernel-forge/scripts/loop.sh).
# Each iteration: profile on the box -> launch an optimization agent with the fresh profile +
# accumulated knowledge -> agent patches/measures/gates/records -> re-launch. Walltime + stop-file.
#
#   ./scripts/solaris_loop.sh [walltime_hours]   (default 12)
#
# The agent step uses `claude -p` with agents/solaris_agent_prompt.md. If you'd rather drive it
# interactively / via scheduled wakeups, run the loop body by hand — the tools + ledger are the same.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WALL_H="${1:-12}"
BOX="${SOLARIS_BOX:-root@95.133.253.31}"
BOX_RUN="/mnt/SFS-nc15dnf9/oasis-port/solaris-run"
STOP="$ROOT/.solaris_stop"
RUNS="$ROOT/runs"; mkdir -p "$RUNS"
DEADLINE=$(( $(date +%s) + WALL_H * 3600 ))
N=0

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$RUNS/loop.log"; }

# push the harness tools to the box once
scp -q "$ROOT"/harness/*.sh "$BOX:$BOX_RUN/" 2>/dev/null || true

while [ ! -f "$STOP" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
	N=$((N+1))
	log "=== iteration $N ==="

	# 1. PROFILE on the box -> ranked kernels
	log "profiling..."
	ssh "$BOX" "GPU=0 bash $BOX_RUN/profile_solaris.sh $BOX_RUN/profile_kernels.txt" \
		> "$RUNS/profile_${N}.txt" 2>&1 || log "profile failed (continuing)"

	# 2-5. launch the optimization agent with prompt + fresh profile + ledger
	PROMPT="$RUNS/prompt_${N}.md"
	{
		cat "$ROOT/agents/solaris_agent_prompt.md"
		echo; echo "## Latest profile (iteration $N)"; echo '```'; cat "$RUNS/profile_${N}.txt"; echo '```'
		echo; echo "## Gains ledger so far"; echo '```'; cat "$ROOT/results/gains.csv"; echo '```'
		echo; echo "Do ONE optimization iteration now: diagnose the top kernel, patch it, measure"
		echo "(GPU=0 bash $BOX_RUN/measure_solaris.sh on the box), quality-gate, KEEP-or-REVERT,"
		echo "append the attempt to results/gains.csv, write a one-line episode, commit + push."
	} > "$PROMPT"

	log "launching agent (iteration $N)..."
	if command -v claude >/dev/null 2>&1; then
		claude -p "$(cat "$PROMPT")" --dangerously-skip-permissions 2>&1 | tee "$RUNS/agent_${N}.log" || log "agent run errored"
	else
		log "claude CLI not found — run the iteration manually with $PROMPT, or drive via scheduled wakeups."
		break
	fi

	log "iteration $N done; best fps now:"; tail -1 "$ROOT/results/gains.csv" | tee -a "$RUNS/loop.log"
done

log "loop stopped (walltime or stop-file). Chart: python results/plot_gains.py --target solaris"
