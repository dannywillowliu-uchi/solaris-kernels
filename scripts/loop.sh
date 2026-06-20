#!/usr/bin/env bash
# Continuous port-and-optimize loop for one kernel problem.
# Mirrors ../../amd-kernel-forge/scripts/loop.sh. STUB until the H100 box + agent are wired.
#
# Usage: ./loop.sh <problem> <gpu> <hours>
set -euo pipefail

PROBLEM="${1:?usage: loop.sh <problem> <gpu> <hours>}"
GPU="${2:-0}"
HOURS="${3:-12}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STOP_FILE="$ROOT/.stop"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))

echo "[loop] problem=$PROBLEM gpu=$GPU hours=$HOURS root=$ROOT"

if [[ ! -f "$ROOT/problems/$PROBLEM/task_files/golden.npz" ]]; then
	echo "[loop] ERROR: no golden.npz for $PROBLEM — run 'oasis-forge harvest $PROBLEM' first."
	echo "[loop] (harvest needs a GPU box running Oasis-500M; see harvest/README.md)"
	exit 1
fi

while [[ ! -f "$STOP_FILE" && $(date +%s) -lt $DEADLINE ]]; do
	echo "[loop] launching agent for $PROBLEM on gpu $GPU ..."
	# TODO: oasis-forge solve "$PROBLEM" --gpu "$GPU"
	# TODO: post_run hook — save best solution, update ledger, append episode
	echo "[loop] (stub) agent launch not yet wired — see agents/agent_prompt.md"
	break
done

echo "[loop] done."
