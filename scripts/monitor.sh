#!/usr/bin/env bash
# Live dashboard: problem status + latest ledger attempts. Refreshes every N seconds.
# Usage: ./monitor.sh [interval_seconds]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${1:-30}"
cd "$ROOT"
while true; do
	clear
	echo "=== oasis-h100-port @ $(date) ==="
	uv run oasis-forge problems || python -m oasis_forge.cli problems
	echo
	uv run oasis-forge ledger || python -m oasis_forge.cli ledger
	sleep "$INTERVAL"
done
