"""Plot the improvement chart from results/gains.csv.

  python results/plot_gains.py                 # default target=solaris
  python results/plot_gains.py --target methodology-demo

Produces results/gains_<target>.png:
  (1) speedup per (scope, stage) bar chart  -- "how much each optimization bought"
  (2) end_to_end latency progression line   -- if 'end_to_end' scope rows exist

Append a row whenever you measure: record_gain.py, or just add a CSV line:
  date,target,device,stage,scope,baseline_ms,optimized_ms,speedup,notes
Stages are ordered by first appearance (record them in the order applied).
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load(target: str) -> list[dict]:
	rows = []
	with (ROOT / "gains.csv").open() as f:
		for r in csv.DictReader(f):
			if r["target"] == target:
				for k in ("baseline_ms", "optimized_ms", "speedup"):
					r[k] = float(r[k]) if r[k] else None
				rows.append(r)
	return rows


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--target", default="kvcraft")
	args = ap.parse_args()
	rows = load(args.target)
	if not rows:
		print(f"no rows for target={args.target} yet (chart will fill as gains are measured)")
		return

	stages = list(OrderedDict((r["stage"], None) for r in rows))
	scopes = list(OrderedDict((r["scope"], None) for r in rows))
	by = {(r["scope"], r["stage"]): r["speedup"] for r in rows}

	fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(stages) * len(scopes) / 2), 5))
	w = 0.8 / max(len(scopes), 1)
	for i, sc in enumerate(scopes):
		ys = [by.get((sc, st), 0) or 0 for st in stages]
		xs = [j + i * w for j in range(len(stages))]
		bars = ax.bar(xs, ys, w, label=sc)
		for b, y in zip(bars, ys):
			if y:
				ax.text(b.get_x() + w / 2, y, f"{y:.2f}x", ha="center", va="bottom", fontsize=8)
	ax.axhline(1.0, color="gray", ls="--", lw=0.8)
	ax.set_xticks([j + 0.4 - w / 2 for j in range(len(stages))])
	ax.set_xticklabels(stages, rotation=20, ha="right")
	ax.set_ylabel("speedup vs default (x)")
	ax.set_title(f"Solaris kernel optimization gains — {args.target}")
	ax.legend(fontsize=8)
	fig.tight_layout()
	out = ROOT / f"gains_{args.target}.png"
	fig.savefig(out, dpi=130)
	print(f"wrote {out}")


if __name__ == "__main__":
	main()
