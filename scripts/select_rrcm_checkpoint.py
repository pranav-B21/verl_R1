#!/usr/bin/env python3
"""Select an RRCM checkpoint from validation-only greedy metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


STEP_PATTERN = re.compile(r"global_step_(\d+)$")


def read_metric(path: Path) -> tuple[float, float]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    row = payload[0]
    return float(row["HR"][0]), float(row["NDCG"][0])


def collect(eval_root: Path, tag_prefix: str, min_step: int, max_step: int) -> list[dict]:
    rows = []
    for step_dir in sorted(eval_root.glob("global_step_*")):
        match = STEP_PATTERN.match(step_dir.name)
        if not match:
            continue
        step = int(match.group(1))
        if step < min_step or step > max_step:
            continue
        metric_paths = sorted(step_dir.glob(f"{tag_prefix}*/test_metrics_top5.json"))
        if not metric_paths:
            continue
        metrics = [read_metric(path) for path in metric_paths]
        rows.append(
            {
                "step": step,
                "n_decodes": len(metrics),
                "hr5_mean": statistics.fmean(value[0] for value in metrics),
                "ndcg5_mean": statistics.fmean(value[1] for value in metrics),
                "metric_files": [str(path) for path in metric_paths],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--tag-prefix", default="valgreedy")
    parser.add_argument("--min-step", type=int, default=500)
    parser.add_argument("--max-step", type=int, default=1300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = collect(args.eval_root, args.tag_prefix, args.min_step, args.max_step)
    if not rows:
        raise SystemExit("no validation metrics matched the requested range and tag prefix")

    # Pre-registered order: HR@5, then NDCG@5, then the earlier checkpoint.
    selected = max(rows, key=lambda row: (row["hr5_mean"], row["ndcg5_mean"], -row["step"]))
    result = {
        "protocol": "rrcm-validation-only-checkpoint-selection-v1",
        "eval_root": str(args.eval_root),
        "tag_prefix": args.tag_prefix,
        "range": [args.min_step, args.max_step],
        "selected_step": selected["step"],
        "selection_order": ["hr5_mean", "ndcg5_mean", "earlier_step"],
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite selection record: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
