"""Orchestrator for phase 2 of the pipeline: metrics processing over the
results/ directory produced by pipeline.run_lifting.

Each metrics category (cost, expansion_ratio, robustness, agnosticism, ...) is a
self-contained block under pipeline/metrics/blocks/ that registers itself
with @register_block. This script only loads manifest.json plus each run's
binary metadata, then hands the run list to whichever blocks were requested --
adding a new metrics category later means adding one file under blocks/, not
editing this script. Removing or replacing a category is the same: edit or
delete that one file.

Usage:
    python -m pipeline.metrics.run_metrics --results-dir results
    python -m pipeline.metrics.run_metrics --results-dir results --blocks cost
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common import ensure_dir, write_json  # noqa: E402
from pipeline.metrics.registry import BLOCKS, load_all_blocks  # noqa: E402

load_all_blocks()

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_runs(results_dir):
    """Load manifest.json and attach each run's binary metadata.json under
    "binary_meta", so blocks can group/filter by arch/opt/bits without each
    one re-reading the file itself.
    """
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest.json under {results_dir} -- run pipeline.run_lifting first")
    runs = json.loads(manifest_path.read_text())["runs"]

    metadata_cache = {}
    for run in runs:
        binary_dir = Path(run["outdir"]).parent
        key = str(binary_dir)
        if key not in metadata_cache:
            meta_path = binary_dir / "metadata.json"
            metadata_cache[key] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        run["binary_meta"] = metadata_cache[key]
    return runs


def write_block_output(outdir, name, summary):
    rows = summary.pop("rows", None)
    write_json(outdir / f"{name}.json", summary)

    csv_path = None
    if rows:
        csv_path = outdir / f"{name}.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()
            # Flushed one row at a time (rather than writerows' single bulk
            # write) to avoid the same OSError as write_whole_binary_dump --
            # shared-folder mounts (vboxsf) corrupt the raw write() return
            # value once the buffered payload gets large enough.
            for row in rows:
                writer.writerow(row)
                f.flush()
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Compute metrics over a completed lifting run")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                         help="Root output directory from run_lifting.py (default: ./results)")
    parser.add_argument("--outdir", type=Path, default=None,
                         help="Directory to write metrics output (default: <results-dir>/metrics)")
    parser.add_argument("--blocks", default=None,
                         help=f"Comma-separated metric blocks to run (available: {', '.join(BLOCKS)}). "
                              "Default: all registered blocks")
    args = parser.parse_args()

    if args.blocks:
        requested = [b.strip() for b in args.blocks.split(",") if b.strip()]
        unknown = set(requested) - set(BLOCKS)
        if unknown:
            parser.error(f"Unknown block(s): {', '.join(sorted(unknown))} (available: {', '.join(BLOCKS)})")
    else:
        requested = list(BLOCKS)

    outdir = args.outdir or (args.results_dir / "metrics")
    runs = load_runs(args.results_dir)
    ensure_dir(outdir)

    print(f"Runs: {len(runs)} (from {args.results_dir / 'manifest.json'})")
    print(f"Blocks: {', '.join(requested)}")

    for name in requested:
        summary = BLOCKS[name](runs, args.results_dir)
        csv_path = write_block_output(outdir, name, summary)

        print(f"\n[{name}]")
        for metric_name, stats in summary.get("metrics", {}).items():
            if stats.get("n"):
                print(f"  {metric_name} ({stats['direction']}): "
                      f"mean={stats['mean']:.4f} median={stats['median']:.4f} n={stats['n']}")
            else:
                print(f"  {metric_name}: no data")
        print(f"  wrote {outdir / f'{name}.json'}")
        if csv_path:
            print(f"  wrote {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
