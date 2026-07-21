"""Coste (cost) metrics block: lifting time and peak memory per (binary, backend) run.

- lifting_time_s: orchestrator-measured wall time of the whole backend
  subprocess (run_lifting.py's wall_time_s) -- includes interpreter/JVM
  startup, not just the backend's own lifting work.
- internal_lifting_time_s: the backend's own self-reported duration_s from
  its summary.json -- excludes process/JVM startup, so it's the fairer
  number once warm-up noise (e.g. pyghidra's JVM boot) needs to be stripped
  out when comparing backends' actual lifting cost.
- max_rss_kb: peak resident set size of the backend subprocess, in KB (see
  run_subprocess_with_rusage in pipeline/run_lifting.py).

All three are already produced by run_lifting.py for every run (in
manifest.json), so this block is pure aggregation -- no new data collection.
Only successful runs ("status" == "ok") are counted: a failed/timed-out run's
wall time reflects how it died, not how long lifting takes.
"""

from collections import defaultdict
from pathlib import Path

from pipeline.metrics.formulas import aggregate_stats
from pipeline.metrics.registry import register_block

METRICS = {
    "lifting_time_s": {"direction": "lower_is_better", "unit": "s"},
    "internal_lifting_time_s": {"direction": "lower_is_better", "unit": "s"},
    "max_rss_kb": {"direction": "lower_is_better", "unit": "KB"},
}


def _run_values(run):
    summary = run.get("summary") or {}
    return {
        "lifting_time_s": run.get("wall_time_s"),
        "internal_lifting_time_s": summary.get("duration_s"),
        "max_rss_kb": run.get("max_rss_kb"),
    }


@register_block("cost")
def compute(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok"]

    overall = defaultdict(list)
    by_backend = defaultdict(lambda: defaultdict(list))
    rows = []

    for run in completed:
        values = _run_values(run)
        meta = run.get("binary_meta") or {}
        rows.append({
            "binary": Path(run["binary"]).name,
            "backend": run["backend"],
            "arch": meta.get("arch"),
            "bits": meta.get("bits"),
            "opt": meta.get("opt"),
            "compiler": meta.get("compiler"),
            **values,
        })
        for metric, value in values.items():
            if value is None:
                continue
            overall[metric].append(value)
            by_backend[run["backend"]][metric].append(value)

    return {
        "block": "cost",
        "num_runs_ok": len(completed),
        "num_runs_total": len(runs),
        "metrics": {
            name: {**spec, **(aggregate_stats(overall[name]) or {"n": 0})}
            for name, spec in METRICS.items()
        },
        "by_backend": {
            backend: {name: aggregate_stats(vals[name]) for name in METRICS}
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
