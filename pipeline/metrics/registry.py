"""Pluggable-block registry for metrics processing.

Each metrics block (pipeline/metrics/blocks/*.py) computes ONE category from
possible_metrics.txt (cost, verbosity, robustness, agnosticism, ...) over the
lifting pipeline's results/ directory (manifest.json + per-run summary.json).
A block registers itself with @register_block("name") at import time; call
load_all_blocks() once (run_metrics.py does this at startup) to import every
module under blocks/ and populate BLOCKS. Adding, removing, or replacing a
metrics category is then a matter of adding/deleting/editing one file under
blocks/ -- nothing else in the pipeline has to change, not even an import
list, since blocks/ is scanned rather than hardcoded.
"""

import importlib
import pkgutil

BLOCKS = {}


def register_block(name):
    """Function decorator: registers a metrics block under `name`.

    The decorated function must implement `compute(runs, results_dir) -> dict`,
    where `runs` is the list of per-(binary,backend) run records from
    manifest.json (each enriched with a "binary_meta" key holding that
    binary's metadata.json), and `results_dir` is the pipeline's results/
    root (for blocks that need to read files manifest.json doesn't already
    carry, e.g. IR dumps). Must return a JSON-serializable summary dict; a
    "rows" key, if present, is treated as the block's per-run/per-function
    CSV export and is written separately rather than inlined into the JSON.
    """
    def deco(fn):
        if name in BLOCKS:
            raise ValueError(f"metrics block '{name}' already registered")
        BLOCKS[name] = fn
        return fn
    return deco


def load_all_blocks():
    """Import every module under pipeline/metrics/blocks/, running each
    one's @register_block decorator. blocks/ has no __init__.py (a plain
    namespace package, like the rest of this project) -- pkgutil.iter_modules
    walks its directory directly, so a new block file is picked up with no
    registration step anywhere else.
    """
    import pipeline.metrics.blocks as blocks_pkg

    for _, module_name, _ in pkgutil.iter_modules(blocks_pkg.__path__, prefix=f"{blocks_pkg.__name__}."):
        importlib.import_module(module_name)
