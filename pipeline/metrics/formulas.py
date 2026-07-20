"""Pure-Python implementations of the inter-architecture invariance metrics
from possible_metrics.txt. No numpy/scipy dependency (not in requirements.txt) --
these operate on plain dicts/lists of ints and floats.

Direction of each metric (see possible_metrics.txt for definitions):
    weighted_jaccard         higher = better (1.0 = identical op-frequency distributions)
    jensen_shannon_similarity higher = better (1.0 = identical distributions)
    coefficient_of_variation lower  = better (0.0 = identical size across the set)
    cyclomatic_complexity_delta lower = better (0 = identical CFG complexity)
    expansion_ratio          descriptive, stability* across architectures 
                             (via coefficient_of_variation over a set of 
                             expansion_ratio values) signals lifting
                             consistency.
"""

import math
import statistics

# Verbosity block
def expansion_ratio(ir_size, native_instruction_count):
    """IR-size expansion ratio: ir_size / native_instruction_count, e.g. IR
    ops (or lines) per native instruction for one lifted function.
    """
    if ir_size is None or not native_instruction_count:
        return None
    return ir_size / native_instruction_count


def aggregate_stats(values):
    """mean/median/min/max/n over `values`, skipping Nones.

    The common summary shape every metrics block reports a metric in (see
    pipeline/metrics/blocks/*.py). Returns None if nothing usable remains.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }
