"""Pure-Python implementations of the inter-architecture agnosticism metrics
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

import statistics

# Expansion ratio block
def expansion_ratio(ir_size, native_instruction_count):
    """IR-size expansion ratio: ir_size / native_instruction_count, e.g. IR
    ops (or lines) per native instruction for one lifted function.
    """
    if ir_size is None or not native_instruction_count:
        return None
    return ir_size / native_instruction_count


def weighted_jaccard(hist_a, hist_b):
    """Weighted Jaccard (Ruzicka) similarity between two op-frequency
    histograms ({op_name: count}): sum(min(a_i, b_i)) / sum(max(a_i, b_i))
    over the union of keys. Generalizes plain set-Jaccard to frequency-
    weighted sets -- plain set-Jaccard saturates near 1.0 for IR opcode
    vocabularies, which are small and closed (almost any function touches
    the same handful of op types regardless of architecture), so weighting
    by how often each op actually occurs is what keeps this metric
    informative. Returns None if both histograms are empty (nothing to
    compare -- distinct from a legitimate 0.0 similarity).
    """
    keys = hist_a.keys() | hist_b.keys()
    if not keys:
        return None
    numer = sum(min(hist_a.get(k, 0), hist_b.get(k, 0)) for k in keys)
    denom = sum(max(hist_a.get(k, 0), hist_b.get(k, 0)) for k in keys)
    if denom == 0:
        return None
    return numer / denom


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
