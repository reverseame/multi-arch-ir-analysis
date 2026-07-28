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

import math
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


def jensen_shannon_similarity(hist_a, hist_b):
    """Jensen-Shannon similarity between two op-frequency histograms
    ({op_name: count}), normalized to [0,1] where 1.0 = identical
    distributions. Complements weighted_jaccard: both measure distributional
    overlap, but JSD is a true probability-distance (each histogram is
    normalized to sum to 1 first) and handles zero-frequency ops cleanly via
    the convention 0*log(0/x) = 0, whereas weighted_jaccard works directly
    on raw counts. Computed as 1 - sqrt(JSD_bits), the Jensen-Shannon
    distance (a proper metric, unlike JSD itself) using log base 2 so
    JSD_bits is bounded in [0,1] and the distance is too. Returns None only
    if BOTH histograms are empty (nothing to compare at all) -- mirrors
    weighted_jaccard's same both-vs-one-sided distinction.
    """
    total_a = sum(hist_a.values())
    total_b = sum(hist_b.values())
    if not total_a and not total_b:
        return None
    if not total_a or not total_b:
        return 0.0
    keys = hist_a.keys() | hist_b.keys()
    p = {k: hist_a.get(k, 0) / total_a for k in keys}
    q = {k: hist_b.get(k, 0) / total_b for k in keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def kl_div(dist):
        return sum(dist[k] * math.log2(dist[k] / m[k]) for k in keys if dist[k] > 0)

    jsd_bits = 0.5 * kl_div(p) + 0.5 * kl_div(q)
    jsd_bits = min(max(jsd_bits, 0.0), 1.0)  # clamp float noise at the [0,1] bound
    return 1.0 - math.sqrt(jsd_bits)


def coefficient_of_variation(values):
    """Coefficient of variation (sample stdev / mean) over a list of IR-size
    values for the same function across different architectures -- 0.0 means
    IR size is identical across every architecture compared, higher means
    lifted size diverges more per architecture for otherwise-equivalent
    source. Needs at least 2 values (sample stdev is undefined for 1) and a
    non-zero mean (an all-zero-size function has no meaningful ratio).
    Returns None in either of those cases.
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    mean = statistics.mean(clean)
    if mean == 0:
        return None
    return statistics.stdev(clean) / mean


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
