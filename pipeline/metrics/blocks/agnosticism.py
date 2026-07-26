"""Agnosticism metrics block: how consistent a binary's IR representation
stays, within one backend/IR level, when the same source is compiled for
different target architectures (x86/ARM/MIPS/...). First metric
implemented: weighted Jaccard (Ruzicka) similarity over op-type frequency
histograms -- see pipeline/metrics/formulas.py's weighted_jaccard and
possible_metrics.txt's "Distributional" section. Jensen-Shannon divergence,
IR-size coefficient of variation, and cyclomatic complexity delta are
planned follow-ups in this same file, added one at a time the way
robustness.py's error_rate_by_category/escape_fraction were.

Only binja_llil/binja_mlil/binja_hlil currently record a per-function
op_histogram field (added alongside this block, see each binja_*_lift.py's
lift_function) -- the other 5 backends get the same instrumentation in a
later pass, following this project's usual one-backend-at-a-time rollout
for a new metrics category.

Grouping and the fixed-compiler/opt-level requirement
------------------------------------------------------
Comparing IR across architectures is only meaningful if the compiler and
optimization level are held fixed -- otherwise the comparison measures
compiler/opt-level differences, not lifting differences. Runs are grouped by
(project, project_version, compiler, compiler_version, opt, name, backend)
from each run's binary_meta (pipeline/common.py's parse_binary_metadata,
already extracted from the BinKit filename convention) and only
(arch, bits) is allowed to vary within a group -- this project's binaries/
corpus already has this shape, e.g. binaries/ls/{x86,arm,mips,mipseb}/32/O0/
coreutils-8.29_gcc-6.4.0_*_32_O0_ls.elf.

Function matching is by name only for now (exact match across the two
builds in a pair) -- no fallback for stripped binaries yet, matching this
project's other metrics blocks' current scope. Function match rate is
computed over every successfully-lifted function (status "ok"), not just
those with a histogram, since it's a precondition metric that should hold
regardless of which specific metric consumes the matched subset afterward.

Every metric is reported three ways: an overall aggregate, broken down
by_backend, and broken down by_arch_pair (e.g. "arm_32_vs_x86_32") -- the
interesting agnosticism signal is expected to show up in *which*
architecture pair diverges, not just the overall mean.
"""

import itertools
from collections import defaultdict

from pipeline.metrics.blocks.expansion_ratio import _load_lift_records
from pipeline.metrics.formulas import aggregate_stats, weighted_jaccard
from pipeline.metrics.registry import register_block

GROUP_META_KEYS = ("project", "project_version", "compiler", "compiler_version", "opt", "name")


def _group_key(meta):
    return tuple(meta.get(k) for k in GROUP_META_KEYS)


def _arch_label(meta):
    return f"{meta.get('arch')}_{meta.get('bits')}"


def _arch_pair_label(label_a, label_b):
    return "_vs_".join(sorted((label_a, label_b)))


def _function_data(outdir):
    """(names_ok, histograms): names_ok is every successfully-lifted
    function's name (the precondition/match-rate universe); histograms is
    name -> op_histogram dict, only for functions that have one (currently
    only binja_llil/mlil/hlil records this field).
    """
    names_ok = set()
    histograms = {}
    for record in _load_lift_records(outdir):
        if record.get("status") != "ok":
            continue
        name = record.get("function")
        names_ok.add(name)
        histogram = record.get("op_histogram")
        if histogram is not None:
            histograms[name] = histogram
    return names_ok, histograms


@register_block("weighted_jaccard")
def compute_weighted_jaccard(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok"]

    groups = defaultdict(dict)
    for run in completed:
        meta = run.get("binary_meta") or {}
        if meta.get("arch") is None or meta.get("bits") is None:
            continue
        key = (_group_key(meta), run["backend"])
        groups[key].setdefault(_arch_label(meta), run)

    overall = []
    by_backend = defaultdict(list)
    by_arch_pair = defaultdict(list)
    match_rates_overall = []
    match_rates_by_pair = defaultdict(list)
    rows = []
    num_groups_evaluated = 0

    for (group_key, backend), by_arch in groups.items():
        if len(by_arch) < 2:
            continue
        num_groups_evaluated += 1

        data_by_arch = {label: _function_data(run["outdir"]) for label, run in by_arch.items()}

        for label_a, label_b in itertools.combinations(sorted(data_by_arch), 2):
            names_a, hist_a = data_by_arch[label_a]
            names_b, hist_b = data_by_arch[label_b]
            union = names_a | names_b
            matched = names_a & names_b
            pair_label = _arch_pair_label(label_a, label_b)

            if union:
                match_rate = len(matched) / len(union)
                match_rates_overall.append(match_rate)
                match_rates_by_pair[pair_label].append(match_rate)

            for name in sorted(matched):
                if name not in hist_a or name not in hist_b:
                    continue
                wj = weighted_jaccard(hist_a[name], hist_b[name])
                if wj is None:
                    continue
                rows.append({
                    "project": group_key[0],
                    "project_version": group_key[1],
                    "compiler": group_key[2],
                    "compiler_version": group_key[3],
                    "opt": group_key[4],
                    "binary": group_key[5],
                    "backend": backend,
                    "arch_pair": pair_label,
                    "function": name,
                    "weighted_jaccard": wj,
                })
                overall.append(wj)
                by_backend[backend].append(wj)
                by_arch_pair[pair_label].append(wj)

    return {
        "block": "weighted_jaccard",
        "num_groups_evaluated": num_groups_evaluated,
        "num_functions_evaluated": len(rows),
        "metrics": {
            "weighted_jaccard": {
                "direction": "higher_is_better", "range": "[0,1]",
                **(aggregate_stats(overall) or {"n": 0}),
            },
            "function_match_rate": {
                "direction": "descriptive", "range": "[0,1]",
                **(aggregate_stats(match_rates_overall) or {"n": 0}),
            },
        },
        "by_backend": {
            backend: {"weighted_jaccard": aggregate_stats(vals)}
            for backend, vals in by_backend.items()
        },
        "by_arch_pair": {
            pair: {
                "weighted_jaccard": aggregate_stats(vals),
                "function_match_rate": aggregate_stats(match_rates_by_pair[pair]),
            }
            for pair, vals in by_arch_pair.items()
        },
        "rows": rows,
    }
