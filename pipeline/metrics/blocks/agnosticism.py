"""Agnosticism metrics block: how consistent a binary's IR representation
stays, within one backend/IR level, when the same source is compiled for
different target architectures (x86/ARM/MIPS/...). Two metrics implemented
so far, both over the same per-function op-type frequency histograms:
weighted Jaccard (Ruzicka) similarity and Jensen-Shannon similarity -- see
pipeline/metrics/formulas.py's weighted_jaccard/jensen_shannon_similarity
and possible_metrics.txt's "Distributional" section. Kept as separate
blocks/metrics (not treated as redundant) since they're a cross-validation
pair: same distributional-overlap question, different math (raw-count
ratio vs. normalized-probability distance), so agreement between them is
itself a signal. IR-size coefficient of variation and cyclomatic complexity
delta are planned follow-ups in this same file, added one at a time the way
robustness.py's error_rate_by_category/escape_fraction were.

All 8 backends are covered. binja_llil/binja_mlil/binja_hlil, pyghidra,
angr, r2, and ida record a per-function op_histogram field directly in
lift_records.json (see each backend's lift_function). retdec is the one
exception, same as expansion_ratio.py's category counts: RetDec has no live
decompiler session to reuse (its .ll output is a static text artifact), so
its histogram is built here instead, straight from whole_binary.ll via
_retdec_op_histograms -- see that function's docstring.

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
import re
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.metrics.blocks.expansion_ratio import _iter_ll_functions, _ll_line_opcode, _load_lift_records
from pipeline.metrics.formulas import aggregate_stats, jensen_shannon_similarity, weighted_jaccard
from pipeline.metrics.registry import register_block

GROUP_META_KEYS = ("project", "project_version", "compiler", "compiler_version", "opt", "name")


def _group_key(meta):
    return tuple(meta.get(k) for k in GROUP_META_KEYS)


def _arch_label(meta):
    return f"{meta.get('arch')}_{meta.get('bits')}"


def _arch_pair_label(label_a, label_b):
    return "_vs_".join(sorted((label_a, label_b)))


# A bare LLVM-IR basic-block label line ("entry:", "if.then:", "16:", optionally
# followed by a "; preds = ..." comment) -- not a real instruction, but
# _ll_line_opcode would otherwise extract the label name itself as a bogus
# "opcode" token. classify_ll_line (category counting) doesn't need this
# filter -- an uncategorized label just lands harmlessly in "other", which
# isn't part of any ratio -- but a raw op-histogram has no such safety net:
# every distinct token becomes its own vocabulary entry, and two
# architectures' compiler-generated label names (or how many basic blocks
# they split a function into) differing is a control-flow-shape artifact,
# not an op-frequency signal.
_LL_LABEL_RE = re.compile(r"^\s*[\w.$]+:\s*(;.*)?$")

# Every real top-level LLVM-IR instruction opcode RetDec can emit. Needed
# because _ll_line_opcode/classify_ll_line assume one instruction per
# physical line, which breaks for multi-line constructs -- chiefly `switch`,
# whose case-target lines ("i32 0, label %case0") and closing "]" are
# CONTINUATIONS of the switch on the preceding line, not their own
# instructions. classify_ll_line already had this same per-line assumption,
# harmlessly, since a stray "i32"/"]" token just falls into its unused
# "other" bucket -- but a raw op-histogram has no such bucket, so an
# unrecognized first token means "not a real instruction, skip it" instead
# of "count it anyway". Only opcodes need to be listed here (not operands),
# since a continuation line's first token is never one of these.
_LL_KNOWN_OPCODES = (
    # terminators
    {"ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume",
     "unreachable", "catchswitch", "catchret", "cleanupret"}
    # binary/bitwise ops
    | {"add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv", "fdiv",
       "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or", "xor"}
    # memory ops
    | {"alloca", "load", "store", "fence", "cmpxchg", "atomicrmw", "getelementptr"}
    # conversion ops
    | {"trunc", "zext", "sext", "fptrunc", "fpext", "fptoui", "fptosi",
       "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast"}
    # other ops
    | {"icmp", "fcmp", "phi", "select", "freeze", "call", "va_arg",
       "landingpad", "catchpad", "cleanuppad", "extractelement",
       "insertelement", "shufflevector", "extractvalue", "insertvalue"}
)


def _retdec_op_histograms(outdir):
    """name -> Counter({opcode: count}) by walking whole_binary.ll's
    top-level `define` blocks via expansion_ratio.py's shared
    _iter_ll_functions walker, keyed by the raw LLVM opcode token
    (_ll_line_opcode) instead of collapsed into a category the way
    expansion_ratio.py's _retdec_ir_category_counts does. Same rationale as
    that function for computing this here instead of inside
    retdec_lift.py: RetDec's .ll output is a static text artifact with no
    live session to reuse, so parsing it can happen at metrics time.

    Skips the function's own "define ... {" header and closing "}" (the
    first/last lines _iter_ll_functions includes) and basic-block label
    lines (see _LL_LABEL_RE) -- none of these are real instructions, and
    unlike classify_ll_line's category counting, a raw histogram has no
    "other" bucket to safely absorb them into.
    """
    ll_path = Path(outdir) / "whole_binary.ll"
    if not ll_path.exists():
        return {}
    lines = ll_path.read_text(errors="replace").splitlines()
    histograms = {}
    for name, func_lines in _iter_ll_functions(lines):
        histogram = Counter()
        for line in func_lines[1:-1]:
            if _LL_LABEL_RE.match(line):
                continue
            opcode = _ll_line_opcode(line)
            if opcode in _LL_KNOWN_OPCODES:
                histogram[opcode] += 1
        histograms[name] = histogram
    return histograms


def _function_data(backend, outdir):
    """(names_ok, histograms): names_ok is every successfully-lifted
    function's name (the precondition/match-rate universe); histograms is
    name -> op_histogram dict, only for functions that have one.
    """
    if backend == "retdec":
        histograms = _retdec_op_histograms(outdir)
        return set(histograms), histograms

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

        data_by_arch = {label: _function_data(backend, run["outdir"]) for label, run in by_arch.items()}

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


@register_block("jsd_similarity")
def compute_jsd_similarity(runs, results_dir):
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

        data_by_arch = {label: _function_data(backend, run["outdir"]) for label, run in by_arch.items()}

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
                jsd = jensen_shannon_similarity(hist_a[name], hist_b[name])
                if jsd is None:
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
                    "jsd_similarity": jsd,
                })
                overall.append(jsd)
                by_backend[backend].append(jsd)
                by_arch_pair[pair_label].append(jsd)

    return {
        "block": "jsd_similarity",
        "num_groups_evaluated": num_groups_evaluated,
        "num_functions_evaluated": len(rows),
        "metrics": {
            "jsd_similarity": {
                "direction": "higher_is_better", "range": "[0,1]",
                **(aggregate_stats(overall) or {"n": 0}),
            },
            "function_match_rate": {
                "direction": "descriptive", "range": "[0,1]",
                **(aggregate_stats(match_rates_overall) or {"n": 0}),
            },
        },
        "by_backend": {
            backend: {"jsd_similarity": aggregate_stats(vals)}
            for backend, vals in by_backend.items()
        },
        "by_arch_pair": {
            pair: {
                "jsd_similarity": aggregate_stats(vals),
                "function_match_rate": aggregate_stats(match_rates_by_pair[pair]),
            }
            for pair, vals in by_arch_pair.items()
        },
        "rows": rows,
    }
