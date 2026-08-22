"""Temporaries metrics block: how many temporary variables/registers each
backend's IR introduces per native instruction, per (binary, backend,
function). Not in possible_metrics.txt yet, but the same shape as the
expansion_ratio block's expansion ratio (IR count / native instruction count) --
just counting distinct temporaries instead of IR ops.

Compound native instructions often get decomposed into several intermediate
steps during lifting, each needing a temporary to hold a partial value (e.g.
x86's `div` splits quotient and remainder into two temps before they're
copied to eax/edx) -- this metric captures how much of that "unsugaring"
each IR does, independent of how many total IR ops it produces (a backend
could be verbose in ops without inventing many temporaries, or vice versa).

What counts as a "temporary" is backend-specific, since each IR's own notion
of a synthetic/compiler-invented variable differs -- see each backend's
lift_function for the exact extraction:
  - P-code (pyghidra_lift.py): varnodes Ghidra's decompiler defines into its
    own "unique" address space, counted at *definition* time rather than via
    a dedup set keyed by raw offset -- that offset is reused across many
    genuinely distinct temporaries within one function (confirmed against
    this project's own binaries: one offset was the defining output of 18
    different, unrelated PcodeOps in a single function), so deduping by
    offset would drastically undercount. Ghidra's SSA property guarantees
    each unique-space value has exactly one defining PcodeOp, so counting
    definitions is both simpler and exact.
  - VEX (angr_lift.py): pyvex's own per-IRSB temp count,
    `len(irsb.tyenv.types)`, summed across the function's blocks -- IRTemp
    numbering (t0, t1, ...) is local to each basic block, not global to the
    function, so a single dedup set across the whole function would wrongly
    collapse unrelated blocks' t0s together.
  - BNIL (binja_*_lift.py via pipeline/binja_il_classify.is_bnil_temp_var):
    LLIL exposes exact built-in counts (`temp_reg_count`/`temp_flag_count`).
    MLIL/HLIL have no such property -- Binary Ninja instead promotes an
    unresolved LLIL temp into a full Variable that keeps the high bit of its
    LLIL temp encoding set, so those are detected via that bit instead.
    Expect MLIL/HLIL counts to be small and often zero -- HLIL's
    expression-inlining eliminates most of them.
  - LLVM IR (retdec): unnamed SSA values (a bare "%12" destination token, as
    opposed to a named "%some_var") in RetDec's textual .ll output --
    computed here at metrics time from whole_binary.ll, the same convention
    this project already uses for retdec's category counts (see
    expansion_ratio.py's module docstring for why: RetDec has no live session to
    reuse, so parsing the on-disk .ll happens at metrics time instead of
    inside retdec_lift.py's Timer()-wrapped run).
  - Hex-Rays microcode (ida_lift.py): mop_l ("local variable") operands,
    filtered to exclude real arguments (lvar.is_arg_var) and real
    named/debug-recovered locals (lvar.has_user_name) -- once Hex-Rays
    reaches its final SSA form, *every* local becomes a mop_l operand, not
    just genuine compiler temporaries (confirmed: 17 mop_l vars in a real
    decompiled `main()`, only 8 of which were actually unnamed/synthetic).
    Counting all mop_l without this filter would measure "how many locals
    exist" more than "how much unsugaring happened".
  - ESIL (r2): always 0 by design -- ESIL is a stack-based representation
    with no operator that declares a named temporary (confirmed against
    radare2's own `ae???` operator table: every operator is a stack math/
    compare/memory/control op, none of them a temp declaration). This is a
    meaningful data point (ESIL genuinely has no notion of a named
    temporary), not a measurement gap.

Only successful runs are counted, same convention as expansion_ratio.py. A
function's row is skipped entirely if no ratio is computable (native
instruction count missing or zero), not reported as a 0 -- matching
expansion_ratio.py's "no data" vs. "genuinely zero" distinction.
"""

from collections import defaultdict
from pathlib import Path

from pipeline.metrics.blocks.expansion_ratio import _load_lift_records, _retdec_native_category_counts
from pipeline.metrics.formulas import aggregate_stats, aggregate_stats_with_iqr, expansion_ratio
from pipeline.metrics.registry import register_block

METRICS = {
    "temp_vars_per_instr": {"direction": "descriptive", "unit": "temp_vars / native_instr"},
}


def _retdec_temp_var_counts(outdir):
    """name -> count of unnamed SSA values (a bare "%12" destination, LLVM's
    own auto-numbered temporaries, as opposed to a named "%some_var")
    defined within that function's whole_binary.ll block. Reuses the same
    brace-depth per-function split retdec_lift.py's own split_ll_by_function
    / expansion_ratio.py's _retdec_ir_category_counts already use, since RetDec
    has no live session to reuse (see this module's docstring).
    """
    from pipeline.backends.retdec_lift import LL_DEFINE_RE

    ll_path = Path(outdir) / "whole_binary.ll"
    if not ll_path.exists():
        return {}
    lines = ll_path.read_text(errors="replace").splitlines()
    counts = {}

    i = 0
    while i < len(lines):
        match = LL_DEFINE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        name = match["name"]
        start = i
        depth = 0
        j = i
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            j += 1
            if depth == 0 and j > start:
                break
        n_temp = 0
        for line in lines[start:j]:
            stripped = line.strip()
            if stripped.startswith("%") and "=" in stripped:
                token = stripped[1:].split("=", 1)[0].strip()
                if token.isdigit():
                    n_temp += 1
        counts[name] = n_temp
        i = j
    return counts


@register_block("temporaries")
def compute(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok"]

    overall = []
    by_backend = defaultdict(list)
    by_backend_arch = defaultdict(lambda: defaultdict(list))
    rows = []

    for run in completed:
        backend = run["backend"]
        meta = run.get("binary_meta") or {}
        arch, bits = meta.get("arch"), meta.get("bits")
        arch_bits = f"{arch}_{bits}" if arch is not None and bits is not None else None

        retdec_temp_counts = retdec_native_counts = None
        if backend == "retdec":
            retdec_temp_counts = _retdec_temp_var_counts(run["outdir"])
            retdec_native_counts = _retdec_native_category_counts(
                run["outdir"], run["binary"], meta.get("arch"), meta.get("bits")
            )

        for record in _load_lift_records(run["outdir"]):
            if record.get("status") != "ok":
                continue
            function = record.get("function")

            if backend == "retdec":
                n_temp = retdec_temp_counts.get(function, 0)
                total_native = sum(retdec_native_counts.get(function, {}).values())
            else:
                n_temp = record.get("num_temp_vars")
                total_native = record.get("num_native_instructions")

            ratio = expansion_ratio(n_temp, total_native)
            if ratio is None:
                continue

            rows.append({
                "binary": Path(run["binary"]).name,
                "backend": backend,
                "arch": arch,
                "bits": bits,
                "opt": meta.get("opt"),
                "compiler": meta.get("compiler"),
                "function": function,
                "num_temp_vars": n_temp,
                "num_native_instructions": total_native,
                "temp_vars_per_instr": ratio,
            })
            overall.append(ratio)
            by_backend[backend].append(ratio)
            if arch_bits is not None:
                by_backend_arch[backend][arch_bits].append(ratio)

    return {
        "block": "temporaries",
        "num_runs_ok": len(completed),
        "num_runs_total": len(runs),
        "num_functions_evaluated": len(rows),
        "metrics": {
            "temp_vars_per_instr": {**METRICS["temp_vars_per_instr"], **(aggregate_stats(overall) or {"n": 0})},
        },
        "by_backend": {
            backend: {
                "temp_vars_per_instr": aggregate_stats(vals),
                "by_arch": {
                    arch_bits: {"temp_vars_per_instr": aggregate_stats_with_iqr(arch_vals)}
                    for arch_bits, arch_vals in by_backend_arch.get(backend, {}).items()
                },
            }
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
