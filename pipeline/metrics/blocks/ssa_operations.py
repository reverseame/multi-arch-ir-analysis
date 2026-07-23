"""SSA-operations metrics block: how many operations exist in a function's
Static Single Assignment form, per (binary, backend, function). A secondary/
observational metric, not part of the main cross-IR comparison (unlike
expansion_ratio.py/temporaries.py/nesting_depth.py) -- SSA form isn't
measurable the same way across every backend, so it can't be compared
apples-to-apples the way the ops/native ratios can.

Only applicable to BNIL (binja_{llil,mlil,hlil}_lift.py) for now:
  - LowLevelILFunction.ssa_form / MediumLevelILFunction.ssa_form /
    HighLevelILFunction.ssa_form are real, already-computed alternate views
    of the same function (BNGetLowLevelILSSAForm etc.) -- not a fresh
    analysis pass, so counting their instructions is cheap. Confirmed on
    this project's own corpus: SSA form always has *more* instructions than
    the non-SSA form (never equal), since SSA variable renaming/phi-node
    insertion (_SSA opcode variants) adds real ops rather than just
    relabeling existing ones.

Not implemented for the other backends, and not just as a placeholder --
each has a real, separate reason:
  - VEX (angr_lift.py): pyvex's own IRStmt docs state outright that "SSA
    rules require each tmp is only assigned to once" -- every VEX temp is
    *always* trivially SSA by construction, not an optional alternate view,
    so there's no discriminating subset to count (same redundancy problem as
    P-code below). VEX's register file (Put/Get), on the other hand, is
    never SSA -- no renaming, no phi nodes, ever. angr's own SSA machinery
    (angr.analyses.decompiler.ssailification) operates on AIL, a completely
    different, higher-level IR built by angr's decompiler pipeline, not
    something pyvex.lift()/this project's angr_lift.py touches -- wiring
    that in would mean lifting a new IR, not extending the existing one.
  - P-code (pyghidra_lift.py): this project already only extracts *High*
    P-code (post-SSA-construction) -- see pyghidra_lift.py's own module
    docstring. Every op already counted is already SSA; there's no non-SSA
    subset in this project's own extraction to compare against.
  - Hex-Rays microcode (ida_lift.py): this project extracts the fully
    matured mba_t (MMAT_LVARS, see ida_lift.py's own docstring), where
    mop_l/lvar_ref_t already exist. An earlier, more SSA-like maturity stage
    (e.g. MMAT_GLBOPT3) uses different operand kinds entirely (mop_r/mop_S
    -- the SDK's own comments say these "exist until MMAT_LVARS" only) and
    isn't reachable without hooking a maturity-level callback
    (Hexrays_Hooks.maturity) to snapshot mba mid-optimization -- new
    extraction infrastructure, not addable as a metrics block alone.
  - ESIL (r2), LLVM IR (retdec): no SSA concept applicable (r2/ESIL is a
    flat stack machine with no variable renaming; retdec already reports
    LLVM IR, whose own SSA-ness is covered by expansion_ratio.py's ir_size
    already, and RetDec doesn't expose a separate non-SSA form to compare).

Reports num_ssa_instructions (raw count, informational) and
ssa_expansion_ratio (num_ssa_instructions / the backend's own non-SSA ops
count, reusing expansion_ratio() from formulas.py -- same "None if the
denominator is missing/zero" convention as every other ratio in this
project). Only successful runs are counted, same convention as the other
blocks.
"""

from collections import defaultdict
from pathlib import Path

from pipeline.metrics.blocks.expansion_ratio import _load_lift_records
from pipeline.metrics.formulas import aggregate_stats, expansion_ratio
from pipeline.metrics.registry import register_block

# Which backends have a real SSA-form alternate view (see module docstring),
# and the lift_records.json field holding that backend's own non-SSA ops
# count -- the same field expansion_ratio.py's IR_SIZE_FIELDS/nesting_depth.py's
# OPS_FIELDS use, so ssa_expansion_ratio is directly comparable in shape.
OPS_FIELDS = {
    "binja_llil": "num_llil_instructions",
    "binja_mlil": "num_mlil_instructions",
    "binja_hlil": "num_hlil_instructions",
}

METRICS = {
    "num_ssa_instructions": {"direction": "descriptive", "unit": "ops"},
    "ssa_expansion_ratio": {"direction": "descriptive", "unit": "ssa_ops / ops"},
}


@register_block("ssa_operations")
def compute(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok" and r["backend"] in OPS_FIELDS]

    overall = {key: [] for key in METRICS}
    by_backend = defaultdict(lambda: {key: [] for key in METRICS})
    rows = []

    for run in completed:
        backend = run["backend"]
        ops_field = OPS_FIELDS[backend]
        meta = run.get("binary_meta") or {}

        for record in _load_lift_records(run["outdir"]):
            if record.get("status") != "ok":
                continue
            function = record.get("function")
            ops_count = record.get(ops_field)
            n_ssa = record.get("num_ssa_instructions")

            ratio = expansion_ratio(n_ssa, ops_count)
            if ratio is None:
                continue

            rows.append({
                "binary": Path(run["binary"]).name,
                "backend": backend,
                "arch": meta.get("arch"),
                "bits": meta.get("bits"),
                "opt": meta.get("opt"),
                "compiler": meta.get("compiler"),
                "function": function,
                "ops_count": ops_count,
                "num_ssa_instructions": n_ssa,
                "ssa_expansion_ratio": ratio,
            })
            overall["num_ssa_instructions"].append(n_ssa)
            overall["ssa_expansion_ratio"].append(ratio)
            by_backend[backend]["num_ssa_instructions"].append(n_ssa)
            by_backend[backend]["ssa_expansion_ratio"].append(ratio)

    return {
        "block": "ssa_operations",
        "num_runs_ok": len(completed),
        "num_runs_total": len(runs),
        "num_functions_evaluated": len(rows),
        "metrics": {
            key: {**METRICS[key], **(aggregate_stats(overall[key]) or {"n": 0})}
            for key in METRICS
        },
        "by_backend": {
            backend: {key: aggregate_stats(vals[key]) for key in METRICS}
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
