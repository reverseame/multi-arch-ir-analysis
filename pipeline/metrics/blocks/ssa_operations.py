"""SSA-operations metrics block: how many operations exist in a function's
Static Single Assignment form, per (binary, backend, function). A secondary/
observational metric, not part of the main cross-IR comparison (unlike
expansion_ratio.py/temporaries.py/nesting_depth.py) -- SSA form isn't
measurable the same way across every backend, so it can't be compared
apples-to-apples the way the ops/native ratios can.

Applicable to BNIL (binja_{llil,mlil,hlil}_lift.py) and, partially, VEX
(angr_lift.py):
  - LowLevelILFunction.ssa_form / MediumLevelILFunction.ssa_form /
    HighLevelILFunction.ssa_form are real, already-computed alternate views
    of the same function (BNGetLowLevelILSSAForm etc.) -- not a fresh
    analysis pass, so counting their instructions is cheap. Confirmed on
    this project's own corpus: SSA form always has *more* instructions than
    the non-SSA form (never equal), since SSA variable renaming/phi-node
    insertion (_SSA opcode variants) adds real ops rather than just
    relabeling existing ones.
  - VEX (angr_lift.py): pyvex's own IRStmt docs state outright that "SSA
    rules require each tmp is only assigned to once" -- every VEX temp is
    *always* trivially SSA by construction. Unlike BNIL there's no separate
    alternate-form view to count instructions in -- angr_lift.py already
    records num_temp_vars (irsb.tyenv.types, summed per function) and
    num_statements per function (temporaries.py's own fields), which this
    block reuses as-is via SSA_FIELDS/OPS_FIELDS, so no re-lift was needed
    to add this. But VEX's register file (Put/Get) is never SSA -- no
    renaming, no phi nodes, ever -- and is excluded from num_temp_vars by
    construction, so this systematically UNDERCOUNTS "real" SSA usage
    relative to BNIL's count and is not the same quantity: BNIL's
    ssa_expansion_ratio compares two full instruction-count views of the
    same function (SSA-form vs base, always >1); VEX's compares temp
    *declarations* against total statements (a temp/statement density, not
    an expansion of a second form -- typically <1, and not directly
    comparable in magnitude to BNIL's ratio despite sharing a field name).
    See the "caveats" key in this block's output. angr's own SSA machinery
    (angr.analyses.decompiler.ssailification) operates on AIL, a completely
    different, higher-level IR built by angr's decompiler pipeline, not
    something pyvex.lift()/this project's angr_lift.py touches -- wiring
    that in would mean lifting a new IR, which is why registers stay out of
    scope here rather than being added some other way.

Not implemented for the remaining backends, and not just as a placeholder --
each has a real, separate reason:
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
blocks. Both fields are reused as-is for VEX (see SSA_FIELDS below) so the
output schema stays uniform across backends; CAVEATS documents per-backend
where the *meaning* of those two fields diverges, and is included verbatim
in this block's output so it survives into anything built on top of it.
"""

from collections import defaultdict
from pathlib import Path

from pipeline.metrics.blocks.expansion_ratio import _load_lift_records
from pipeline.metrics.formulas import aggregate_stats, aggregate_stats_with_iqr, expansion_ratio
from pipeline.metrics.registry import register_block

# Which backends have a countable SSA-eligible subset (see module docstring),
# and the lift_records.json field holding that backend's own ops count to use
# as the denominator -- the same field expansion_ratio.py's IR_SIZE_FIELDS/
# nesting_depth.py's OPS_FIELDS use for binja, and temporaries.py's own field
# for angr, so ssa_expansion_ratio is directly comparable in shape (not
# necessarily in magnitude -- see CAVEATS) across backends.
OPS_FIELDS = {
    "binja_llil": "num_llil_instructions",
    "binja_mlil": "num_mlil_instructions",
    "binja_hlil": "num_hlil_instructions",
    "angr": "num_statements",
}

# The lift_records.json field holding each backend's own SSA-eligible count.
# BNIL: instructions in the real alternate SSA-form IL view. angr/VEX: temp
# *declarations* (irsb.tyenv.types) -- not an alternate view, see CAVEATS.
SSA_FIELDS = {
    "binja_llil": "num_ssa_instructions",
    "binja_mlil": "num_ssa_instructions",
    "binja_hlil": "num_ssa_instructions",
    "angr": "num_temp_vars",
}

# Per-backend caveat about what num_ssa_instructions/ssa_expansion_ratio
# actually measure here, carried into this block's JSON output (not just a
# code comment) so it isn't lost by anything downstream that consumes it.
CAVEATS = {
    "angr": (
        "VEX has no alternate SSA-form view: num_ssa_instructions is VEX temp "
        "declarations (irsb.tyenv.types), not a second instruction count. "
        "Registers (Put/Get) are excluded by design and are never SSA, so this "
        "systematically undercounts real SSA usage relative to BNIL. "
        "ssa_expansion_ratio is a temp/statement density here, not an "
        "expansion of a second form -- not comparable in magnitude to "
        "BNIL's ratio despite the shared field name."
    ),
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
    by_backend_arch = defaultdict(lambda: defaultdict(lambda: {key: [] for key in METRICS}))
    rows = []

    for run in completed:
        backend = run["backend"]
        ops_field = OPS_FIELDS[backend]
        ssa_field = SSA_FIELDS[backend]
        meta = run.get("binary_meta") or {}
        arch, bits = meta.get("arch"), meta.get("bits")
        arch_bits = f"{arch}_{bits}" if arch is not None and bits is not None else None

        for record in _load_lift_records(run["outdir"]):
            if record.get("status") != "ok":
                continue
            function = record.get("function")
            ops_count = record.get(ops_field)
            n_ssa = record.get(ssa_field)

            ratio = expansion_ratio(n_ssa, ops_count)
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
                "ops_count": ops_count,
                "num_ssa_instructions": n_ssa,
                "ssa_expansion_ratio": ratio,
            })
            overall["num_ssa_instructions"].append(n_ssa)
            overall["ssa_expansion_ratio"].append(ratio)
            by_backend[backend]["num_ssa_instructions"].append(n_ssa)
            by_backend[backend]["ssa_expansion_ratio"].append(ratio)
            if arch_bits is not None:
                by_backend_arch[backend][arch_bits]["num_ssa_instructions"].append(n_ssa)
                by_backend_arch[backend][arch_bits]["ssa_expansion_ratio"].append(ratio)

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
            backend: {
                **{key: aggregate_stats(vals[key]) for key in METRICS},
                "by_arch": {
                    arch_bits: {key: aggregate_stats_with_iqr(arch_vals[key]) for key in METRICS}
                    for arch_bits, arch_vals in by_backend_arch.get(backend, {}).items()
                },
                "caveat": CAVEATS.get(backend),
            }
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
