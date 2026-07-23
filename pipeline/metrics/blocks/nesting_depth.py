"""Nesting-depth metrics block: how many levels deep a single IR
statement/instruction's own expression tree goes, per (binary, backend,
function). A different axis from expansion_ratio.py's ast/ops node counts --
ast/ops measures total *breadth* (how many nodes exist across a function),
this measures *depth* (how deep any one statement's own tree gets) -- a
function could have many flat, wide statements (high ast/ops, low nesting
depth) or few narrow-but-deep ones (the reverse), and the two metrics
together distinguish those cases where either alone can't.

Only applicable to IRs that actually nest sub-expressions inside a single
statement/instruction -- confirmed per-backend by the ast/ops divergence
expansion_ratio.py already measures (an ast/ops ratio of 1.00x means every
statement is already a single flat node, nothing to measure depth of):
  - VEX (angr_lift.py): confirmed nesting (ast/ops ~1.7x on this project's
    own corpus). depth computed via angr_lift.vex_node_depth, a manual walk
    over pyvex IRStmt/IRExpr __slots__ -- IRExpr.child_expressions flattens
    the whole subtree into one list and discards tree shape, so it can't be
    reused for depth; the manual walk mirrors its same slot-recursion logic
    (including Qop/Triop/Binop/Unop/CCall's list-valued 'args' slot) but
    preserves levels instead of flattening them.
  - BNIL (binja_*_lift.py via pipeline/binja_il_classify.bnil_instruction_depth):
    confirmed nesting at all three levels (ast/ops ~2.9x/3.1x/3.8x for
    LLIL/MLIL/HLIL respectively on this project's own corpus -- HLIL nests
    the most, LLIL the least, matching how each optimization pass folds
    more sub-expressions into fewer top-level statements). Depth is a
    manual walk over detailed_operands (the same recursion
    Instruction.traverse() uses internally, see binaryninja's own
    lowlevelil.py/mediumlevelil.py/highlevelil.py), skipping HLIL's
    structural body/cases/default/true/false operands so a WHILE/SWITCH's
    entire loop/case body -- a different axis, CFG structure, not
    expression nesting -- doesn't inflate the number (confirmed empirically:
    an unfiltered walk found a HLIL_WHILE with "depth" 24 by walking its
    whole loop body as if it were one expression).
  - Hex-Rays microcode (ida_lift.py via _mcode_insn_depth/_mcode_operand_depth):
    confirmed nesting (ast/ops ~1.76x). Reuses the same four recursable
    operand kinds _walk_operand_ast already established (mop_d/mop_f/mop_a/
    mop_p), as a separate depth-returning walk rather than fused into the
    existing count-accumulating one.

Not applicable -- confirmed flat (ast/ops == 1.00x exactly, i.e. every
statement is already a single node with nothing nested inside it) on this
project's own corpus, so not implemented for these backends:
  - P-code (pyghidra_lift.py): each PcodeOp's inputs are always Varnodes
    (registers/constants/SSA-linked temporaries referencing another op's
    output by *link*, never an embedded op *value*) -- there is no
    expression-tree nesting model in High P-code to measure the depth of.
  - LLVM IR (retdec): three-address-code text, one opcode per line, operands
    reference other named/numbered SSA values, never embed another
    instruction inline.
  - ESIL (r2): a flat RPN token stream per instruction; ast/ops divergence
    doesn't even apply here the way it does for the other backends (see
    expansion_ratio.py's own module docstring on r2's ops being a rougher
    proxy than the rest), so there's no comparable per-statement tree to
    measure either.

Reports two metrics per function: mean_nesting_depth (sum_nesting_depth /
the backend's own ops-count field, reusing expansion_ratio() from
formulas.py -- same "None if the denominator is missing/zero" convention as
every other ratio in this project) and max_nesting_depth (the single
deepest statement/instruction found anywhere in the function, a worst-case
companion to the mean). Only successful runs are counted, same convention
as expansion_ratio.py/temporaries.py; a function's row is skipped entirely
if the ops-count denominator is missing or zero.
"""

from collections import defaultdict
from pathlib import Path

from pipeline.metrics.blocks.expansion_ratio import _load_lift_records
from pipeline.metrics.formulas import aggregate_stats, expansion_ratio
from pipeline.metrics.registry import register_block

# Which backends have a nesting-capable IR (see module docstring), and the
# lift_records.json field holding that backend's own top-level ops count --
# the same field expansion_ratio.py's IR_SIZE_FIELDS uses, so
# mean_nesting_depth is directly comparable in shape to expansion_ratio_ops.
OPS_FIELDS = {
    "angr": "num_statements",
    "binja_llil": "num_llil_instructions",
    "binja_mlil": "num_mlil_instructions",
    "binja_hlil": "num_hlil_instructions",
    "ida": "num_microcode_ops",
}

METRICS = {
    "mean_nesting_depth": {"direction": "descriptive", "unit": "levels/statement"},
    "max_nesting_depth": {"direction": "descriptive", "unit": "levels"},
}


@register_block("nesting_depth")
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

            mean_depth = expansion_ratio(record.get("sum_nesting_depth"), ops_count)
            if mean_depth is None:
                continue
            max_depth = record.get("max_nesting_depth")

            rows.append({
                "binary": Path(run["binary"]).name,
                "backend": backend,
                "arch": meta.get("arch"),
                "bits": meta.get("bits"),
                "opt": meta.get("opt"),
                "compiler": meta.get("compiler"),
                "function": function,
                "sum_nesting_depth": record.get("sum_nesting_depth"),
                "ops_count": ops_count,
                "mean_nesting_depth": mean_depth,
                "max_nesting_depth": max_depth,
            })
            overall["mean_nesting_depth"].append(mean_depth)
            overall["max_nesting_depth"].append(max_depth)
            by_backend[backend]["mean_nesting_depth"].append(mean_depth)
            by_backend[backend]["max_nesting_depth"].append(max_depth)

    return {
        "block": "nesting_depth",
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
