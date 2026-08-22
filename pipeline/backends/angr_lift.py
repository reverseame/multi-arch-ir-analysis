"""List functions and lift them to VEX IR using angr/pyvex.

Standalone CLI:
    python -m pipeline.backends.angr_lift --binary <path> --outdir <dir>

Function discovery uses angr's CFGFast (recovers a function knowledge base),
kept separate from the actual per-block lifting call (pyvex.lift), which is
the piece being compared across backends.

python3 -m pipeline.backends.angr_lift \
    --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf --outdir results/ls_x86_32/angr
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common import (
    Timer,
    build_backend_arg_parser,
    make_function_entry,
    write_json,
    write_summary,
    write_whole_binary_dump,
)
from pipeline.native_classify import arch_family, classify_insn, cs_arch_for

import angr
import pyvex

CATEGORIES = ("arithmetic", "control", "memory", "other")

# VEX Iop_ operations that are ALU work (arithmetic/bitwise/shift/
# compare)
_VEX_ARITH_OP_PREFIXES = tuple(
    f"Iop_{p}" for p in (
        "Add", "Sub", "Mul", "Div", "Mod", "And", "Or", "Xor", "Not",
        "Shl", "Shr", "Sar", "Cmp", "Neg", "Clz", "Ctz", "Sqrt", "Rsqrt",
        "Abs", "Rol", "Ror",
    )
)

_ANGR_ARCH_KEY = {
    "X86": ("x86", 32),
    "AMD64": ("x86", 64),
    "ARMEL": ("arm", 32),
    "ARMHF": ("arm", 32),
    "AARCH64": ("arm", 64),
}


def angr_arch_key(arch):
    """(arch, bits) in this project's metadata.json convention (see
    pipeline/native_classify.py) for an angr/archinfo Arch object -- MIPS32/
    MIPS64 need memory_endness too since archinfo, unlike BinKit, doesn't
    fold endianness into the architecture name.
    """
    if arch.name in _ANGR_ARCH_KEY:
        return _ANGR_ARCH_KEY[arch.name]
    if arch.name in ("MIPS32", "MIPS64"):
        bits = 32 if arch.name == "MIPS32" else 64
        base = "mipseb" if arch.memory_endness == "Iend_BE" else "mips"
        return (base, bits)
    return None


def _classify_vex_expr(expr):
    """One VEX IRExpr node -> "arithmetic"/"memory"/"other". Never "control"
    -- VEX represents control flow only at the statement (Ist_Exit) and
    block-exit (irsb.next) level, never inside an expression node itself.
    """
    tag = expr.tag
    if tag == "Iex_Load":
        return "memory"
    if tag in ("Iex_Binop", "Iex_Unop") and expr.op.startswith(_VEX_ARITH_OP_PREFIXES):
        return "arithmetic"
    return "other"


def _classify_vex_statement_tag(stmt):
    """The statement's own node, by VEX statement tag alone. Ist_WrTmp and
    Ist_Put no longer inherit from their source expression's classification
    -- that expression (and everything nested inside it) is now walked and
    counted as its own separate node(s), see classify_vex_statement_ast.
    """
    tag = stmt.tag
    if tag == "Ist_Exit":
        return "control"
    if tag in ("Ist_Store", "Ist_StoreG", "Ist_LoadG", "Ist_CAS", "Ist_LLSC"):
        return "memory"
    return "other"  # Ist_WrTmp, Ist_Put, Ist_IMark, Ist_PutI, Ist_Dirty, Ist_AbiHint, Ist_NoOp, Ist_MBE


def classify_vex_statement_ops(stmt):
    """One VEX IRStmt -> a single "arithmetic"/"control"/"memory"/"other"
    label, by its own top-level node only: Ist_WrTmp/Ist_Put inherit from
    their *source expression's own top-level tag* (not descending into
    whatever that expression nests), everything else via
    _classify_vex_statement_tag. This is the ops-level / flat granularity --
    one label per statement, the same counting unit as num_statements --
    kept alongside classify_vex_statement_ast's deeper per-AST-node count so
    expansion-ratio can be reported at both granularities.
    """
    if stmt.tag in ("Ist_WrTmp", "Ist_Put"):
        return _classify_vex_expr(stmt.data)
    return _classify_vex_statement_tag(stmt)


def classify_vex_statement_ast(stmt):
    """One VEX IRStmt -> list of "arithmetic"/"control"/"memory"/"other"
    labels, one per AST node: the statement's own node
    (_classify_vex_statement_tag) plus every sub-expression in its tree,
    classified individually via _classify_vex_expr.

    stmt.child_expressions (pyvex.expr.IRExpr.child_expressions) is already
    fully recursive -- it descends into every IRExpr-valued slot and
    flattens the whole subtree -- so no manual recursion is needed here.
    This mirrors pipeline/binja_il_classify.py's classify_il_function_ast,
    which does the same full-expression-tree walk for LLIL/MLIL/HLIL via
    il_func.traverse().
    """
    labels = [_classify_vex_statement_tag(stmt)]
    labels.extend(_classify_vex_expr(expr) for expr in stmt.child_expressions)
    return labels


def _vex_op_name(expr):
    """VEX IRExpr -> op-identity string: the specific IROp (e.g. "Iop_Add32")
    for Binop/Unop/Triop/Qop nodes, since that's the actual operation being
    performed, not just its arity shape; the plain tag (e.g. "Iex_Load",
    "Iex_Const", "Iex_RdTmp") for everything else, where the tag alone
    already names the operation.
    """
    if expr.tag in ("Iex_Binop", "Iex_Unop", "Iex_Triop", "Iex_Qop"):
        return expr.op
    return expr.tag


def vex_statement_op_histogram(stmt):
    """{op_name: count} over stmt's own node (by VEX statement tag) and
    every nested sub-expression in its tree (via IRExpr.child_expressions,
    already fully recursive -- see classify_vex_statement_ast's own use of
    it). Used by the agnosticism metrics' weighted_jaccard (see
    pipeline/metrics/blocks/agnosticism.py) to compare op-frequency
    distributions across architecture builds of the same binary.
    """
    counts = Counter()
    counts[stmt.tag] += 1
    for expr in stmt.child_expressions:
        counts[_vex_op_name(expr)] += 1
    return counts


def vex_node_depth(node):
    """How many levels deep node's own expression tree nests -- 1 for a
    leaf/flat node with no nested IRExpr-valued slot, +1 for each level of
    embedded sub-expression (e.g. "t5 = Add32(t1,t2)" is depth 2: the WrTmp
    statement's own node, plus the Add32 nested inside it).
    """
    max_child_depth = 0
    for slot in node.__slots__:
        value = getattr(node, slot)
        if isinstance(value, pyvex.expr.IRExpr):
            max_child_depth = max(max_child_depth, vex_node_depth(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, pyvex.expr.IRExpr):
                    max_child_depth = max(max_child_depth, vex_node_depth(item))
    return 1 + max_child_depth


def list_functions(cfg):
    functions = []
    for func in cfg.kb.functions.values():
        functions.append(
            make_function_entry(
                name=func.name,
                address=func.addr,
                size=func.size,
                external=bool(func.is_simprocedure),
                extra={"num_blocks": len(list(func.blocks))},
            )
        )
    return functions


def lift_function(proj, func, cs_arch, family):
    # Cyclomatic-complexity metric: func.graph is angr's own local transition
    # graph for this function (block-level nodes, transition/exception/
    # fake_return edges only -- calls to *other* functions are excluded, and
    # a call's fallthrough is represented as a fake_return edge back into
    # this same graph), already cached from CFGFast -- verified live against
    # this project's own x86/32 `ls` corpus (main: 131 nodes/175 edges ->
    # M=46, a plausible complexity for a large branchy function). See
    # pipeline/metrics/blocks/agnosticism.py's cyclomatic_complexity_delta.
    cfg_graph = func.graph
    num_cfg_blocks = cfg_graph.number_of_nodes()
    num_cfg_edges = cfg_graph.number_of_edges()

    lines = [f"; ---- function {func.name} @ {hex(func.addr)} ----"]
    total_statements = 0
    total_native_instructions = 0
    total_vex_temps = 0
    num_escape_ops = 0
    max_nesting_depth = 0
    sum_nesting_depth = 0
    ir_ops_counts = {c: 0 for c in CATEGORIES}
    ir_ast_counts = {c: 0 for c in CATEGORIES}
    native_counts = {c: 0 for c in CATEGORIES}
    # Agnosticism metric: op-type frequency histogram, compared across
    # architecture builds of the same binary by pipeline/metrics/blocks/
    # agnosticism.py's weighted_jaccard.
    op_histogram = Counter()
    for block in func.blocks:
        data = proj.loader.memory.load(block.addr, block.size)
        irsb = pyvex.lift(data, block.addr, proj.arch)
        # Temporaries metric: pyvex's own per-IRSB temp count. IRTemp
        # numbering (t0, t1, ...) is local to each basic block, not global to
        # the function, so this is summed across blocks rather than deduped
        # into one set
        total_vex_temps += len(irsb.tyenv.types)
        lines.append(f"; ---- block 0x{block.addr:x} (size={block.size}) ----")
        for stmt in irsb.statements:
            lines.append(f"  {stmt}")
            ir_ops_counts[classify_vex_statement_ops(stmt)] += 1
            for label in classify_vex_statement_ast(stmt):
                ir_ast_counts[label] += 1
            op_histogram.update(vex_statement_op_histogram(stmt))
            # Escape-valve metric: Ist_Dirty is VEX's own generic fallback for
            # instructions libVEX can't express as primitive IR and instead
            # models as an opaque call to a "dirty helper" C function (e.g.
            # x86 CPUID/string ops, ARM NEON)
            if stmt.tag == "Ist_Dirty":
                num_escape_ops += 1
            # Nesting-depth metric: how deep this one statement's own
            # expression tree goes, summed/maxed across the function so the
            # metrics block can report both a typical (mean) and worst-case
            # (max) depth per function.
            depth = vex_node_depth(stmt)
            max_nesting_depth = max(max_nesting_depth, depth)
            sum_nesting_depth += depth
        lines.append(f"  NEXT: {irsb.next} ; jumpkind={irsb.jumpkind}")
        lines.append("")
        total_statements += len(irsb.statements)
        # The block's own terminating jump/call/ret/conditional-fallthrough
        # (irsb.next/jumpkind) isn't part of irsb.statements -- VEX always
        # keeps it separate -- so it needs to be added to the control count
        # by hand; skipping it would undercount "control" statements. At the
        # ops/flat granularity that's the whole story (one label for the
        # terminator). At the AST granularity, its target expression (a bare
        # Const for direct jumps, but a computed expression for indirect
        # jumps/calls through a register) is additionally walked the same
        # way as any other statement's expression tree.
        ir_ops_counts["control"] += 1
        ir_ast_counts["control"] += 1
        # The terminator's own op-identity is its jumpkind (Ijk_Call,
        # Ijk_Ret, Ijk_Boring, ...) -- a real, architecture-independent
        # signal of what kind of control transfer this is, unlike
        # irsb.next's own tag (Iex_Const/Iex_RdTmp), which only describes
        # how the target address is computed.
        op_histogram[irsb.jumpkind] += 1
        for expr in irsb.next.child_expressions:
            ir_ast_counts[_classify_vex_expr(expr)] += 1
            op_histogram[_vex_op_name(expr)] += 1

        # block.instructions for expansion ratio
        total_native_instructions += block.instructions
        if cs_arch is not None:
            for wrapped in block.capstone.insns:
                native_counts[classify_insn(wrapped.insn, cs_arch, family)] += 1

    text = "\n".join(lines)
    return (
        total_statements, total_native_instructions, ir_ops_counts, ir_ast_counts, native_counts,
        total_vex_temps, num_escape_ops, max_nesting_depth, sum_nesting_depth, op_histogram, text,
        num_cfg_blocks, num_cfg_edges,
    )


def run(binary_path, outdir, limit):
    fatal_error = None
    functions = []
    lifted_records = []

    with Timer() as timer:
        try:
            proj = angr.Project(str(binary_path), auto_load_libs=False)
            print("Discovering functions (CFGFast)...")
            cfg = proj.analyses.CFGFast(normalize=True, data_references=False)

            arch_key = angr_arch_key(proj.arch)
            cs_arch = cs_arch_for(*arch_key) if arch_key else None
            family = arch_family(*arch_key) if arch_key else None

            functions = list_functions(cfg)
            write_json(outdir / "functions.json", {
                "binary": str(binary_path),
                "backend": "angr_vex",
                "functions": functions,
            })

            target_addrs = {int(f["address"], 16) for f in (functions if limit is None else functions[:limit])}

            whole_binary_chunks = []
            for func in cfg.kb.functions.values():
                if func.addr not in target_addrs:
                    continue
                record = {"function": func.name, "address": hex(func.addr)}

                if func.is_simprocedure:
                    record["status"] = "skipped"
                    record["reason"] = "external/simprocedure function, not lifted"
                    lifted_records.append(record)
                    continue

                try:
                    (
                        n_stmts, n_native, ir_ops_counts, ir_ast_counts, native_counts,
                        n_temp_vars, n_escape_ops, max_nesting_depth, sum_nesting_depth, op_histogram, text,
                        n_cfg_blocks, n_cfg_edges,
                    ) = lift_function(proj, func, cs_arch, family)
                    record["status"] = "ok"
                    record["num_statements"] = n_stmts
                    record["num_native_instructions"] = n_native
                    record["num_temp_vars"] = n_temp_vars
                    record["num_escape_ops"] = n_escape_ops
                    record["max_nesting_depth"] = max_nesting_depth
                    record["sum_nesting_depth"] = sum_nesting_depth
                    record["num_cfg_blocks"] = n_cfg_blocks
                    record["num_cfg_edges"] = n_cfg_edges
                    record["op_histogram"] = dict(op_histogram)
                    for cat in CATEGORIES:
                        record[f"ir_ops_{cat}"] = ir_ops_counts[cat]
                        record[f"ir_ast_{cat}"] = ir_ast_counts[cat]
                        record[f"native_{cat}"] = native_counts[cat]
                    whole_binary_chunks.append((func.addr, text))
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"{type(e).__name__}: {e}"
                lifted_records.append(record)

            write_whole_binary_dump(outdir, "whole_binary.vex.txt", whole_binary_chunks)
        except Exception as e:
            fatal_error = f"{type(e).__name__}: {e}"

    summary = write_summary(outdir, "angr_vex", binary_path, len(functions), lifted_records, timer.duration_s, fatal_error)
    write_json(outdir / "lift_records.json", lifted_records)

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}  skipped={summary['num_lifted_skipped']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to VEX IR with angr/pyvex")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit)


if __name__ == "__main__":
    sys.exit(main())
