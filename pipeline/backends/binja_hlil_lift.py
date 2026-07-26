"""List functions and lift them to HLIL using Binary Ninja headless.

Standalone CLI:
    python -m pipeline.backends.binja_hlil_lift --binary <path> --outdir <dir>

python3 -m pipeline.backends.binja_hlil_lift \
    --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf --outdir results/ls_x86_32/binja_hlil
"""

import sys
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
from pipeline.binja_il_classify import (
    bnil_instruction_depth,
    classify_il_function_ast,
    classify_il_function_escape,
    classify_il_function_op_histogram,
    classify_il_function_ops,
    classify_native_instructions,
    is_bnil_temp_var,
)

from binaryninja import load

CATEGORIES = ("arithmetic", "control", "memory", "other")


def list_functions(bv):
    functions = []
    for func in bv.functions:
        functions.append(
            make_function_entry(
                name=func.name,
                address=func.start,
                size=func.total_bytes,
            )
        )
    return functions


def lift_function(func, bv):
    hlil = func.hlil
    lines = [f"; ---- function {func.name} @ {hex(func.start)} ----"]
    num_instructions = 0
    max_nesting_depth = 0
    sum_nesting_depth = 0
    for insn in hlil.instructions:
        lines.append(f"  0x{insn.address:x}  {insn}")
        num_instructions += 1
        # Nesting-depth metric: how deep this one instruction's own
        # expression tree goes (see binja_il_classify.bnil_instruction_depth).
        # HLIL is where this is most informative.
        depth = bnil_instruction_depth(insn)
        max_nesting_depth = max(max_nesting_depth, depth)
        sum_nesting_depth += depth
    text = "\n".join(lines)
    # func.instructions for expansion ratio.
    num_native_instructions = sum(1 for _ in func.instructions)
    ir_ops_counts = classify_il_function_ops(hlil)
    ir_ast_counts = classify_il_function_ast(hlil)
    native_counts = classify_native_instructions(func, bv)
    # Temporaries metric: same promoted-Variable encoding as MLIL (see
    # pipeline/binja_il_classify.is_bnil_temp_var) -- expect this to be small
    # and often zero, since HLIL's expression-inlining eliminates most
    # surviving MLIL temps.
    num_temp_vars = sum(1 for v in hlil.vars if is_bnil_temp_var(v))
    # Escape-valve metric: see pipeline/binja_il_classify.py's
    # classify_il_function_escape / pipeline/metrics/blocks/robustness.py.
    num_escape_ops = classify_il_function_escape(hlil)
    # SSA-operations metric: hlil.ssa_form is a cheap, already-computed
    # alternate view of this same function.
    num_ssa_instructions = sum(1 for _ in hlil.ssa_form.instructions)
    # Agnosticism metric: op-type frequency histogram, compared across
    # architecture builds of the same binary by pipeline/metrics/blocks/
    # agnosticism.py's weighted_jaccard.
    op_histogram = classify_il_function_op_histogram(hlil)
    return (
        num_instructions, num_native_instructions, ir_ops_counts, ir_ast_counts, native_counts,
        num_temp_vars, num_escape_ops, max_nesting_depth, sum_nesting_depth, num_ssa_instructions,
        op_histogram, text,
    )


def run(binary_path, outdir, limit):
    fatal_error = None
    functions = []
    lifted_records = []

    with Timer() as timer:
        try:
            with load(str(binary_path)) as bv:
                if bv is None:
                    raise RuntimeError("Binary Ninja could not load the binary (unsupported format?)")

                functions = list_functions(bv)
                write_json(outdir / "functions.json", {
                    "binary": str(binary_path),
                    "backend": "binja_hlil",
                    "functions": functions,
                })

                target_addrs = {int(f["address"], 16) for f in (functions if limit is None else functions[:limit])}

                whole_binary_chunks = []
                for func in bv.functions:
                    if func.start not in target_addrs:
                        continue
                    record = {"function": func.name, "address": hex(func.start)}
                    try:
                        (
                            n_instr, n_native, ir_ops_counts, ir_ast_counts, native_counts,
                            n_temp_vars, n_escape_ops, max_nesting_depth, sum_nesting_depth, n_ssa_instr,
                            op_histogram, text,
                        ) = lift_function(func, bv)
                        record["status"] = "ok"
                        record["num_hlil_instructions"] = n_instr
                        record["num_native_instructions"] = n_native
                        record["num_temp_vars"] = n_temp_vars
                        record["num_escape_ops"] = n_escape_ops
                        record["max_nesting_depth"] = max_nesting_depth
                        record["sum_nesting_depth"] = sum_nesting_depth
                        record["num_ssa_instructions"] = n_ssa_instr
                        record["op_histogram"] = dict(op_histogram)
                        for cat in CATEGORIES:
                            record[f"ir_ops_{cat}"] = ir_ops_counts[cat]
                            record[f"ir_ast_{cat}"] = ir_ast_counts[cat]
                            record[f"native_{cat}"] = native_counts[cat]
                        whole_binary_chunks.append((func.start, text))
                    except Exception as e:
                        record["status"] = "error"
                        record["error"] = f"{type(e).__name__}: {e}"
                    lifted_records.append(record)

                write_whole_binary_dump(outdir, "whole_binary.hlil.txt", whole_binary_chunks)
        except Exception as e:
            fatal_error = f"{type(e).__name__}: {e}"

    summary = write_summary(outdir, "binja_hlil", binary_path, len(functions), lifted_records, timer.duration_s, fatal_error)
    write_json(outdir / "lift_records.json", lifted_records)

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to HLIL with Binary Ninja")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit)


if __name__ == "__main__":
    sys.exit(main())
