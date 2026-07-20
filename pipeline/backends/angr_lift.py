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
    tag = expr.tag
    if tag == "Iex_Load":
        return "memory"
    if tag in ("Iex_Binop", "Iex_Unop") and expr.op.startswith(_VEX_ARITH_OP_PREFIXES):
        return "arithmetic"
    return "other"


def classify_vex_statement(stmt):
    """One VEX IRStmt -> "arithmetic"/"control"/"memory"/"other". Ist_WrTmp
    and Ist_Put are classified by their *source expression* (a temp/register
    write is only as interesting as the value computed for it) so that
    e.g. `PUT(offset=68) = 0x08057613`
    """
    tag = stmt.tag
    if tag == "Ist_Exit":
        return "control"
    if tag in ("Ist_Store", "Ist_StoreG", "Ist_LoadG", "Ist_CAS", "Ist_LLSC"):
        return "memory"
    if tag in ("Ist_WrTmp", "Ist_Put"):
        return _classify_vex_expr(stmt.data)
    return "other"  # Ist_IMark, Ist_PutI, Ist_Dirty, Ist_AbiHint, Ist_NoOp, Ist_MBE


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
    lines = [f"; ---- function {func.name} @ {hex(func.addr)} ----"]
    total_statements = 0
    total_native_instructions = 0
    ir_counts = {c: 0 for c in CATEGORIES}
    native_counts = {c: 0 for c in CATEGORIES}
    for block in func.blocks:
        data = proj.loader.memory.load(block.addr, block.size)
        irsb = pyvex.lift(data, block.addr, proj.arch)
        lines.append(f"; ---- block 0x{block.addr:x} (size={block.size}) ----")
        for stmt in irsb.statements:
            lines.append(f"  {stmt}")
            ir_counts[classify_vex_statement(stmt)] += 1
        lines.append(f"  NEXT: {irsb.next} ; jumpkind={irsb.jumpkind}")
        lines.append("")
        total_statements += len(irsb.statements)
        # The block's own terminating jump/call/ret/conditional-fallthrough
        # (irsb.next/jumpkind) isn't part of irsb.statements -- VEX always
        # keeps it separate -- so it needs to be added to the control count
        # by hand; skipping it would undercount "control" statements
        ir_counts["control"] += 1

        # block.instructions for expansion ratio
        total_native_instructions += block.instructions
        if cs_arch is not None:
            for wrapped in block.capstone.insns:
                native_counts[classify_insn(wrapped.insn, cs_arch, family)] += 1

    text = "\n".join(lines)
    return total_statements, total_native_instructions, ir_counts, native_counts, text


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
                    n_stmts, n_native, ir_counts, native_counts, text = lift_function(proj, func, cs_arch, family)
                    record["status"] = "ok"
                    record["num_statements"] = n_stmts
                    record["num_native_instructions"] = n_native
                    for cat in CATEGORIES:
                        record[f"ir_ops_{cat}"] = ir_counts[cat]
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
