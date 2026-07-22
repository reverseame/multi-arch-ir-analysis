"""List functions and lift them to Hex-Rays microcode using IDA Pro's headless
idalib (the `idapro` Python package).

"Microcode" here is the final, fully matured mba_t produced by
ida_hexrays.decompile() (cfunc.mba) -- the same microcode the decompiler's own
pseudocode view is built from, after all of Hex-Rays' own optimization/SSA
passes. This is the closest IDA-side analog to pyghidra_lift.py's high P-code,
which is likewise pulled from a full decompiler run rather than a raw
per-instruction IR.

Standalone CLI:
    python -m pipeline.backends.ida_lift --binary <path> --outdir <dir>

python3 -m pipeline.backends.ida_lift \
    --binary binaries/ls/x86/32/O0/coreutils-8.29_gcc-8.2.0_x86_32_O0_ls.elf --outdir results/ls_x86_32/ida

Requires an activated `idapro` (idalib) install: from the IDA Pro installation
directory, `pip install idalib/python`, then run `py-activate-idalib.py` once
to point it at that installation. `import idapro` must happen before any
other `ida_*` import (it loads the IDA kernel and extends sys.path for them).
The other option is to use Headless IDA with the IDA Pro installation.
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
from pipeline.native_classify import arch_family, classify_bytes, make_disassembler

import idapro

import ida_auto
import ida_bytes
import ida_funcs
import ida_hexrays
import ida_ida
import ida_idaapi
import ida_segment
import idautils

CATEGORIES = ("arithmetic", "control", "memory", "other")

# mcode_t opcode name -> category, matching the other backends' IR taxonomy:
# real ALU/compare/float work is "arithmetic", memory traffic (including
# push/pop's implicit stack access) is "memory", branches/calls/returns are
# "control", and pure bookkeeping/type machinery (casts, truncation/extension,
# plain register moves, ldc/nop/und) is "other" -- the same convention
# pyghidra_lift.py uses for P-code's COPY/CAST/INT_ZEXT/INT_SEXT.
_MCODE_CONTROL = {
    "m_call", "m_icall", "m_goto", "m_ijmp", "m_ret",
    "m_ja", "m_jae", "m_jb", "m_jbe", "m_jcnd", "m_jg", "m_jge", "m_jl", "m_jle",
    "m_jnz", "m_jtbl", "m_jz",
}
_MCODE_MEMORY = {"m_ldx", "m_stx", "m_push", "m_pop"}
_MCODE_ARITHMETIC = {
    "m_add", "m_sub", "m_mul", "m_udiv", "m_sdiv", "m_umod", "m_smod",
    "m_and", "m_or", "m_xor", "m_bnot", "m_lnot", "m_neg",
    "m_shl", "m_shr", "m_sar", "m_cfadd", "m_cfshl", "m_cfshr", "m_ofadd",
    "m_seta", "m_setae", "m_setb", "m_setbe", "m_setg", "m_setge", "m_setl", "m_setle",
    "m_setnz", "m_seto", "m_setp", "m_sets", "m_setz",
    "m_fadd", "m_fsub", "m_fmul", "m_fdiv", "m_fneg",
}

# ida_ida.inf_get_procname() -> this project's metadata.json arch name.
# MIPS endianness is a separate processor module in IDA ("mipsl"/"mipsb")
# rather than a flag, mirroring how BinKit/this project split it into two
# arch names ("mips"=little, "mipseb"=big) instead of one arch + endian bit.
_IDA_ARCH_KEY = {"metapc": "x86", "ARM": "arm", "mipsl": "mips", "mipsb": "mipseb"}


def _mcode_categories():
    """int mcode_t opcode -> category, built from ida_hexrays's own m_*
    constants rather than hardcoding numeric opcode values, which aren't a
    stable ABI across IDA versions.
    """
    categories = {}
    for name in dir(ida_hexrays):
        if not name.startswith("m_"):
            continue
        value = getattr(ida_hexrays, name)
        if name in _MCODE_CONTROL:
            categories[value] = "control"
        elif name in _MCODE_MEMORY:
            categories[value] = "memory"
        elif name in _MCODE_ARITHMETIC:
            categories[value] = "arithmetic"
        else:
            categories[value] = "other"
    return categories


def ida_arch_key():
    """(arch, bits) in this project's metadata.json convention (see
    pipeline/native_classify.py) for the currently open database, or None if
    unsupported.
    """
    arch = _IDA_ARCH_KEY.get(ida_ida.inf_get_procname())
    if arch is None:
        return None
    bits = 64 if ida_ida.inf_is_64bit() else 32
    return (arch, bits)


def is_external_placeholder(f):
    """True for functions with no real code to decompile: IDA's synthetic
    entries for unresolved import targets, which live in a dedicated
    "extern" segment (SEG_XTRN) of their own.

    Deliberately NOT keyed off FUNC_THUNK: a real .plt trampoline is also
    flagged FUNC_THUNK (it thunks to that placeholder) but has actual
    instruction bytes in the binary's own .plt section, so FUNC_THUNK alone
    would wrongly also skip those -- mirrors pyghidra_lift.py's
    is_external_placeholder, which hits this exact pitfall with Ghidra's
    isThunk() and checks segment/block membership instead.
    """
    seg = ida_segment.getseg(f.start_ea)
    return seg is not None and seg.type == ida_segment.SEG_XTRN


def list_functions():
    functions = []
    for fea in idautils.Functions():
        f = ida_funcs.get_func(fea)
        functions.append(
            make_function_entry(
                name=ida_funcs.get_func_name(fea),
                address=fea,
                size=int(f.end_ea - f.start_ea),
                external=bool(is_external_placeholder(f)),
                thunk=bool(f.flags & ida_funcs.FUNC_THUNK),
            )
        )
    return functions


def _walk_operand_ast(op, mcode_categories, ir_ast_counts):
    """Recurse into one mop_t (or mop_t-like: mcallarg_t/mop_addr_t share
    the same .t/.d/.f/.a/.pair interface) that may itself hold an embedded
    sub-instruction.

    mop_d wraps another minsn_t directly (see _walk_minsn_ast's docstring);
    mop_f (call info) can hold further embedded instructions in its own
    argument list; mop_a ("address of") wraps the operand it takes the
    address of; mop_p (a lo/hi register pair, e.g. a 64-bit value on a
    32-bit target) wraps two further operands. Verified against this
    project's own real binaries via a live idalib session that all four
    occur and expose these attributes. Any other operand kind (register,
    stack var, immediate, ...) is a true leaf -- nothing to recurse into.
    """
    if op.t == ida_hexrays.mop_d:
        _walk_minsn_ast(op.d, mcode_categories, ir_ast_counts)
    elif op.t == ida_hexrays.mop_f:
        for arg in op.f.args:
            _walk_operand_ast(arg, mcode_categories, ir_ast_counts)
    elif op.t == ida_hexrays.mop_a:
        _walk_operand_ast(op.a, mcode_categories, ir_ast_counts)
    elif op.t == ida_hexrays.mop_p:
        _walk_operand_ast(op.pair.lop, mcode_categories, ir_ast_counts)
        _walk_operand_ast(op.pair.hop, mcode_categories, ir_ast_counts)


def _walk_minsn_ast(insn, mcode_categories, ir_ast_counts):
    """Classify one minsn_t by its own opcode, then recurse into its
    operands for any embedded sub-instruction -- the ast-granularity walk
    (see classify_vex_statement_ast/classify_il_function_ast for the same
    concept in angr_lift.py/binja_il_classify.py), kept alongside the flat
    ops-granularity count in lift_function so expansion-ratio can be
    reported at both, matching the rest of this project's nesting-capable
    backends.

    Hex-Rays microcode often nests a call (or other instruction) as an
    operand of another instruction via mop_d instead of always emitting it
    as its own top-level minsn_t in the block -- e.g. `mov call $foo() =>
    result, ret` is ONE linked-list instruction textually, but two real
    micro-ops: the outer mov and the nested call. Classifying only the
    outer opcode (what the ops granularity does, by design) folds the
    nested op into whatever category the wrapping instruction falls into
    -- confirmed against a real binary in this project's own corpus, where
    ~55% of all call sub-instructions are embedded this way and would
    land in "other" instead of "control" if never walked at all.
    """
    ir_ast_counts[mcode_categories.get(insn.opcode, "other")] += 1
    for op in (insn.l, insn.r, insn.d):
        _walk_operand_ast(op, mcode_categories, ir_ast_counts)


def lift_function(f, mcode_categories, md, family):
    hf = ida_hexrays.hexrays_failure_t()
    cfunc = ida_hexrays.decompile(f.start_ea, hf)
    if cfunc is None:
        raise RuntimeError(hf.desc() or "decompilation failed")

    mba = cfunc.mba
    lines = [f"; ---- function {ida_funcs.get_func_name(f.start_ea)} @ {hex(f.start_ea)} ----"]
    total_ops = 0
    current_ea = None
    ir_ops_counts = {c: 0 for c in CATEGORIES}
    ir_ast_counts = {c: 0 for c in CATEGORIES}
    for i in range(mba.qty):
        insn = mba.get_mblock(i).head
        while insn:
            if insn.ea != ida_idaapi.BADADDR and insn.ea != current_ea:
                lines.append(f"; 0x{insn.ea:x}")
                current_ea = insn.ea
            lines.append(f"  {insn.dstr()}")
            total_ops += 1
            ir_ops_counts[mcode_categories.get(insn.opcode, "other")] += 1
            _walk_minsn_ast(insn, mcode_categories, ir_ast_counts)
            insn = insn.next

    # Native disassembly instruction count over the function's own items
    native_counts = {c: 0 for c in CATEGORIES}
    num_native_instructions = 0
    for item_ea in idautils.FuncItems(f.start_ea):
        size = ida_bytes.get_item_size(item_ea)
        data = ida_bytes.get_bytes(item_ea, size)
        num_native_instructions += 1
        if md is not None and data:
            decoded = next(classify_bytes(md, data, item_ea, family), None)
            if decoded is not None:
                native_counts[decoded[3]] += 1

    text = "\n".join(lines)
    return total_ops, num_native_instructions, ir_ops_counts, ir_ast_counts, native_counts, text


def run(binary_path, outdir, limit):
    functions = []
    lifted_records = []
    fatal_error = None

    with Timer() as timer:
        db_opened = False
        try:
            db_path = outdir / "ida_project" / "db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            rc = idapro.open_database(str(binary_path), True, args=f"-o{db_path}")
            if rc != 0:
                raise RuntimeError(f"idapro.open_database failed (rc={rc})")
            db_opened = True

            ida_auto.auto_wait()
            if not ida_hexrays.init_hexrays_plugin():
                raise RuntimeError("Hex-Rays decompiler unavailable for this binary's architecture/license")

            arch_key = ida_arch_key()
            md = make_disassembler(*arch_key) if arch_key else None
            family = arch_family(*arch_key) if arch_key else None
            mcode_categories = _mcode_categories()

            functions = list_functions()
            write_json(outdir / "functions.json", {
                "binary": str(binary_path),
                "backend": "ida_microcode",
                "functions": functions,
            })

            target = functions if limit is None else functions[:limit]
            target_eas = {int(func["address"], 16) for func in target}
            whole_binary_chunks = []

            for fea in idautils.Functions():
                if fea not in target_eas:
                    continue
                f = ida_funcs.get_func(fea)
                record = {"function": ida_funcs.get_func_name(fea), "address": hex(fea)}

                if is_external_placeholder(f):
                    record["status"] = "skipped"
                    record["reason"] = "external function, not decompiled"
                    lifted_records.append(record)
                    continue

                try:
                    n_ops, n_native, ir_ops_counts, ir_ast_counts, native_counts, text = lift_function(
                        f, mcode_categories, md, family
                    )
                    record["status"] = "ok"
                    record["num_microcode_ops"] = n_ops
                    record["num_native_instructions"] = n_native
                    for cat in CATEGORIES:
                        record[f"ir_ops_{cat}"] = ir_ops_counts[cat]
                        record[f"ir_ast_{cat}"] = ir_ast_counts[cat]
                        record[f"native_{cat}"] = native_counts[cat]
                    whole_binary_chunks.append((fea, text))
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"{type(e).__name__}: {e}"
                lifted_records.append(record)

            write_whole_binary_dump(outdir, "whole_binary.microcode.txt", whole_binary_chunks)
        except Exception as e:
            fatal_error = f"{type(e).__name__}: {e}"
        finally:
            if db_opened:
                idapro.close_database(False)

    summary = write_summary(outdir, "ida_microcode", binary_path, len(functions), lifted_records, timer.duration_s, fatal_error)
    write_json(outdir / "lift_records.json", lifted_records)

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}  skipped={summary['num_lifted_skipped']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to Hex-Rays microcode with IDA Pro (idalib)")
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    args.binary = args.binary.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit)


if __name__ == "__main__":
    sys.exit(main())
