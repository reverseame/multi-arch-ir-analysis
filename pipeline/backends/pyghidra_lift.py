"""List functions and lift them to high P-code using Ghidra's decompiler via pyghidra.

High P-code comes out of Ghidra's DecompInterface/HighFunction: unlike low
P-code (Instruction.getPcode()), it has been through SSA construction and
variable merging, so operations are grouped by function and no longer map
1:1 to individual native instructions (no per-instruction disassembly
comment -- only an address comment where the source address changes).

Standalone CLI:
    python -m pipeline.backends.pyghidra_lift --binary <path> --outdir <dir>

python3 -m pipeline.backends.pyghidra_lift --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf --outdir results/ls_x86_32/ghidra

Requires GHIDRA_INSTALL_DIR to be set (or pyghidra to find Ghidra some other
way) before the JVM starts.
"""

import sys
import traceback
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

import pyghidra

pyghidra.start()

from java.lang import Exception as JavaException
from ghidra.app.decompiler import DecompInterface, DecompileOptions
from ghidra.util.task import ConsoleTaskMonitor

DEFAULT_DECOMP_TIMEOUT_S = 60

CATEGORIES = ("arithmetic", "control", "memory", "other")

# High P-code opcode mnemonics (ghidra.program.model.pcode.PcodeOp.getMnemonic,
# enumerated 0-74 in this Ghidra build), classified the same way as the other
# backends' IR: real ALU/compare/float work is "arithmetic", memory traffic is
# "memory", branches/calls/returns are "control", and pure bookkeeping/type
# machinery (SSA phi-nodes, casts, zero/sign-extension, CALLOTHER pcode
# userops, ...) is "other".
_PCODE_CONTROL = {"BRANCH", "CBRANCH", "BRANCHIND", "CALL", "CALLIND", "CALLOTHER", "RETURN"}
_PCODE_MEMORY = {"LOAD", "STORE"}
_PCODE_ARITHMETIC = {
    "INT_EQUAL", "INT_NOTEQUAL", "INT_SLESS", "INT_SLESSEQUAL", "INT_LESS", "INT_LESSEQUAL",
    "INT_ADD", "INT_SUB", "INT_CARRY", "INT_SCARRY", "INT_SBORROW", "INT_2COMP", "INT_NEGATE",
    "INT_XOR", "INT_AND", "INT_OR", "INT_LEFT", "INT_RIGHT", "INT_SRIGHT",
    "INT_MULT", "INT_DIV", "INT_SDIV", "INT_REM", "INT_SREM",
    "BOOL_NEGATE", "BOOL_XOR", "BOOL_AND", "BOOL_OR",
    "FLOAT_EQUAL", "FLOAT_NOTEQUAL", "FLOAT_LESS", "FLOAT_LESSEQUAL", "FLOAT_NAN",
    "FLOAT_ADD", "FLOAT_DIV", "FLOAT_MULT", "FLOAT_SUB", "FLOAT_NEG", "FLOAT_ABS", "FLOAT_SQRT",
    "POPCOUNT", "LZCOUNT", "PTRADD", "PTRSUB",
}

_PCODE_LANG_ARCH_KEY = {
    "x86": "x86",
    "ARM": "arm",
    "AARCH64": "arm",
    "MIPS": "mips",
}


def classify_pcode_op(mnemonic):
    if mnemonic in _PCODE_CONTROL:
        return "control"
    if mnemonic in _PCODE_MEMORY:
        return "memory"
    if mnemonic in _PCODE_ARITHMETIC:
        return "arithmetic"
    return "other"


def pyghidra_arch_key(language):
    """(arch, bits) in this project's metadata.json convention (see
    pipeline/native_classify.py) for a Ghidra Language, from its
    "FAMILY:ENDIAN:BITS:VARIANT" LanguageID (e.g. "x86:LE:32:default",
    "ARM:LE:32:v8", "MIPS:BE:32:default"), or None if unsupported.

    ARM32 is always treated as non-Thumb (CS_MODE_ARM): Ghidra tracks
    per-instruction Thumb/ARM mode via a "TMode" context register, but every
    arm/32 binary in this project's own corpus disassembles as pure ARM (no
    Thumb interworking) when checked with r2 (`aflj`'s per-function "bits"
    was 32, never 16, across ls/arm/32 binaries) -- not worth the extra
    per-instruction context lookup for a mode that doesn't occur here.
    """
    parts = str(language.getLanguageID()).split(":")
    if len(parts) != 4:
        return None
    family, endian, bits_str, _variant = parts
    arch = _PCODE_LANG_ARCH_KEY.get(family)
    if arch is None:
        return None
    if arch == "mips" and endian == "BE":
        arch = "mipseb"
    return (arch, int(bits_str))


def varnode_high_str(vn, language):
    """Prefer the decompiler's merged HighVariable name (e.g. local_28,
    param_1) when one exists; fall back to the raw varnode representation
    for temporaries/constants the decompiler didn't name.
    """
    if vn is None:
        return "-"
    high = vn.getHigh()
    if high is not None:
        name = high.getName()
        if name and name != "UNNAMED":
            return f"{name}:{vn.getSize()}"
    return vn.toString(language)


def pcodeop_high_str(op, language):
    mnemonic = op.getMnemonic()
    out = op.getOutput()
    inputs = [op.getInput(i) for i in range(op.getNumInputs())]
    inputs_str = ", ".join(varnode_high_str(i, language) for i in inputs)
    if out is not None:
        return f"{varnode_high_str(out, language)} = {mnemonic} {inputs_str}"
    return f"{mnemonic} {inputs_str}"


def is_external_placeholder(func, memory):
    """True for functions with no real code to decompile: genuinely external
    functions, and Ghidra's synthetic placeholders for unresolved import
    targets (parked in a memory block literally named "EXTERNAL" since no
    library is loaded to back them).

    A real .plt trampoline is also isThunk() (it thunks to that placeholder)
    but has actual instruction bytes in the binary's own .plt section, so
    isThunk() alone would wrongly also skip those -- check block membership
    instead of relying on isThunk().
    """
    if func.isExternal():
        return True
    block = memory.getBlock(func.getEntryPoint())
    return block is not None and block.getName() == "EXTERNAL"


def list_functions(program):
    func_manager = program.getFunctionManager()
    memory = program.getMemory()
    functions = []
    for func in func_manager.getFunctions(True):
        functions.append(
            make_function_entry(
                name=func.getName(),
                address=str(func.getEntryPoint()),
                size=int(func.getBody().getNumAddresses()),
                external=bool(is_external_placeholder(func, memory)),
                thunk=bool(func.isThunk()),
            )
        )
    return functions


def lift_function(ifc, func, language, monitor, timeout_s, listing, md, family):
    result = ifc.decompileFunction(func, timeout_s, monitor)
    if not result.decompileCompleted():
        msg = result.getErrorMessage() or "timeout/cancelled"
        raise RuntimeError(f"decompilation not completed: {msg}")

    high_func = result.getHighFunction()
    if high_func is None:
        raise RuntimeError("decompiler produced no HighFunction")

    lines = [f"; ---- function {func.getName()} @ {func.getEntryPoint()} ----"]
    total_ops = 0
    current_addr = None
    ir_counts = {c: 0 for c in CATEGORIES}
    num_temp_vars = 0
    num_escape_ops = 0
    op_iter = high_func.getPcodeOps()
    while op_iter.hasNext():
        op = op_iter.next()
        addr = op.getSeqnum().getTarget()
        if addr != current_addr:
            lines.append(f"; {addr}")
            current_addr = addr
        lines.append(f"  {pcodeop_high_str(op, language)}")
        total_ops += 1
        mnemonic = op.getMnemonic()
        ir_counts[classify_pcode_op(mnemonic)] += 1
        # Escape-valve metric: CALLOTHER is P-code's own generic fallback for
        # semantics it can't express as a primitive op (e.g. x86 CPUID/RDTSC,
        # architecture-specific intrinsics)
        if mnemonic == "CALLOTHER":
            num_escape_ops += 1
        # Temporaries metric: count at *definition* time rather than
        # deduping by the unique-space varnode's raw offset -- that offset
        # is reused across many genuinely distinct temporaries within the
        # same function, so a set() keyed by offset would drastically
        # undercount. Ghidra's SSA property guarantees each unique-space
        # value has exactly one defining PcodeOp, so counting definitions is
        # both simpler and exact.
        out = op.getOutput()
        if out is not None and out.isUnique():
            num_temp_vars += 1

    # Native disassembly instruction count over the function's own address
    # range for obtaining expansion ratio.
    native_counts = {c: 0 for c in CATEGORIES}
    num_native_instructions = 0
    for insn in listing.getInstructions(func.getBody(), True):
        num_native_instructions += 1
        if md is not None:
            data = bytes(insn.getBytes())
            decoded = next(classify_bytes(md, data, int(insn.getMinAddress().getOffset()), family), None)
            if decoded is not None:
                native_counts[decoded[3]] += 1

    text = "\n".join(lines)
    return total_ops, num_native_instructions, ir_counts, native_counts, num_temp_vars, num_escape_ops, text


def run(binary_path, outdir, limit, decomp_timeout_s):
    project_dir = outdir / "ghidra_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_name = "lift"
    program_path = "/" + binary_path.name

    lifted_records = []
    functions = []
    fatal_error = None

    with Timer() as timer:
        try:
            with pyghidra.open_project(str(project_dir), project_name, create=True) as project:
                try:
                    program_ctx = pyghidra.program_context(project, program_path)
                    program = program_ctx.__enter__()
                    print(f"Reusing existing Ghidra program: {program_path}")
                except FileNotFoundError:
                    print(f"Importing binary into Ghidra project: {binary_path}")
                    loader = (
                        pyghidra.program_loader()
                        .project(project)
                        .source(str(binary_path))
                        .projectFolderPath("/")
                    )
                    with loader.load() as load_results:
                        load_results.save(pyghidra.task_monitor())
                    program_ctx = pyghidra.program_context(project, program_path)
                    program = program_ctx.__enter__()
                    print("Analyzing binary...")
                    pyghidra.analyze(program)

                try:
                    language = program.getLanguage()
                    arch_key = pyghidra_arch_key(language)
                    md = make_disassembler(*arch_key) if arch_key else None
                    family = arch_family(*arch_key) if arch_key else None

                    functions = list_functions(program)
                    write_json(outdir / "functions.json", {
                        "binary": str(binary_path),
                        "backend": "pyghidra_highpcode",
                        "functions": functions,
                    })

                    func_manager = program.getFunctionManager()
                    target = functions if limit is None else functions[:limit]
                    target_addrs = {f["address"] for f in target}
                    whole_binary_chunks = []

                    monitor = ConsoleTaskMonitor()
                    ifc = DecompInterface()
                    ifc.setOptions(DecompileOptions())
                    ifc.openProgram(program)
                    memory = program.getMemory()
                    listing = program.getListing()
                    try:
                        for func in func_manager.getFunctions(True):
                            if str(func.getEntryPoint()) not in target_addrs:
                                continue
                            record = {"function": func.getName(), "address": str(func.getEntryPoint())}

                            if is_external_placeholder(func, memory):
                                record["status"] = "skipped"
                                record["reason"] = "external function, not decompiled"
                                lifted_records.append(record)
                                continue

                            try:
                                n_ops, n_native, ir_counts, native_counts, n_temp_vars, n_escape_ops, text = lift_function(
                                    ifc, func, language, monitor, decomp_timeout_s, listing, md, family
                                )
                                record["status"] = "ok"
                                record["num_pcode_ops"] = n_ops
                                record["num_native_instructions"] = n_native
                                record["num_temp_vars"] = n_temp_vars
                                record["num_escape_ops"] = n_escape_ops
                                for cat in CATEGORIES:
                                    record[f"ir_ops_{cat}"] = ir_counts[cat]
                                    record[f"native_{cat}"] = native_counts[cat]
                                whole_binary_chunks.append((int(func.getEntryPoint().getOffset()), text))
                            except (JavaException, Exception) as e:
                                record["status"] = "error"
                                record["error"] = f"{type(e).__name__}: {e}"
                            lifted_records.append(record)
                    finally:
                        ifc.dispose()

                    write_whole_binary_dump(outdir, "whole_binary.highpcode.txt", whole_binary_chunks)
                finally:
                    program_ctx.__exit__(None, None, None)
        except Exception as e:
            fatal_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    summary = write_summary(outdir, "pyghidra_highpcode", binary_path, len(functions), lifted_records, timer.duration_s, fatal_error)
    write_json(outdir / "lift_records.json", lifted_records)

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}  skipped={summary['num_lifted_skipped']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to high P-code with Ghidra's decompiler (pyghidra)")
    parser.add_argument("--decomp-timeout", type=float, default=DEFAULT_DECOMP_TIMEOUT_S,
                         help=f"Per-function decompilation timeout in seconds (default: {DEFAULT_DECOMP_TIMEOUT_S})")
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    args.binary = args.binary.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit, args.decomp_timeout)


if __name__ == "__main__":
    sys.exit(main())
