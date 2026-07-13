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
    safe_filename,
    write_json,
    write_summary,
    write_whole_binary_dump,
)

import pyghidra  

pyghidra.start()

from java.lang import Exception as JavaException  
from ghidra.app.decompiler import DecompInterface, DecompileOptions  
from ghidra.util.task import ConsoleTaskMonitor  

DEFAULT_DECOMP_TIMEOUT_S = 60


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


def lift_function(ifc, func, language, ir_dir, monitor, timeout_s):
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
    op_iter = high_func.getPcodeOps()
    while op_iter.hasNext():
        op = op_iter.next()
        addr = op.getSeqnum().getTarget()
        if addr != current_addr:
            lines.append(f"; {addr}")
            current_addr = addr
        lines.append(f"  {pcodeop_high_str(op, language)}")
        total_ops += 1

    text = "\n".join(lines)
    out_name = f"{safe_filename(func.getName())}_{func.getEntryPoint()}.txt".replace(":", "_")
    out_path = ir_dir / out_name
    out_path.write_text(text)
    return out_path, total_ops, text


def run(binary_path, outdir, limit, decomp_timeout_s):
    ir_dir = outdir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)

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
                                out_path, n_ops, text = lift_function(ifc, func, language, ir_dir, monitor, decomp_timeout_s)
                                record["status"] = "ok"
                                record["output_file"] = str(out_path.relative_to(outdir))
                                record["num_pcode_ops"] = n_ops
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
