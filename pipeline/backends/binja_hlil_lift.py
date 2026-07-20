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

from binaryninja import load


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


def lift_function(func):
    hlil = func.hlil
    lines = [f"; ---- function {func.name} @ {hex(func.start)} ----"]
    num_instructions = 0
    for insn in hlil.instructions:
        lines.append(f"  0x{insn.address:x}  {insn}")
        num_instructions += 1
    text = "\n".join(lines)
    # func.instructions for expansion ratio.
    num_native_instructions = sum(1 for _ in func.instructions)
    return num_instructions, num_native_instructions, text


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
                        n_instr, n_native, text = lift_function(func)
                        record["status"] = "ok"
                        record["num_hlil_instructions"] = n_instr
                        record["num_native_instructions"] = n_native
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
