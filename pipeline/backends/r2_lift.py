"""List functions and lift them to ESIL using radare2 via r2pipe.

Standalone CLI:
    python -m pipeline.backends.r2_lift --binary <path> --outdir <dir>

python3 -m pipeline.backends.r2_lift \
    --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf --outdir results/ls_x86_32/r2
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

import r2pipe  

ANALYSIS_COMMAND = "aaa"


def list_functions(r2):
    raw = r2.cmdj("aflj") or []
    functions = []
    for func in raw:
        addr = func.get("addr", func.get("offset"))
        if addr is None:
            continue
        functions.append(
            make_function_entry(
                name=func.get("name", hex(addr)),
                address=addr,
                size=func.get("size"),
                extra={"num_instructions": func.get("ninstrs")},
            )
        )
    return functions


def lift_function(r2, addr, func_name):
    data = r2.cmdj(f"pdfj @ {addr}")
    if not data or "ops" not in data:
        raise RuntimeError("pdfj returned no ops")

    ops = data["ops"]
    lines = [f"; ---- function {func_name} @ {hex(addr)} ----"]
    num_uncovered = 0
    for op in ops:
        op_addr = op.get("offset", op.get("addr", addr))
        mnem = op.get("opcode", "")
        esil = op.get("esil") or ""
        if not esil or op.get("type") in ("invalid", "unk", None):
            num_uncovered += 1
        lines.append(f"; 0x{op_addr:x}  {mnem}")
        lines.append(f"  {esil or '(no esil)'}")
        lines.append("")

    text = "\n".join(lines)
    return len(ops), num_uncovered, text


def run(binary_path, outdir, limit):
    fatal_error = None
    functions = []
    lifted_records = []

    with Timer() as timer:
        try:
            r2 = r2pipe.open(str(binary_path))
            try:
                print(f"Analyzing binary ({ANALYSIS_COMMAND})...")
                r2.cmd(ANALYSIS_COMMAND)

                functions = list_functions(r2)
                write_json(outdir / "functions.json", {
                    "binary": str(binary_path),
                    "backend": "r2_esil",
                    "functions": functions,
                })

                target = functions if limit is None else functions[:limit]
                whole_binary_chunks = []
                for func in target:
                    addr = int(func["address"], 16)
                    record = {"function": func["name"], "address": func["address"]}
                    try:
                        n_ops, n_uncovered, text = lift_function(r2, addr, func["name"])
                        record["status"] = "ok"
                        record["num_instructions"] = n_ops
                        record["num_uncovered"] = n_uncovered
                        whole_binary_chunks.append((addr, text))
                    except Exception as e:
                        record["status"] = "error"
                        record["error"] = f"{type(e).__name__}: {e}"
                    lifted_records.append(record)

                write_whole_binary_dump(outdir, "whole_binary.esil.txt", whole_binary_chunks)
            finally:
                r2.quit()
        except Exception as e:
            fatal_error = f"{type(e).__name__}: {e}"

    summary = write_summary(outdir, "r2_esil", binary_path, len(functions), lifted_records, timer.duration_s, fatal_error)
    write_json(outdir / "lift_records.json", lifted_records)

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to ESIL with radare2/r2pipe")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit)


if __name__ == "__main__":
    sys.exit(main())
