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
    safe_filename,
    write_json,
    write_summary,
    write_whole_binary_dump,
)

import angr  
import pyvex  


def list_functions(cfg):
    functions = []
    for func in cfg.kb.functions.values():
        functions.append(
            make_function_entry(
                name=func.name,
                address=func.addr,
                size=func.size,
                extra={"num_blocks": len(list(func.blocks))},
            )
        )
    return functions


def lift_function(proj, func, ir_dir):
    lines = [f"; ---- function {func.name} @ {hex(func.addr)} ----"]
    total_statements = 0
    for block in func.blocks:
        data = proj.loader.memory.load(block.addr, block.size)
        irsb = pyvex.lift(data, block.addr, proj.arch)
        lines.append(f"; ---- block 0x{block.addr:x} (size={block.size}) ----")
        for stmt in irsb.statements:
            lines.append(f"  {stmt}")
        lines.append(f"  NEXT: {irsb.next} ; jumpkind={irsb.jumpkind}")
        lines.append("")
        total_statements += len(irsb.statements)

    text = "\n".join(lines)
    out_name = f"{safe_filename(func.name)}_{hex(func.addr)}.txt"
    out_path = ir_dir / out_name
    out_path.write_text(text)
    return out_path, total_statements, text


def run(binary_path, outdir, limit):
    ir_dir = outdir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)

    fatal_error = None
    functions = []
    lifted_records = []

    with Timer() as timer:
        try:
            proj = angr.Project(str(binary_path), auto_load_libs=False)
            print("Discovering functions (CFGFast)...")
            cfg = proj.analyses.CFGFast(normalize=True, data_references=False)

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
                try:
                    out_path, n_stmts, text = lift_function(proj, func, ir_dir)
                    record["status"] = "ok"
                    record["output_file"] = str(out_path.relative_to(outdir))
                    record["num_statements"] = n_stmts
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

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to VEX IR with angr/pyvex")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit)


if __name__ == "__main__":
    sys.exit(main())
