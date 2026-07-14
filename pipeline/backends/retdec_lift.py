"""List functions and lift them to LLVM IR using the RetDec CLI decompiler.

RetDec has no Python API: it's invoked as a subprocess that decompiles the
whole binary in one pass, producing (among others) a .dsm disassembly listing
and a .ll LLVM IR module. This script runs that pass once, then:
  - parses functions.json out of the .dsm "; function: NAME at START -- END" lines
  - splits the .ll module into one file per function under ir/

Standalone CLI:
    python -m pipeline.backends.retdec_lift --binary <path> --outdir <dir> \
        [--retdec-bin /path/to/retdec-decompiler] [--timeout 900]

python3 -m pipeline.backends.retdec_lift \
    --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf --outdir results/ls_x86_32/retdec \
    --retdec-bin /home/venator/Downloads/RetDec/bin/retdec-decompiler
"""

import os
import re
import shutil
import subprocess
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
)

DSM_FUNCTION_RE = re.compile(r"^; function: (?P<name>\S+) at (?P<start>0x[0-9a-fA-F]+) -- (?P<end>0x[0-9a-fA-F]+)")
LL_DEFINE_RE = re.compile(r"^define[^@]*@(?P<name>[\w.$]+)\s*\(")


def find_retdec_bin(explicit):
    if explicit:
        return str(explicit)
    env_bin = os.environ.get("RETDEC_BIN")
    if env_bin:
        return env_bin
    found = shutil.which("retdec-decompiler")
    if found:
        return found
    raise RuntimeError(
        "Could not find retdec-decompiler. Pass --retdec-bin, or set RETDEC_BIN, "
        "or add it to PATH."
    )


def list_functions_from_dsm(dsm_path):
    functions = []
    with open(dsm_path, "r", errors="replace") as f:
        for line in f:
            match = DSM_FUNCTION_RE.match(line)
            if not match:
                continue
            start = int(match["start"], 16)
            end = int(match["end"], 16)
            functions.append(make_function_entry(
                name=match["name"],
                address=start,
                size=end - start,
            ))
    return functions


def split_ll_by_function(ll_path, ir_dir):
    """Split a .ll module into one file per top-level `define` block, using
    brace depth to find each function's end (RetDec doesn't nest functions,
    but a naive line-range split without brace tracking would break on
    functions containing nested `{`/`}` in literals).
    """
    lines = Path(ll_path).read_text(errors="replace").splitlines()
    records = {}  # name -> (output_path, num_lines)

    i = 0
    while i < len(lines):
        match = LL_DEFINE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        name = match["name"]
        start = i
        depth = 0
        j = i
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            j += 1
            if depth == 0 and j > start:
                break
        block = lines[start:j]
        out_path = ir_dir / f"{safe_filename(name)}.ll"
        out_path.write_text("\n".join(block))
        records[name] = (out_path, len(block))
        i = j
    return records


def run(binary_path, outdir, limit, retdec_bin, timeout_s):
    ir_dir = outdir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)

    fatal_error = None
    functions = []
    lifted_records = []

    with Timer() as timer:
        try:
            retdec_bin = find_retdec_bin(retdec_bin)
            output_c = outdir / f"{binary_path.stem}.c"
            # By default RetDec drops functions it can't statically prove are
            # reachable from main (e.g. qsort comparators, signal handlers
            # invoked only through function pointers) and functions it
            # recognizes as library code (e.g. PLT trampolines) -- both are
            # real functions we still want IR for, so keep them.
            cmd = [
                retdec_bin, str(binary_path), "-o", str(output_c), "--cleanup",
                "--keep-unreachable-funcs", "--backend-keep-library-funcs",
            ]
            print(f"Running: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
            (outdir / "retdec_stdout.log").write_text(proc.stdout)
            (outdir / "retdec_stderr.log").write_text(proc.stderr)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"retdec-decompiler exited with code {proc.returncode}; "
                    f"see retdec_stderr.log"
                )

            dsm_path = output_c.with_suffix(".dsm")
            ll_path = output_c.with_suffix(".ll")
            if not dsm_path.exists() or not ll_path.exists():
                raise RuntimeError(f"Expected outputs missing: {dsm_path.name}, {ll_path.name}")

            functions = list_functions_from_dsm(dsm_path)
            write_json(outdir / "functions.json", {
                "binary": str(binary_path),
                "backend": "retdec_llvmir",
                "functions": functions,
            })

            ll_functions = split_ll_by_function(ll_path, ir_dir)
            shutil.copyfile(ll_path, outdir / "whole_binary.ll")

            target = functions if limit is None else functions[:limit]
            for func in target:
                record = {"function": func["name"], "address": func["address"]}
                hit = ll_functions.get(func["name"])
                if hit is None:
                    record["status"] = "error"
                    record["error"] = "no LLVM IR definition found (likely external/pruned by RetDec)"
                else:
                    out_path, n_lines = hit
                    record["status"] = "ok"
                    record["output_file"] = str(out_path.relative_to(outdir))
                    record["num_ll_lines"] = n_lines
                lifted_records.append(record)
        except Exception as e:
            fatal_error = f"{type(e).__name__}: {e}"

    summary = write_summary(outdir, "retdec_llvmir", binary_path, len(functions), lifted_records, timer.duration_s, fatal_error)
    write_json(outdir / "lift_records.json", lifted_records)

    print(f"Functions: {len(functions)}  ok={summary['num_lifted_ok']}  errors={summary['num_lifted_error']}")
    print(f"Duration: {timer.duration_s:.2f}s")
    return 0 if fatal_error is None else 1


def main():
    parser = build_backend_arg_parser("List functions and lift to LLVM IR with the RetDec CLI")
    parser.add_argument("--retdec-bin", type=Path, default=None, help="Path to retdec-decompiler (default: $RETDEC_BIN or PATH)")
    parser.add_argument("--timeout", type=float, default=900, help="Subprocess timeout in seconds (default: 900)")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    return run(args.binary, args.outdir, args.limit, args.retdec_bin, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
