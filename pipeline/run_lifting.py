"""Orchestrator for phase 1 of the pipeline: list functions + lift binaries
across multiple backends (pyghidra/P-code, r2pipe/ESIL, angr/VEX,
Binary Ninja/LLIL+MLIL+HLIL, RetDec/LLVM IR).

Each (binary, backend) pair is run as its OWN subprocess, using the backend's
standalone script under pipeline/backends/. This isolates crashes and JVM/
license state between backends, and each backend script stays runnable and
debuggable on its own.

Usage:
    python -m pipeline.run_lifting --binaries-dir binaries
    python -m pipeline.run_lifting --binary binaries/ls/foo.elf --backends r2,angr
    python -m pipeline.run_lifting --binaries-dir binaries --bits 64
    python -m pipeline.run_lifting --binaries-dir binaries --opt O0,O2
    python -m pipeline.run_lifting --binaries-dir binaries --arch arm,x86
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import discover_binaries, ensure_dir, parse_binary_metadata, write_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

BACKENDS = {
    "pyghidra": "pipeline.backends.pyghidra_lift",
    "r2": "pipeline.backends.r2_lift",
    "angr": "pipeline.backends.angr_lift",
    "binja_llil": "pipeline.backends.binja_llil_lift",
    "binja_mlil": "pipeline.backends.binja_mlil_lift",
    "binja_hlil": "pipeline.backends.binja_hlil_lift",
    "retdec": "pipeline.backends.retdec_lift",
}

DEFAULT_PYTHON_CANDIDATES = [REPO_ROOT / "bin" / "python3", Path(sys.executable)]


def default_python():
    for candidate in DEFAULT_PYTHON_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_one(python_exe, backend, module, binary, outdir, limit, timeout_s, retdec_bin):
    ensure_dir(outdir)
    cmd = [python_exe, "-m", module, "--binary", str(binary), "--outdir", str(outdir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if backend == "retdec" and retdec_bin:
        cmd += ["--retdec-bin", str(retdec_bin)]

    print(f"\n=== {binary.name} :: {backend} ===")
    print(" ".join(cmd))

    t0 = time.perf_counter()
    record = {"binary": str(binary), "backend": backend, "outdir": str(outdir)}
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_s
        )
        (outdir / "orchestrator_stdout.log").write_text(proc.stdout)
        (outdir / "orchestrator_stderr.log").write_text(proc.stderr)
        record["returncode"] = proc.returncode
        record["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}) -- see {outdir}/orchestrator_stderr.log")
        else:
            print("  ok")
    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
        record["returncode"] = None
        print(f"  TIMEOUT after {timeout_s}s")
    except Exception as e:
        record["status"] = "crashed"
        record["returncode"] = None
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"  CRASHED: {record['error']}")

    record["wall_time_s"] = time.perf_counter() - t0

    summary_path = outdir / "summary.json"
    if summary_path.exists():
        try:
            import json
            record["summary"] = json.loads(summary_path.read_text())
        except Exception:
            record["summary"] = None
    else:
        record["summary"] = None

    return record


def main():
    parser = argparse.ArgumentParser(description="Run the lifting pipeline across binaries and backends")
    parser.add_argument("--binaries-dir", type=Path, default=REPO_ROOT / "binaries",
                         help="Directory to recursively scan for ELF binaries (default: ./binaries)")
    parser.add_argument("--binary", action="append", type=Path, default=None,
                         help="Explicit binary path (repeatable). Overrides --binaries-dir when given.")
    parser.add_argument("--arch", default=None,
                         help="Comma-separated architectures to include, parsed from the BinKit filename "
                              "(e.g. arm,x86). Default: all architectures")
    parser.add_argument("--bits", type=int, choices=[32, 64], default=None,
                         help="Only include binaries of this bitness, parsed from the BinKit filename "
                              "(default: both 32 and 64)")
    parser.add_argument("--opt", default=None,
                         help="Comma-separated optimization levels to include, parsed from the BinKit "
                              "filename (e.g. O0,O2). Default: all optimization levels")
    parser.add_argument("--backends", default="pyghidra,r2,angr,binja_llil,binja_mlil,binja_hlil,retdec",
                         help=f"Comma-separated backend list (available: {', '.join(BACKENDS)})")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                         help="Root output directory (default: ./results)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only lift the first N functions per binary (debugging/smoke tests)")
    parser.add_argument("--timeout", type=float, default=1800,
                         help="Per (binary, backend) subprocess timeout in seconds (default: 1800)")
    parser.add_argument("--python", default=None,
                         help="Python executable to run backend scripts with (default: ./bin/python3 if present)")
    parser.add_argument("--retdec-bin", type=Path, default=None,
                         help="Path to retdec-decompiler (default: $RETDEC_BIN or PATH)")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = set(backends) - set(BACKENDS)
    if unknown:
        parser.error(f"Unknown backend(s): {', '.join(sorted(unknown))} (available: {', '.join(BACKENDS)})")

    if args.binary:
        binaries = args.binary
    else:
        binaries = discover_binaries(args.binaries_dir)

    if args.arch is not None:
        archs = {a.strip() for a in args.arch.split(",") if a.strip()}
        binaries = [b for b in binaries if parse_binary_metadata(b)["arch"] in archs]

    if args.bits is not None:
        binaries = [b for b in binaries if parse_binary_metadata(b)["bits"] == args.bits]

    if args.opt is not None:
        opts = {o.strip() for o in args.opt.split(",") if o.strip()}
        binaries = [b for b in binaries if parse_binary_metadata(b)["opt"] in opts]

    if not binaries:
        parser.error("No ELF binaries found (check --binaries-dir / --binary / --arch / --bits / --opt)")

    python_exe = args.python or default_python()
    ensure_dir(args.results_dir)

    print(f"Python: {python_exe}")
    print(f"Binaries ({len(binaries)}):")
    for b in binaries:
        print(f"  {b}")
    print(f"Backends: {', '.join(backends)}")

    runs = []
    for binary in binaries:
        meta = parse_binary_metadata(binary)
        binary_dir = args.results_dir / binary.stem
        write_json(binary_dir / "metadata.json", meta)

        for backend in backends:
            module = BACKENDS[backend]
            outdir = binary_dir / backend
            record = run_one(python_exe, backend, module, binary, outdir, args.limit, args.timeout, args.retdec_bin)
            runs.append(record)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_binaries": len(binaries),
        "backends": backends,
        "runs": runs,
    }
    write_json(args.results_dir / "manifest.json", manifest)

    print("\n--- Summary ---")
    ok = sum(1 for r in runs if r["status"] == "ok")
    print(f"{ok}/{len(runs)} runs completed successfully")
    for r in runs:
        if r["status"] != "ok":
            print(f"  {r['status'].upper():8s} {Path(r['binary']).name} :: {r['backend']}")
    print(f"Manifest: {args.results_dir / 'manifest.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
