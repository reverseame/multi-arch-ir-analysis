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
import os
import subprocess
import sys
import threading
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


def run_subprocess_with_rusage(cmd, cwd, timeout_s, stdout_path, stderr_path):
    """Run cmd as a subprocess and return (returncode, wall_time_s, max_rss_kb, timed_out).

    This allows to record the maximum resource usage of each backend execution.
    """
    with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out_f, stderr=err_f)

        reaped = {}

        def reap():
            reaped["result"] = os.wait4(proc.pid, 0)

        reaper = threading.Thread(target=reap, daemon=True)
        reaper.start()
        reaper.join(timeout_s)

        timed_out = reaper.is_alive()
        if timed_out:
            proc.kill()
            reaper.join()

        wall_time_s = time.perf_counter() - t0
        _, status, rusage = reaped["result"]

    returncode = os.waitstatus_to_exitcode(status)
    return returncode, wall_time_s, rusage.ru_maxrss, timed_out


def run_one(python_exe, backend, module, binary, outdir, limit, timeout_s, retdec_bin):
    ensure_dir(outdir)
    cmd = [python_exe, "-m", module, "--binary", str(binary), "--outdir", str(outdir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if backend == "retdec" and retdec_bin:
        cmd += ["--retdec-bin", str(retdec_bin)]

    stdout_path = outdir / "orchestrator_stdout.log"
    stderr_path = outdir / "orchestrator_stderr.log"
    record = {"binary": str(binary), "backend": backend, "outdir": str(outdir)}
    try:
        returncode, wall_time_s, max_rss_kb, timed_out = run_subprocess_with_rusage(
            cmd, str(REPO_ROOT), timeout_s, stdout_path, stderr_path
        )
        record["returncode"] = returncode
        record["wall_time_s"] = wall_time_s
        record["max_rss_kb"] = max_rss_kb
        if timed_out:
            record["status"] = "timeout"
        else:
            record["status"] = "ok" if returncode == 0 else "failed"
            if record["status"] == "ok":
                stdout_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
    except Exception as e:
        record["status"] = "crashed"
        record["returncode"] = None
        record["wall_time_s"] = None
        record["max_rss_kb"] = None
        record["error"] = f"{type(e).__name__}: {e}"

    if record["status"] == "ok":
        print(f"  {backend:<12} OK")
    elif record["status"] == "failed":
        print(f"  {backend:<12} ERROR -- see {outdir}/orchestrator_stderr.log")
    else:
        print(f"  {backend:<12} ERROR ({record['status']})")

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

        print(f"\n{binary.name}")
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
