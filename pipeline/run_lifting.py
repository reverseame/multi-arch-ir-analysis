"""Orchestrator for phase 1 of the pipeline: list functions + lift binaries
across multiple backends (pyghidra/P-code, r2pipe/ESIL, angr/VEX,
Binary Ninja/LLIL+MLIL+HLIL, RetDec/LLVM IR, IDA Pro/Hex-Rays microcode).

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
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "ida": "pipeline.backends.ida_lift",
}

# Rounded up from results/metrics/cost.json's by_backend max_rss_kb (2026-07-29
# full-corpus baseline, 320 runs), used to schedule concurrent subprocesses
# without exceeding --max-mem-mb.
BACKEND_MEM_ESTIMATE_MB = {
    "pyghidra": 1600,
    "r2": 300,
    "angr": 900,
    "binja_llil": 2000,
    "binja_mlil": 2100,
    "binja_hlil": 2000,
    "retdec": 4000,
    "ida": 800,
}
DEFAULT_BACKEND_MEM_ESTIMATE_MB = 1000

# Binary Ninja's license on this machine is confirmed single-seat (count: 1
# in ~/.binaryninja/license.dat), so the 3 IL variants are serialized to 1
# concurrent instance regardless of --max-parallel/--max-mem-mb. ida was
# under the same conservative assumption, but running 2 idalib processes
# concurrently (2026-07-31 smoke test, grep_x86 + ls_arm) completed cleanly
# with no license/lock errors, just ordinary CPU contention -- so ida is not
# gated here.
BACKEND_LICENSE_GROUP = {
    "binja_llil": "binja",
    "binja_mlil": "binja",
    "binja_hlil": "binja",
}

DEFAULT_PYTHON_CANDIDATES = [REPO_ROOT / "bin" / "python3", Path(sys.executable)]


def available_memory_mb():
    """Best-effort read of currently available memory (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return None


class MemoryBudget:
    """Weighted semaphore: gates concurrent jobs by estimated RSS, not just count.

    A job whose own estimate exceeds the whole budget is still allowed to run
    (just never alongside anything else), so a single oversized job can't deadlock
    the scheduler.
    """

    def __init__(self, total_mb):
        self.total_mb = total_mb
        self.used_mb = 0.0
        self._cv = threading.Condition()

    def acquire(self, amount_mb):
        with self._cv:
            while self.used_mb > 0 and self.used_mb + amount_mb > self.total_mb:
                self._cv.wait()
            self.used_mb += amount_mb

    def release(self, amount_mb):
        with self._cv:
            self.used_mb -= amount_mb
            self._cv.notify_all()


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

    tag = f"[{Path(binary).name}] {backend:<12}"
    if record["status"] == "ok":
        print(f"  {tag} OK")
    elif record["status"] == "failed":
        print(f"  {tag} ERROR -- see {outdir}/orchestrator_stderr.log")
    else:
        print(f"  {tag} ERROR ({record['status']})")

    summary_path = outdir / "summary.json"
    if summary_path.exists():
        try:
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
    parser.add_argument("--compiler", default=None,
                         help="Comma-separated compiler+version to include, parsed from the BinKit "
                              "filename as '<compiler>-<compiler_version>' (e.g. gcc-6.4.0,clang-7.0). "
                              "Default: all compilers")
    parser.add_argument("--backends", default="pyghidra,r2,angr,binja_llil,binja_mlil,binja_hlil,retdec,ida",
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
    parser.add_argument("--max-parallel", type=int, default=os.cpu_count() or 4,
                         help="Max concurrent (binary, backend) subprocesses (default: CPU count). "
                              "Use 1 to restore the old fully-sequential behavior.")
    parser.add_argument("--max-mem-mb", type=float, default=None,
                         help="Memory budget (MB) for scheduling concurrent jobs; a job waits if "
                              "starting it would exceed this. Default: --mem-safety-frac of currently "
                              "available memory (/proc/meminfo), or 4000 if that can't be read.")
    parser.add_argument("--mem-safety-frac", type=float, default=0.85,
                         help="Fraction of available memory to use as the budget when --max-mem-mb "
                              "isn't given (default: 0.85)")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip (binary, backend) pairs that already have a summary.json with no "
                              "fatal_error, instead of re-running and overwriting them. For resuming a "
                              "large corpus across multiple sessions -- errors/timeouts are NOT considered "
                              "done and will be retried. Off by default: a bare rerun still overwrites "
                              "everything it's given, per this project's usual explicit-rerun convention.")
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

    if args.compiler is not None:
        compilers = {c.strip() for c in args.compiler.split(",") if c.strip()}
        binaries = [
            b for b in binaries
            if "{compiler}-{compiler_version}".format(**parse_binary_metadata(b)) in compilers
        ]

    if not binaries:
        parser.error("No ELF binaries found (check --binaries-dir / --binary / --arch / --bits / --opt / --compiler)")

    python_exe = args.python or default_python()
    ensure_dir(args.results_dir)

    if args.max_mem_mb is not None:
        mem_budget_mb = args.max_mem_mb
    else:
        avail = available_memory_mb()
        mem_budget_mb = avail * args.mem_safety_frac if avail else 4000.0

    print(f"Python: {python_exe}")
    print(f"Binaries ({len(binaries)}):")
    for b in binaries:
        print(f"  {b}")
    print(f"Backends: {', '.join(backends)}")
    print(f"Scheduling: max_parallel={args.max_parallel} + 1 dedicated binja worker, "
          f"mem_budget={mem_budget_mb:.0f}MB (binja capped to 1 concurrent instance)")

    binary_dirs = {}
    for binary in binaries:
        meta = parse_binary_metadata(binary)
        binary_dir = args.results_dir / binary.stem
        write_json(binary_dir / "metadata.json", meta)
        binary_dirs[binary] = binary_dir

    jobs = [(binary, backend) for binary in binaries for backend in backends]

    skipped_records = {}
    if args.skip_existing:
        jobs_to_run = []
        for binary, backend in jobs:
            outdir = binary_dirs[binary] / backend
            summary_path = outdir / "summary.json"
            done = False
            existing_summary = None
            if summary_path.exists():
                try:
                    existing_summary = json.loads(summary_path.read_text())
                    done = existing_summary.get("fatal_error") is None
                except Exception:
                    done = False
            if done:
                skipped_records[(binary, backend)] = {
                    "binary": str(binary), "backend": backend, "outdir": str(outdir),
                    "status": "ok", "skipped_existing": True,
                    "returncode": None, "wall_time_s": None, "max_rss_kb": None,
                    "summary": existing_summary,
                }
            else:
                jobs_to_run.append((binary, backend))
        print(f"Skip-existing: {len(skipped_records)}/{len(jobs)} already done, "
              f"{len(jobs_to_run)} to run")
    else:
        jobs_to_run = jobs

    mem_budget = MemoryBudget(mem_budget_mb)

    def run_job(binary, backend):
        module = BACKENDS[backend]
        outdir = binary_dirs[binary] / backend
        mem_mb = BACKEND_MEM_ESTIMATE_MB.get(backend, DEFAULT_BACKEND_MEM_ESTIMATE_MB)

        mem_budget.acquire(mem_mb)
        try:
            return run_one(python_exe, backend, module, binary, outdir,
                            args.limit, args.timeout, args.retdec_bin)
        finally:
            mem_budget.release(mem_mb)

    # binja jobs go to their own dedicated 1-worker pool instead of a lock
    # inside the shared pool. A lock would have a worker thread dequeue a
    # blocked binja_mlil/hlil job and then just sit there blocked for the
    # whole wait, wasting one of --max-parallel's worker slots the entire
    # time another binja job is running (confirmed via the 2026-07-31 14-
    # binary batch: jobs ran almost fully serial per-binary, only 1.5x
    # speedup, because 2-3 of the 4 workers were perpetually stuck blocked
    # on this exact lock). A separate single-worker pool serializes binja
    # jobs among themselves for free (only one worker to hand them to) while
    # never touching the main pool's workers, so pyghidra/r2/angr/retdec/ida
    # actually run concurrently with whatever binja is doing.
    binja_jobs = [(b, be) for b, be in jobs_to_run if be in BACKEND_LICENSE_GROUP]
    other_jobs = [(b, be) for b, be in jobs_to_run if be not in BACKEND_LICENSE_GROUP]

    results_by_job = dict(skipped_records)
    with ThreadPoolExecutor(max_workers=args.max_parallel) as main_pool, \
            ThreadPoolExecutor(max_workers=1) as binja_pool:
        futures = {main_pool.submit(run_job, binary, backend): (binary, backend) for binary, backend in other_jobs}
        futures.update({binja_pool.submit(run_job, binary, backend): (binary, backend) for binary, backend in binja_jobs})
        for future in as_completed(futures):
            job = futures[future]
            results_by_job[job] = future.result()

    runs = [results_by_job[job] for job in jobs]

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
