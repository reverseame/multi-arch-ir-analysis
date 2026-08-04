"""Robustness metrics: how much lifting fails and how much semantic
information leaks through each backend's escape valve. Not in
possible_metrics.txt yet -- a new category, same status temporaries.py had
when it was added (see pipeline/metrics/blocks/temporaries.py's docstring).
Registers two blocks, "error_rate_by_category" and "escape_fraction", kept
in one file since both are robustness-category metrics, but registered
separately (rather than merged into one block like expansion_ratio.py's 8
metrics) because they operate at different row granularities -- one row per
(binary, backend) run for error rates, one row per (binary, backend,
function) for escape fraction -- and registry.py's CSV export
(run_metrics.write_block_output) needs every row in a block's "rows" list to
share the same columns.

error_rate_by_category
-----------------------
Percentage of native instructions, broken down by arithmetic/control/
memory/other (pipeline/native_classify.py's taxonomy), that belong to a
function the backend failed to lift.

Every backend in this pipeline only fails at *whole-function* granularity:
Ghidra/Hex-Rays decompile a function atomically (one hexrays_failure_t / one
decompileFunction call covers the whole thing), and angr/Binary Ninja/r2/
retdec's per-function try/except discards whatever partial IR was produced
before the exception. There is no per-instruction failure signal anywhere
in this pipeline to key off of -- so a failed function's error is attributed
across every native-instruction category proportional to how many
instructions of that category the function actually contains. This is
computed straight from the ORIGINAL ELF (pipeline/elf_bytes.py's
ElfByteReader + pipeline/native_classify.py's classify_bytes), not from any
backend's own native-count bookkeeping (record["native_{category}"]) --
that bookkeeping is only ever populated for status=="ok" records, since
every backend computes it inside the same try block as the IR lift itself
(see e.g. pyghidra_lift.py's lift_function). Reading directly from the ELF
works identically regardless of whether the function lifted, using each
function's (address, size) from functions.json, which list_functions()
always writes before any per-function lift is attempted.

The numerator/denominator pair is summed across every attempted function
(status "ok" or "error", never "skipped" -- external functions were never
really attempted) within one (binary, backend) run, not computed as one
ratio per function -- a single function's rate would trivially be 0% or
100%, throwing away exactly the "failures concentrate in category X"
signal this metric exists to surface. That signal only appears once you
look at the instruction-category mix across many functions together.

escape_fraction
----------------
Percentage of a backend's own IR that is its escape valve -- the generic
fallback construct used when the lifter can't produce a proper semantic
translation and instead emits an opaque marker construct:
  - pyghidra (P-code): CALLOTHER pcode ops.
  - angr (VEX): Ist_Dirty statements (opaque "dirty helper" C-function calls).
  - binja_llil/mlil/hlil (BNIL): INTRINSIC/INTRINSIC_SSA operations.
  - ida (Hex-Rays microcode): mop_h ("helper function") operands.
  - r2 (ESIL): no operator-level escape valve exists (see
    pipeline/backends/r2_lift.py's temporaries-metric note on ESIL being a
    flat RPN stack language) -- reuses r2_lift.py's own existing
    num_uncovered field instead (instructions with no ESIL translation at
    all), over num_native_instructions rather than an IR-op count, since
    that's the granularity ESIL's own "no real semantic translation" signal
    is already tracked at.
  - retdec (LLVM IR): NOT included. LLVM IR has no single confirmed
    construct RetDec reliably uses as an escape valve.

Each per-function fraction is num_escape_ops / <the backend's own IR-size
field> (num_pcode_ops, num_statements, num_{llil,mlil,hlil}_instructions,
num_microcode_ops -- the same IR_SIZE_FIELDS mapping expansion_ratio.py
uses), counted only over successfully-lifted functions (an escape-valve
count only exists if the IR itself exists).
"""

import json
from collections import defaultdict
from pathlib import Path

from pipeline.elf_bytes import ElfByteReader
from pipeline.metrics.blocks.expansion_ratio import IR_SIZE_FIELDS, _load_lift_records
from pipeline.metrics.formulas import aggregate_stats, aggregate_stats_with_iqr
from pipeline.metrics.registry import register_block
from pipeline.native_classify import arch_family, classify_bytes, make_disassembler

REPO_ROOT = Path(__file__).resolve().parents[3]

CATEGORIES = ("arithmetic", "control", "memory", "other")

ERROR_RATE_METRICS = {
    f"error_rate_{cat}": {"direction": "lower_is_better", "unit": "fraction of native instructions"}
    for cat in CATEGORIES
}
ESCAPE_METRICS = {
    "escape_fraction": {"direction": "descriptive", "unit": "escape ops / ir ops (native instrs for r2)"},
}


def _empty_counts():
    return {c: 0 for c in CATEGORIES}


def _load_functions(outdir):
    """address (hex string, the same "address" convention lift_records.json
    entries use -- see expansion_ratio.py's module docstring re: how every
    backend derives both from the same underlying function-object
    expression) -> size, from functions.json. lift_records.json entries
    don't carry a function's size themselves, only functions.json does.
    """
    path = Path(outdir) / "functions.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {f["address"]: f.get("size") for f in data.get("functions", [])}


def _resolve_binary_path(run):
    path = Path(run["binary"])
    if path.exists():
        return path
    candidate = REPO_ROOT / run["binary"]
    return candidate if candidate.exists() else None


def _run_native_category_counts(run):
    """(ok_counts, error_counts): {"arithmetic": n, ...} each, summed across
    every attempted (status "ok" or "error") function in one (binary,
    backend) run, split by whether that function's lift succeeded -- see
    module docstring for why this reads the ELF directly instead of reusing
    any backend's own native-count bookkeeping.
    """
    ok_counts, error_counts = _empty_counts(), _empty_counts()
    meta = run.get("binary_meta") or {}
    arch, bits = meta.get("arch"), meta.get("bits")
    if not arch or not bits:
        return ok_counts, error_counts

    md = make_disassembler(arch, bits)
    if md is None:
        return ok_counts, error_counts
    family = arch_family(arch, bits)

    functions_by_addr = _load_functions(run["outdir"])
    records = _load_lift_records(run["outdir"])
    if not records or not functions_by_addr:
        return ok_counts, error_counts

    binary_path = _resolve_binary_path(run)
    if binary_path is None:
        return ok_counts, error_counts

    with ElfByteReader(binary_path) as reader:
        for record in records:
            status = record.get("status")
            if status not in ("ok", "error"):
                continue
            size = functions_by_addr.get(record.get("address"))
            if not size:
                continue
            va = int(record["address"], 16)
            data = reader.read(va, size)
            if not data:
                continue
            target = ok_counts if status == "ok" else error_counts
            for _addr, _size, _mnem, category in classify_bytes(md, data, va, family):
                target[category] += 1

    return ok_counts, error_counts


@register_block("error_rate_by_category")
def compute_error_rate(runs, results_dir):
    attempted = [r for r in runs if r.get("status") in ("ok", "error")]

    overall = {cat: [] for cat in CATEGORIES}
    by_backend = defaultdict(lambda: {cat: [] for cat in CATEGORIES})
    by_backend_arch = defaultdict(lambda: defaultdict(lambda: {cat: [] for cat in CATEGORIES}))
    # Errors concentrate in specific runs rather than spreading continuously
    # across a group -- median/IQR on the per-category rates alone would
    # hide that, so track how many of a (backend, arch, bits) group's
    # attempted runs had *any* function-level error separately.
    run_counts = defaultdict(lambda: defaultdict(lambda: {"n_runs": 0, "n_errors": 0}))
    rows = []

    for run in attempted:
        backend = run["backend"]
        meta = run.get("binary_meta") or {}
        arch, bits = meta.get("arch"), meta.get("bits")
        arch_bits = f"{arch}_{bits}" if arch is not None and bits is not None else None
        ok_counts, error_counts = _run_native_category_counts(run)

        if arch_bits is not None:
            run_counts[backend][arch_bits]["n_runs"] += 1
            if sum(error_counts.values()) > 0:
                run_counts[backend][arch_bits]["n_errors"] += 1

        row = {
            "binary": Path(run["binary"]).name,
            "backend": backend,
            "arch": arch,
            "bits": bits,
            "opt": meta.get("opt"),
            "compiler": meta.get("compiler"),
        }
        any_rate = False
        for cat in CATEGORIES:
            total = ok_counts[cat] + error_counts[cat]
            row[f"native_ok_{cat}"] = ok_counts[cat]
            row[f"native_error_{cat}"] = error_counts[cat]
            if not total:
                row[f"error_rate_{cat}"] = None
                continue
            rate = error_counts[cat] / total
            row[f"error_rate_{cat}"] = rate
            any_rate = True
            overall[cat].append(rate)
            by_backend[backend][cat].append(rate)
            if arch_bits is not None:
                by_backend_arch[backend][arch_bits][cat].append(rate)

        if any_rate:
            rows.append(row)

    return {
        "block": "error_rate_by_category",
        "num_runs_total": len(runs),
        "num_runs_attempted": len(attempted),
        "num_runs_evaluated": len(rows),
        "metrics": {
            f"error_rate_{cat}": {**ERROR_RATE_METRICS[f"error_rate_{cat}"], **(aggregate_stats(overall[cat]) or {"n": 0})}
            for cat in CATEGORIES
        },
        "by_backend": {
            backend: {
                **{f"error_rate_{cat}": aggregate_stats(vals[cat]) for cat in CATEGORIES},
                "by_arch": {
                    arch_bits: {
                        **{
                            f"error_rate_{cat}": aggregate_stats_with_iqr(by_backend_arch[backend][arch_bits][cat])
                            for cat in CATEGORIES
                        },
                        "n_errors": counts["n_errors"],
                        "n_runs": counts["n_runs"],
                    }
                    for arch_bits, counts in run_counts.get(backend, {}).items()
                },
            }
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }


def _escape_fraction_for_record(backend, record):
    if backend == "r2":
        denom = record.get("num_native_instructions")
        numer = record.get("num_uncovered")
    else:
        ir_field = IR_SIZE_FIELDS.get(backend)
        if ir_field is None:
            return None
        denom = record.get(ir_field)
        numer = record.get("num_escape_ops")
    if not denom or numer is None:
        return None
    return numer / denom


@register_block("escape_fraction")
def compute_escape_fraction(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok" and r["backend"] != "retdec"]

    overall = []
    by_backend = defaultdict(list)
    by_backend_arch = defaultdict(lambda: defaultdict(list))
    rows = []

    for run in completed:
        backend = run["backend"]
        meta = run.get("binary_meta") or {}
        arch, bits = meta.get("arch"), meta.get("bits")
        arch_bits = f"{arch}_{bits}" if arch is not None and bits is not None else None

        for record in _load_lift_records(run["outdir"]):
            if record.get("status") != "ok":
                continue
            fraction = _escape_fraction_for_record(backend, record)
            if fraction is None:
                continue

            rows.append({
                "binary": Path(run["binary"]).name,
                "backend": backend,
                "arch": arch,
                "bits": bits,
                "opt": meta.get("opt"),
                "compiler": meta.get("compiler"),
                "function": record.get("function"),
                "num_escape_ops": record.get("num_uncovered") if backend == "r2" else record.get("num_escape_ops"),
                "escape_fraction": fraction,
            })
            overall.append(fraction)
            by_backend[backend].append(fraction)
            if arch_bits is not None:
                by_backend_arch[backend][arch_bits].append(fraction)

    return {
        "block": "escape_fraction",
        "num_runs_ok": len(completed),
        "num_runs_total": len(runs),
        "num_functions_evaluated": len(rows),
        "metrics": {
            "escape_fraction": {**ESCAPE_METRICS["escape_fraction"], **(aggregate_stats(overall) or {"n": 0})},
        },
        "by_backend": {
            backend: {
                "escape_fraction": aggregate_stats(vals),
                "by_arch": {
                    arch_bits: {"escape_fraction": aggregate_stats_with_iqr(arch_vals)}
                    for arch_bits, arch_vals in by_backend_arch.get(backend, {}).items()
                },
            }
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
