"""Verbosity metrics block: IR-size expansion ratio, per (binary, backend, function).

expansion_ratio_ops: (IR op count) / (native instruction count), from
possible_metrics.txt's "IR-size expansion ratio" -- the operations variant
(IR ops, not lines). No cross-backend function matching is needed: the
comparison is IR size vs. native size *within* the same backend's own run,
unlike the cross-architecture invariance metrics in binja_invariance.py.

The native instruction count comes from two different places depending on
the backend:
  - angr/binja/pyghidra record it themselves, in lift_records.json (see
    NATIVE_FIELD), because computing it needs their own already-open
    session (loaded VEX blocks, an open Binary Ninja database, an open
    Ghidra program) -- getting it later here would mean reloading the whole
    binary in that tool, far more expensive than the near-free count taken
    while the session is already open for lifting.
  - retdec's is parsed here instead, straight from the .dsm disassembly
    listing retdec_lift.py already leaves on disk under the run's outdir
    (see _retdec_native_instruction_counts). Unlike the tools above, RetDec
    has no live session to reuse -- the .dsm is a static text artifact --
    so parsing it can happen at metrics time (not timed as lifting cost)
    instead of adding a second pass inside retdec_lift.py's Timer()-wrapped
    run.

RetDec's IR_SIZE_FIELDS entry (num_ll_lines) is LLVM-IR line count, not an op
count -- possible_metrics.txt allows either for this metric ("IR ops or
lines"), so it's included, just not directly comparable in magnitude to the
op-counting backends.

r2/ESIL's numerator (num_esil_ops) counts only real ESIL *operator* tokens
per instruction (see r2_lift.py's esil_op_count), against radare2's own
`ae???` operator table -- not every comma-separated token, since roughly
half of them are operand values (registers, immediates) pushed onto ESIL's
RPN stack rather than operations. This is the closest analog to a
sub-instruction op count that a flat, stack-based IR like ESIL has, but
it's still a rougher proxy than the other backends' real per-op counts (VEX
statements, P-code ops, IL instructions), so treat r2's ratio as
lower-confidence/directional rather than directly comparable to the rest.

Only successful runs ("status" == "ok") are counted, and within a run, only
per-function lift records with "status" == "ok" and a non-zero native
instruction count (stub/empty functions can't produce a meaningful ratio).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from pipeline.metrics.formulas import aggregate_stats, expansion_ratio
from pipeline.metrics.registry import register_block

IR_SIZE_FIELDS = {
    "binja_llil": "num_llil_instructions",
    "binja_mlil": "num_mlil_instructions",
    "binja_hlil": "num_hlil_instructions",
    "angr": "num_statements",
    "pyghidra": "num_pcode_ops",
    "retdec": "num_ll_lines",
    "r2": "num_esil_ops",
}

NATIVE_FIELD = "num_native_instructions"

DSM_FUNCTION_RE = re.compile(r"^; function: (?P<name>\S+) at (?P<start>0x[0-9a-fA-F]+) -- (?P<end>0x[0-9a-fA-F]+)")
DSM_INSTRUCTION_ADDR_RE = re.compile(r"^0x([0-9a-fA-F]+):")

METRICS = {
    "expansion_ratio_ops": {"direction": "descriptive", "unit": "ir_ops_or_lines / native_instr"},
}


def _load_lift_records(outdir):
    path = Path(outdir) / "lift_records.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _retdec_native_instruction_counts(outdir, binary_path):
    """name -> native instruction count, by counting "0xADDR: <bytes> <mnem>"
    lines whose address falls inside the current function's own [start, end)
    range from its "; function: NAME at START -- END" header (same regex
    retdec_lift.py's own list_functions_from_dsm uses).

    Deliberately NOT "count instructions up to the next function header":
    the .dsm ends with a "Data Segment" section that reuses the exact same
    "0xADDR: <bytes> ..." line format for raw data as for instructions, with
    no function header of its own -- so the last real function before it
    (e.g. _fini) would otherwise silently absorb thousands of data-dump
    lines as if they were its own instructions. Address-range membership is
    immune to that, and to any other non-function content between two
    function bodies, since it doesn't depend on what comes next in the file.
    """
    dsm_path = Path(outdir) / f"{Path(binary_path).stem}.dsm"
    if not dsm_path.exists():
        return {}
    counts = {}
    current_name = None
    current_start = current_end = None
    with open(dsm_path, "r", errors="replace") as f:
        for line in f:
            match = DSM_FUNCTION_RE.match(line)
            if match:
                current_name = match["name"]
                current_start = int(match["start"], 16)
                current_end = int(match["end"], 16)
                counts[current_name] = 0
                continue
            if current_name is None:
                continue
            instr_match = DSM_INSTRUCTION_ADDR_RE.match(line)
            if not instr_match:
                continue
            addr = int(instr_match.group(1), 16)
            if not (current_start <= addr < current_end):
                # Past this function's declared byte range (data segment,
                # gap, or anything else not covered by a header) -- stop
                # attributing lines to it until the next real header.
                current_name = None
                continue
            counts[current_name] += 1
    return counts


@register_block("verbosity")
def compute(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok" and r["backend"] in IR_SIZE_FIELDS]

    overall = []
    by_backend = defaultdict(list)
    rows = []

    for run in completed:
        backend = run["backend"]
        ir_field = IR_SIZE_FIELDS[backend]
        meta = run.get("binary_meta") or {}
        retdec_native_counts = (
            _retdec_native_instruction_counts(run["outdir"], run["binary"])
            if backend == "retdec" else None
        )

        for record in _load_lift_records(run["outdir"]):
            if record.get("status") != "ok":
                continue
            native_count = (
                retdec_native_counts.get(record.get("function"))
                if retdec_native_counts is not None
                else record.get(NATIVE_FIELD)
            )
            ratio = expansion_ratio(record.get(ir_field), native_count)
            if ratio is None:
                continue
            rows.append({
                "binary": Path(run["binary"]).name,
                "backend": backend,
                "arch": meta.get("arch"),
                "bits": meta.get("bits"),
                "opt": meta.get("opt"),
                "compiler": meta.get("compiler"),
                "function": record.get("function"),
                "ir_size": record.get(ir_field),
                "num_native_instructions": native_count,
                "expansion_ratio_ops": ratio,
            })
            overall.append(ratio)
            by_backend[backend].append(ratio)

    return {
        "block": "verbosity",
        "num_runs_ok": len(completed),
        "num_runs_total": len(runs),
        "num_functions_evaluated": len(rows),
        "metrics": {
            "expansion_ratio_ops": {**METRICS["expansion_ratio_ops"], **(aggregate_stats(overall) or {"n": 0})},
        },
        "by_backend": {
            backend: {"expansion_ratio_ops": aggregate_stats(vals)}
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
