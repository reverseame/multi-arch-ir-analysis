"""Verbosity metrics block: IR-size expansion ratio, per (binary, backend,
function), from possible_metrics.txt's "IR-size expansion ratio". Reports
two independent axes side by side:

  - granularity: "ops" (one count per top-level instruction/statement,
    e.g. num_llil_instructions, num_statements, num_pcode_ops, ...) vs.
    "ast" (one count per AST/expression-tree node, descending into every
    sub-expression -- see below for which backends this actually differs
    for).
  - category split: a plain total (expansion_ratio_ops / expansion_ratio_ast)
    vs. broken down by instruction type (expansion_ratio_ops_{arithmetic,
    control,memory} / expansion_ratio_ast_{arithmetic,control,memory}).

That's 2x2 = 8 metrics total, all descriptive, all (IR count) / (native
instruction count) within one function -- no cross-backend function matching
is needed for any of them: the comparison is IR size vs. native size
*within* the same backend's own run, unlike the cross-architecture
invariance metrics in binja_invariance.py.

Both sides of the ratio are classified by the same rule for a given category
(see pipeline/native_classify.py for the native side; each backend module's
own IR-op classifier for the numerator -- angr_lift.classify_vex_statement_
{ops,ast}, binja_il_classify.classify_il_function_{ops,ast}, pyghidra_lift.
classify_pcode_op, r2_lift.classify_esil_expression, ida_lift's mcode
category table (walked flat for ops via lift_function, recursively for ast
via _walk_minsn_ast), and this module's classify_ll_line for retdec's LLVM
IR). "other" (data movement, casts, SSA bookkeeping, NOPs) is computed too
and counted into the *_ops/*_ast totals, but not reported as its own ratio
-- see each classifier's own docstring for why it's excluded from the
category breakdown (mainly: type conversions/reinterprets are deliberately
not counted as "arithmetic" everywhere, consistently).

The ops/ast distinction only actually differs for VEX (angr), Binary
Ninja's LLIL/MLIL/HLIL, and IDA's Hex-Rays microcode -- the backends whose
IR can nest sub-expressions inside a single instruction/statement
(angr_lift.classify_vex_statement_ast walks every statement's full
expression tree via pyvex's already-recursive IRExpr.child_expressions;
binja_il_classify.classify_il_function_ast walks every instruction and
sub-expression via il_func.traverse(); ida_lift._walk_minsn_ast walks every
embedded sub-instruction reachable through a minsn_t's mop_d/mop_f/mop_a/
mop_p operands) -- e.g. `t5 = Add32(Mul32(t1,t2), t3)` counts as one
"arithmetic" node at the ops granularity (the whole statement) but two at
the ast granularity (the Add and the Mul); similarly Hex-Rays often prints
`mov call $foo() => result, ret` as one microcode instruction but that's an
outer mov wrapping a nested call, two ops at the ast granularity. pyghidra
(P-code), r2 (ESIL), and retdec (LLVM IR text) are already atomic/flat at
the operation level -- their ops and ast numbers are identical by
construction (same underlying count, reported under both names) since
there's no nesting to distinguish.

Per-function category counts come from two different places depending on
the backend:
  - angr/binja/pyghidra/r2/ida record them directly in lift_records.json
    (the ir_ops_{category}/ir_ast_{category}/native_{category} fields each
    backend's lift_function now produces -- pyghidra/r2 only produce
    ir_ops_{category}, since ops==ast for them; see _record_category_counts'
    fallback), because computing the native side needs their own
    already-open session (loaded VEX blocks, an open Binary Ninja database,
    an open Ghidra program, a live r2 session, an open IDA database) --
    getting it later here would mean reloading the whole binary in that
    tool, far more expensive than the near-free count taken while the
    session is already open for lifting.
  - retdec's are computed here instead, straight from the .dsm disassembly
    listing and .ll LLVM IR module retdec_lift.py already leaves on disk
    under the run's outdir (see _retdec_native_category_counts and
    _retdec_ir_category_counts). Unlike the tools above, RetDec has no live
    session to reuse -- both files are static text artifacts -- so parsing
    them can happen at metrics time (not timed as lifting cost) instead of
    adding a second pass inside retdec_lift.py's Timer()-wrapped run.

r2/ESIL's numerator counts only real ESIL *operator* tokens per instruction
(see r2_lift.classify_esil_expression), against radare2's own `ae???`
operator table -- not every comma-separated token, since roughly half of
them are operand values (registers, immediates) pushed onto ESIL's RPN
stack rather than operations. This is the closest analog to a
sub-instruction op count that a flat, stack-based IR like ESIL has, but
it's still a rougher proxy than the other backends' real per-op counts (VEX
statements, P-code ops, IL instructions), so treat r2's ratios as
lower-confidence/directional rather than directly comparable to the rest --
particularly its control ratio, which runs structurally higher on MIPS
(delay-slot branches wrap every jump in fixed "?{"/"}"/BREAK/SETJT/SETD
boilerplate that x86/ARM don't have) -- a real property of ESIL's own
representation, not a classification bug.

Only successful runs ("status" == "ok") are counted, and within a run, only
per-function lift records with "status" == "ok" and a non-zero native
instruction count for that category (stub/empty functions, or functions
with no instructions of a given category, can't produce a meaningful ratio
for it).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from pipeline.backends.retdec_lift import DSM_FUNCTION_RE, LL_DEFINE_RE
from pipeline.metrics.formulas import aggregate_stats, expansion_ratio
from pipeline.metrics.registry import register_block
from pipeline.native_classify import arch_family, classify_bytes, make_disassembler

CATEGORIES = ("arithmetic", "control", "memory", "other")
RATIO_CATEGORIES = ("arithmetic", "control", "memory")
GRANULARITIES = ("ops", "ast")

# Backends whose lift_records.json already carries per-category
# ir_ops_{category}/native_{category} fields (see each backend module).
# retdec is deliberately excluded -- classified here instead (see module
# docstring) -- and "ir_size" still comes from its own field per backend,
# purely for the informational (non-ratio) column in rows/CSV.
IR_SIZE_FIELDS = {
    "binja_llil": "num_llil_instructions",
    "binja_mlil": "num_mlil_instructions",
    "binja_hlil": "num_hlil_instructions",
    "angr": "num_statements",
    "pyghidra": "num_pcode_ops",
    "retdec": "num_ll_lines",
    "r2": "num_esil_ops",
    "ida": "num_microcode_ops",
}

DSM_INSTRUCTION_RE = re.compile(r"^0x(?P<addr>[0-9a-fA-F]+):\s+(?P<bytes>[0-9a-fA-F ]+?)\s*\t")


def _ratio_keys():
    keys = []
    for gran in GRANULARITIES:
        keys.append(gran)
        keys.extend(f"{gran}_{cat}" for cat in RATIO_CATEGORIES)
    return tuple(keys)


RATIO_KEYS = _ratio_keys()

METRICS = {}
for _gran in GRANULARITIES:
    METRICS[f"expansion_ratio_{_gran}"] = {
        "direction": "descriptive", "unit": "ir_ops_or_lines / native_instr"
    }
    for _cat in RATIO_CATEGORIES:
        METRICS[f"expansion_ratio_{_gran}_{_cat}"] = {
            "direction": "descriptive", "unit": "ir_ops / native_instr"
        }

# LLVM IR opcode -> category, keyed off the same three buckets every other
# backend's IR classifier uses (see module docstring). Type-conversion/cast
# opcodes (trunc, zext, sext, bitcast, ...) and SSA bookkeeping (phi) are
# deliberately left uncategorized ("other"), the same call made for VEX's
# Iop_*to* ops and P-code's INT_ZEXT/INT_SEXT/CAST -- consistent across
# every backend rather than just this one. getelementptr (pointer/index
# address arithmetic, never itself a memory access) is "arithmetic",
# matching this project's native-instruction classifier treating x86 `lea`
# the same way.
_LL_CONTROL = {"br", "switch", "ret", "call", "invoke", "callbr", "indirectbr", "unreachable", "resume"}
_LL_MEMORY = {"load", "store", "alloca", "cmpxchg", "atomicrmw", "fence"}
_LL_ARITHMETIC = {
    "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv", "fdiv",
    "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or", "xor",
    "icmp", "fcmp", "getelementptr",
}
_LL_CALL_MARKERS = {"tail", "musttail", "notail"}


def classify_ll_line(line):
    """One line of RetDec's LLVM-IR text output -> "arithmetic"/"control"/
    "memory"/"other". Handles both instruction forms LLVM IR text uses:
    "%N = OPCODE ..." (value-producing) and a bare "OPCODE ..." (void, e.g.
    store/br/ret) -- labels, comments, blank lines, and directives
    (uselistorder, metadata) don't start with any recognized opcode token
    and fall to "other" with no special-casing needed.
    """
    stripped = line.strip()
    if not stripped:
        return "other"
    rhs = stripped.split("=", 1)[1].strip() if "=" in stripped else stripped
    tokens = rhs.split()
    if not tokens:
        return "other"
    opcode = tokens[0]
    if opcode in _LL_CALL_MARKERS and len(tokens) > 1:
        opcode = tokens[1]
    if opcode in _LL_CONTROL:
        return "control"
    if opcode in _LL_MEMORY:
        return "memory"
    if opcode in _LL_ARITHMETIC:
        return "arithmetic"
    return "other"


def _load_lift_records(outdir):
    path = Path(outdir) / "lift_records.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _empty_counts():
    return {c: 0 for c in CATEGORIES}


def _retdec_native_category_counts(outdir, binary_path, arch, bits):
    """name -> {"arithmetic": n, "control": n, "memory": n, "other": n} by
    decoding the raw instruction bytes on each "0xADDR: <bytes> <mnem>" .dsm
    line whose address falls inside the current function's own [start, end)
    range from its "; function: NAME at START -- END" header (same regex
    retdec_lift.py's own list_functions_from_dsm uses), via this project's
    shared native-instruction classifier (pipeline/native_classify.py) --
    the same rule angr/binja/pyghidra/r2's native-side counts use.
    """
    counts = {}
    dsm_path = Path(outdir) / f"{Path(binary_path).stem}.dsm"
    if not dsm_path.exists() or arch is None or bits is None:
        return counts
    md = make_disassembler(arch, bits)
    if md is None:
        return counts
    family = arch_family(arch, bits)

    current_name = None
    current_start = current_end = None
    with open(dsm_path, "r", errors="replace") as f:
        for line in f:
            match = DSM_FUNCTION_RE.match(line)
            if match:
                current_name = match["name"]
                current_start = int(match["start"], 16)
                current_end = int(match["end"], 16)
                counts[current_name] = _empty_counts()
                continue
            if current_name is None:
                continue
            instr_match = DSM_INSTRUCTION_RE.match(line)
            if not instr_match:
                continue
            addr = int(instr_match["addr"], 16)
            if not (current_start <= addr < current_end):
                # Past this function's declared byte range (data segment,
                # gap, or anything else not covered by a header) -- stop
                # attributing lines to it until the next real header.
                current_name = None
                continue
            data = bytes.fromhex(instr_match["bytes"].replace(" ", ""))
            decoded = next(classify_bytes(md, data, addr, family), None)
            if decoded is not None:
                counts[current_name][decoded[3]] += 1
    return counts


def _retdec_ir_category_counts(outdir, binary_path):
    """name -> {"arithmetic": n, "control": n, "memory": n, "other": n} by
    walking whole_binary.ll's top-level `define` blocks with the same
    brace-depth tracking retdec_lift.py's own split_ll_by_function uses
    (RetDec doesn't nest functions, but a naive line-range split without
    brace tracking would break on functions containing nested `{`/`}` in
    literals), classifying every line inside each function's body with
    classify_ll_line.

    LLVM IR text is already atomic/flat at the operation level -- one
    "opcode token" per line, no sub-expression nesting -- so this single
    count serves as both the ops-level and ast-level numerator for retdec
    (see module docstring); the caller uses the same dict for both.
    """
    ll_path = Path(outdir) / "whole_binary.ll"
    if not ll_path.exists():
        return {}
    lines = ll_path.read_text(errors="replace").splitlines()
    counts = {}

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
        func_counts = _empty_counts()
        for line in lines[start:j]:
            func_counts[classify_ll_line(line)] += 1
        counts[name] = func_counts
        i = j
    return counts


def _record_category_counts(record):
    """(ops_counts, ast_counts, native_counts) for one non-retdec
    lift_records.json entry, from the ir_ops_{category}/ir_ast_{category}/
    native_{category} fields each backend's lift_function now produces.

    Only angr, the three Binary Ninja backends, and ida produce
    ir_ast_{category} (their IR can nest sub-expressions -- see module
    docstring); pyghidra and r2 only produce ir_ops_{category}, so
    ast_counts falls back to the same ops values for them (ops and ast are
    identical by construction there, no nesting to distinguish).
    """
    ops_counts = {c: record.get(f"ir_ops_{c}") for c in CATEGORIES}
    ast_counts = {c: record.get(f"ir_ast_{c}", record.get(f"ir_ops_{c}")) for c in CATEGORIES}
    native_counts = {c: record.get(f"native_{c}") for c in CATEGORIES}
    return ops_counts, ast_counts, native_counts


def _total(counts):
    """Sum of all four category counts, or None if any is missing (rather
    than silently treating a missing category as zero).
    """
    values = [counts.get(c) for c in CATEGORIES]
    if any(v is None for v in values):
        return None
    return sum(values)


@register_block("verbosity")
def compute(runs, results_dir):
    completed = [r for r in runs if r.get("status") == "ok" and r["backend"] in IR_SIZE_FIELDS]

    overall = {key: [] for key in RATIO_KEYS}
    by_backend = defaultdict(lambda: {key: [] for key in RATIO_KEYS})
    rows = []

    for run in completed:
        backend = run["backend"]
        ir_field = IR_SIZE_FIELDS[backend]
        meta = run.get("binary_meta") or {}

        retdec_ir_counts = retdec_native_counts = None
        if backend == "retdec":
            retdec_ir_counts = _retdec_ir_category_counts(run["outdir"], run["binary"])
            retdec_native_counts = _retdec_native_category_counts(
                run["outdir"], run["binary"], meta.get("arch"), meta.get("bits")
            )

        for record in _load_lift_records(run["outdir"]):
            if record.get("status") != "ok":
                continue
            function = record.get("function")

            if backend == "retdec":
                # LLVM IR text has no sub-expression nesting -- the same
                # counts serve as both the ops-level and ast-level side of
                # the ratio (see _retdec_ir_category_counts docstring).
                ops_counts = ast_counts = retdec_ir_counts.get(function, _empty_counts())
                native_counts = retdec_native_counts.get(function, _empty_counts())
                # retdec has no "num_native_instructions" record field (its
                # native side is parsed from the .dsm here, not recorded
                # during lifting) -- the per-category counts already cover
                # every native instruction, so their sum is the total.
                total_native = sum(native_counts.values())
            else:
                ops_counts, ast_counts, native_counts = _record_category_counts(record)
                # Deliberately record["num_native_instructions"], not
                # sum(native_counts.values()): for archs pipeline.native_classify
                # can't disassemble (unsupported by capstone), native_counts
                # stays all-zero while the block/function-level instruction
                # count is still real -- summing the categories would silently
                # zero out the denominator instead of reporting "no data".
                total_native = record.get("num_native_instructions")

            row = {
                "binary": Path(run["binary"]).name,
                "backend": backend,
                "arch": meta.get("arch"),
                "bits": meta.get("bits"),
                "opt": meta.get("opt"),
                "compiler": meta.get("compiler"),
                "function": function,
                "ir_size": record.get(ir_field),
            }

            any_ratio = False

            for gran, counts in (("ops", ops_counts), ("ast", ast_counts)):
                total_ratio = expansion_ratio(_total(counts), total_native)
                row[f"expansion_ratio_{gran}"] = total_ratio
                if total_ratio is not None:
                    any_ratio = True
                    overall[gran].append(total_ratio)
                    by_backend[backend][gran].append(total_ratio)

                for cat in RATIO_CATEGORIES:
                    ratio = expansion_ratio(counts.get(cat), native_counts.get(cat))
                    row[f"ir_{gran}_{cat}"] = counts.get(cat)
                    row[f"expansion_ratio_{gran}_{cat}"] = ratio
                    if ratio is not None:
                        any_ratio = True
                        key = f"{gran}_{cat}"
                        overall[key].append(ratio)
                        by_backend[backend][key].append(ratio)

            for cat in CATEGORIES:
                row[f"native_{cat}"] = native_counts.get(cat)

            if not any_ratio:
                continue
            rows.append(row)

    return {
        "block": "verbosity",
        "num_runs_ok": len(completed),
        "num_runs_total": len(runs),
        "num_functions_evaluated": len(rows),
        "metrics": {
            f"expansion_ratio_{key}": {**METRICS[f"expansion_ratio_{key}"], **(aggregate_stats(overall[key]) or {"n": 0})}
            for key in RATIO_KEYS
        },
        "by_backend": {
            backend: {f"expansion_ratio_{key}": aggregate_stats(vals[key]) for key in RATIO_KEYS}
            for backend, vals in by_backend.items()
        },
        "rows": rows,
    }
