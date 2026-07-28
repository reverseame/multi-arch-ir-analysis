"""List functions and lift them to ESIL using radare2 via r2pipe.

Standalone CLI:
    python -m pipeline.backends.r2_lift --binary <path> --outdir <dir>

python3 -m pipeline.backends.r2_lift \
    --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf --outdir results/ls_x86_32/r2
"""

import re
import sys
from collections import Counter
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
from pipeline.native_classify import arch_family, classify_bytes, make_disassembler

import r2pipe

ANALYSIS_COMMAND = "aaa"
CATEGORIES = ("arithmetic", "control", "memory", "other")

ESIL_OPERATOR_LINE_RE = re.compile(r"^: (?P<op>\S+) \(.*?\)\s*\\.*?(?:\((?P<tag>[a-z+]+)\))?\s*$")

_R2_ARCH_KEY = {"x86": "x86", "arm": "arm", "mips": "mips"}

# ESIL doesn't have a dedicated "jump" operator the way it has "?{"/"}" for
# conditionals -- an unconditional jmp/call/ret just assigns the target
# straight to the program-counter register, e.g. x86 `jmp 0x804b7f8` is
# ESIL "0x804b7f8,eip,=" and `ret` is "esp,[4],eip,=,4,esp,+=" -- confirmed
# against real x86/32 and arm/32 disassembly. r2's own `ae???` table tags
# "=" itself as uncategorized (it's used for every plain register write,
# not just pc), so on their own these would misclassify as "other" instead
# of "control". Fixed by treating "=" / ":=" as control specifically when
# the token immediately before it (ESIL's assignment target, in
# "src,dst,=" order) is this architecture's PC register.
_PC_REGISTER_BY_ARCH = {"x86": {32: "eip", 64: "rip"}, "arm": {32: "pc", 64: "pc"},
                         "mips": {32: "pc", 64: "pc"}, "mipseb": {32: "pc", 64: "pc"}}

# MIPS (delay-slot) branches don't even use a "pc,=" assignment -- r2
# represents the jump target through its own SETJT/SETD pseudo-ops instead
# (e.g. `bal` is "...,pc,4,+,ra,=,<target>,SETJT,1,SETD"), tagged
# "(unknown)" in ae??? like any other bookkeeping op.
_ESIL_FORCE_CONTROL = {"SETJT", "SETD"}


def pc_register_for(arch_key):
    if arch_key is None:
        return None
    arch, bits = arch_key
    return _PC_REGISTER_BY_ARCH.get(arch, {}).get(bits)


def r2_arch_key(r2):
    """(arch, bits) in this project's metadata.json convention (see
    pipeline/native_classify.py) for the currently open r2 session, from
    `ij`'s bin.arch/bin.bits/bin.endian, or None if unsupported.
    """
    info = (r2.cmdj("ij") or {}).get("bin", {})
    arch = _R2_ARCH_KEY.get(info.get("arch"))
    bits = info.get("bits")
    if arch is None or bits is None:
        return None
    if arch == "mips" and info.get("endian") == "big":
        arch = "mipseb"
    return (arch, bits)


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
                extra={"num_instructions": func.get("ninstrs"), "bits": func.get("bits")},
            )
        )
    return functions


def load_esil_operators(r2):
    """token -> "arithmetic"/"control"/"memory"/"other" for every ESIL
    operator this radare2 build recognizes, from its own `ae???` listing.

    radare2 tags each operator with its own category in that listing's
    trailing "(math)"/"(math+regw)"/"(control)"/"(memr)"/"(memw)"/
    "(unknown)" annotation (or no annotation at all) -- reused directly
    instead of inventing a separate token classification, except for one
    case ae??? doesn't disambiguate: a bare load/store like "[4]" or "=[4]"
    is tagged "(memr)"/"(memw)", but a combined read-modify-write like
    "+=[4]" (pop VAL and ADDR, mem[ADDR] = mem[ADDR] + VAL) gets no tag at
    all despite touching memory.
    """
    raw = r2.cmd("e scr.color=0; ae???") or ""
    classified = {}
    for line in raw.splitlines():
        match = ESIL_OPERATOR_LINE_RE.match(line)
        if not match:
            continue
        op = match["op"]
        if "[" in op:
            classified[op] = "memory" if op.startswith(("[", "=[")) else "arithmetic"
            continue
        tag = match["tag"] or ""
        if tag == "control":
            classified[op] = "control"
        elif tag.startswith("math"):
            classified[op] = "arithmetic"
        else:
            classified[op] = "other"
    return classified


def classify_esil_expression(esil, esil_operators, pc_register=None):
    """{"arithmetic": n, "control": n, "memory": n, "other": n} of real ESIL
    *operator* tokens in one instruction's ESIL expression -- e.g.
    "ebx,4,esp,-,=[4],4,esp,-=" has 8 comma-separated tokens total, but only
    3 are operators ("-", "=[4]", "-="); the rest ("ebx", "4", "esp", ...)
    are operand values being pushed onto ESIL's RPN stack, not operations.

    `pc_register`, if given, additionally reclassifies "="/":=" as control
    when writing directly to it (see _PC_REGISTER_BY_ARCH), and any
    SETJT/SETD token as control unconditionally (see _ESIL_FORCE_CONTROL) --
    both real branches r2's own per-operator tags don't mark as control.
    """
    counts = {c: 0 for c in CATEGORIES}
    if not esil:
        return counts
    tokens = esil.split(",")
    for i, tok in enumerate(tokens):
        if tok in _ESIL_FORCE_CONTROL:
            counts["control"] += 1
            continue
        category = esil_operators.get(tok)
        if category is None:
            continue
        if (
            pc_register is not None
            and tok in ("=", ":=")
            and i > 0
            and tokens[i - 1] == pc_register
        ):
            category = "control"
        counts[category] += 1
    return counts


def esil_op_histogram(esil, esil_operators):
    """{operator_token: count} (e.g. "+" -> 3, "=[4]" -> 1) of real ESIL
    *operator* tokens in one instruction's ESIL expression -- same
    operator/operand split as classify_esil_expression, but keyed by the raw
    token instead of collapsed into arithmetic/control/memory/other. Used by
    the agnosticism metrics' weighted_jaccard (see
    pipeline/metrics/blocks/agnosticism.py) to compare op-frequency
    distributions across architecture builds.
    """
    counts = Counter()
    if not esil:
        return counts
    for tok in esil.split(","):
        if tok in esil_operators or tok in _ESIL_FORCE_CONTROL:
            counts[tok] += 1
    return counts


# r2's own sentinel for "no separate default target" on a switch_op (an
# unsigned 64-bit -1) -- distinct from a real default target address.
_R2_NO_DEFAULT = 18446744073709551615


def esil_block_cfg_counts(r2, addr):
    """(num_blocks, num_edges) over the function's own basic-block-level CFG
    at `addr`, from radare2's `afbj` (function basic blocks). Verified live
    against this project's own x86/32 `ls` corpus, including a switch-heavy
    function (dbg.quotearg_buffer_restyled).

    Each block's "jump"/"fail" keys (if present) are its real successor
    edges -- "jump" is the unconditional or true-branch target, "fail" is
    the conditional-false/fallthrough target; a terminal block (e.g. ending
    in `ret`) has neither. A block ending in a switch reports neither key at
    all, using "switch_op" instead: one edge per entry in switch_op["cases"]
    (even when several cases share the same jump target -- each case is
    still its own distinct control-flow edge, same convention a real switch
    statement's own case labels get), plus one more for switch_op["def_val"]
    (the default-case target) unless it equals _R2_NO_DEFAULT, meaning this
    switch has no separate default branch to count.
    """
    blocks = r2.cmdj(f"afbj @ {addr}") or []
    num_edges = 0
    for block in blocks:
        switch_op = block.get("switch_op")
        if switch_op is not None:
            num_edges += len(switch_op.get("cases", []))
            if switch_op.get("def_val") != _R2_NO_DEFAULT:
                num_edges += 1
            continue
        if "jump" in block:
            num_edges += 1
        if "fail" in block:
            num_edges += 1
    return len(blocks), num_edges


def lift_function(r2, addr, func_name, esil_operators, md, family, pc_register):
    data = r2.cmdj(f"pdfj @ {addr}")
    if not data or "ops" not in data:
        raise RuntimeError("pdfj returned no ops")

    ops = data["ops"]
    lines = [f"; ---- function {func_name} @ {hex(addr)} ----"]
    num_uncovered = 0
    ir_counts = {c: 0 for c in CATEGORIES}
    native_counts = {c: 0 for c in CATEGORIES}
    # Agnosticism metric: op-type frequency histogram, compared across
    # architecture builds of the same binary by pipeline/metrics/blocks/
    # agnosticism.py's weighted_jaccard.
    op_histogram = Counter()
    for op in ops:
        op_addr = op.get("offset", op.get("addr", addr))
        mnem = op.get("opcode", "")
        esil = op.get("esil") or ""
        if not esil or op.get("type") in ("invalid", "unk", None):
            num_uncovered += 1
        op_counts = classify_esil_expression(esil, esil_operators, pc_register)
        for cat in CATEGORIES:
            ir_counts[cat] += op_counts[cat]
        op_histogram.update(esil_op_histogram(esil, esil_operators))
        if md is not None and op.get("bytes"):
            decoded = next(classify_bytes(md, bytes.fromhex(op["bytes"]), op_addr, family), None)
            if decoded is not None:
                native_counts[decoded[3]] += 1
        lines.append(f"; 0x{op_addr:x}  {mnem}")
        lines.append(f"  {esil or '(no esil)'}")
        lines.append("")

    num_esil_ops = sum(ir_counts.values())
    # Cyclomatic-complexity metric: see esil_block_cfg_counts /
    # pipeline/metrics/blocks/agnosticism.py's cyclomatic_complexity_delta.
    num_cfg_blocks, num_cfg_edges = esil_block_cfg_counts(r2, addr)
    text = "\n".join(lines)
    # Temporaries metric: always 0 by design -- ESIL is a stack-based
    # representation (an RPN expression per instruction) with no operator
    # that declares a named temporary, confirmed against radare2's own
    # `ae???` operator table (every operator is a stack math/compare/memory/
    # control op, none of them a temp declaration). A meaningful data point
    # in its own right (ESIL genuinely has no notion of a named temporary),
    # not a measurement gap -- see pipeline/metrics/blocks/temporaries.py.
    num_temp_vars = 0
    return (
        len(ops), num_esil_ops, num_uncovered, ir_counts, native_counts, num_temp_vars,
        op_histogram, text, num_cfg_blocks, num_cfg_edges,
    )


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

                esil_operators = load_esil_operators(r2)

                arch_key = r2_arch_key(r2)
                family = arch_family(*arch_key) if arch_key else None
                pc_register = pc_register_for(arch_key)
                # ARM interworking means a function can be Thumb even though
                # the binary is arm/32 overall
                disassemblers = {}

                def disassembler_for(func_bits):
                    if arch_key is None:
                        return None
                    arch, bits = arch_key
                    thumb = arch == "arm" and func_bits == 16
                    if thumb not in disassemblers:
                        disassemblers[thumb] = make_disassembler(arch, bits, thumb=thumb)
                    return disassemblers[thumb]

                target = functions if limit is None else functions[:limit]
                whole_binary_chunks = []
                for func in target:
                    addr = int(func["address"], 16)
                    record = {"function": func["name"], "address": func["address"]}
                    try:
                        md = disassembler_for(func.get("bits"))
                        (
                            n_native, n_esil_ops, n_uncovered, ir_counts, native_counts, n_temp_vars,
                            op_histogram, text, n_cfg_blocks, n_cfg_edges,
                        ) = lift_function(
                            r2, addr, func["name"], esil_operators, md, family, pc_register
                        )
                        record["status"] = "ok"
                        record["num_native_instructions"] = n_native
                        record["num_esil_ops"] = n_esil_ops
                        record["num_uncovered"] = n_uncovered
                        record["num_temp_vars"] = n_temp_vars
                        record["num_cfg_blocks"] = n_cfg_blocks
                        record["num_cfg_edges"] = n_cfg_edges
                        record["op_histogram"] = dict(op_histogram)
                        for cat in CATEGORIES:
                            record[f"ir_ops_{cat}"] = ir_counts[cat]
                            record[f"native_{cat}"] = native_counts[cat]
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
