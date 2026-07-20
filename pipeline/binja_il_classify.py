"""Shared Binary Ninja classification helpers for binja_llil_lift.py,
binja_mlil_lift.py, and binja_hlil_lift.py: an IL-operation classifier
(arithmetic/control/memory/other) plus a native-instruction classifier
built on pipeline/native_classify.py.

Binary Ninja names its three IL levels' `.operation` enum members with
near-identical suffixes after the level prefix (LLIL_ADD/MLIL_ADD/HLIL_ADD,
LLIL_STORE/MLIL_STORE/HLIL_ASSIGN_MEM_SSA, LLIL_JUMP/MLIL_JUMP/HLIL_JUMP,
...) -- confirmed by enumerating LowLevelILOperation/MediumLevelILOperation/
HighLevelILOperation directly. That means one suffix-keyed table, applied
after stripping the level's own prefix, covers all three levels instead of
duplicating near-identical tables three times.

Classification walks the *entire* expression tree (every instruction and
every sub-expression, via `il_func.traverse()` -- the same traversal
pipeline/metrics/binja_invariance.py already uses for opcode-frequency
histograms), not just top-level instructions. This matters most at HLIL:
a single top-level `if (a + b > c)` is one HLIL_IF instruction, but the
addition and comparison it contains are real operations that would
otherwise vanish from the count entirely -- undercounting is exactly the
failure mode being measured (IR verbosity/expansion), so it can't be
skipped. This is a deliberately different unit than num_{llil,mlil,hlil}_
instructions (top-level only, existing field, left unchanged) -- the two
aren't meant to sum to the same total.

Type-conversion/reinterpret ops (SX, ZX, LOW_PART, FLOAT_CONV, FLOAT_TO_INT,
INT_TO_FLOAT, BOOL_TO_INT, ROUND_TO_INT, FLOOR, CEIL, FTRUNC) are
deliberately left uncategorized ("other"), mirroring the same call made for
VEX's analogous width-cast Iop_ ops in angr_lift.py -- they move/reinterpret
a value rather than compute a new one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.native_classify import arch_family, classify_bytes, cs_arch_for, make_disassembler

_ARITH_SUFFIXES = {
    "ADD", "ADC", "SUB", "SBB",
    "AND", "OR", "XOR", "NOT", "NEG",
    "LSL", "LSR", "ASR", "ROL", "RLC", "ROR", "RRC",
    "MUL", "MULU_DP", "MULS_DP",
    "DIVU", "DIVU_DP", "DIVS", "DIVS_DP", "MODU", "MODU_DP", "MODS", "MODS_DP",
    "CMP_E", "CMP_NE", "CMP_SLT", "CMP_ULT", "CMP_SLE", "CMP_ULE",
    "CMP_SGE", "CMP_UGE", "CMP_SGT", "CMP_UGT", "TEST_BIT", "ADD_OVERFLOW",
    "FADD", "FSUB", "FMUL", "FDIV", "FSQRT", "FNEG", "FABS",
    "FCMP_E", "FCMP_NE", "FCMP_LT", "FCMP_LE", "FCMP_GE", "FCMP_GT", "FCMP_O", "FCMP_UO",
}

_CONTROL_SUFFIXES = {
    "JUMP", "JUMP_TO", "CALL", "CALL_STACK_ADJUST", "TAILCALL", "RET", "NORET",
    "IF", "GOTO", "SYSCALL", "TRAP", "BP",
    # HLIL-only structured control flow (harmless no-ops at LLIL/MLIL since
    # these suffixes never occur in those enums).
    "WHILE", "DO_WHILE", "FOR", "SWITCH", "CASE", "BREAK", "CONTINUE",
    "LABEL", "UNREACHABLE",
    # SSA-form variants
    "CALL_SSA", "TAILCALL_SSA", "SYSCALL_SSA", "WHILE_SSA", "DO_WHILE_SSA", "FOR_SSA",
}

_MEMORY_SUFFIXES = {
    "LOAD", "STORE", "LOAD_STRUCT", "STORE_STRUCT", "PUSH", "POP",
    "DEREF", "DEREF_FIELD",
    "LOAD_SSA", "STORE_SSA", "LOAD_STRUCT_SSA", "STORE_STRUCT_SSA",
    "DEREF_SSA", "DEREF_FIELD_SSA", "ASSIGN_MEM_SSA", "ASSIGN_UNPACK_MEM_SSA",
}

_LEVEL_PREFIXES = ("LLIL_", "MLIL_", "HLIL_")


def classify_il_operation(operation_name):
    """`operation.name` (e.g. "LLIL_ADD", "HLIL_DEREF") -> "arithmetic" /
    "control" / "memory" / "other".
    """
    suffix = operation_name
    for prefix in _LEVEL_PREFIXES:
        if operation_name.startswith(prefix):
            suffix = operation_name[len(prefix):]
            break

    if suffix in _CONTROL_SUFFIXES:
        return "control"
    if suffix in _MEMORY_SUFFIXES:
        return "memory"
    if suffix in _ARITH_SUFFIXES:
        return "arithmetic"
    return "other"


def classify_il_function(il_func):
    """{"arithmetic": n, "control": n, "memory": n, "other": n} over every
    instruction and sub-expression in `il_func` (LLIL/MLIL/HLIL), via the
    same il_func.traverse() binja_invariance.py already uses.
    """
    counts = {"arithmetic": 0, "control": 0, "memory": 0, "other": 0}
    for operation in il_func.traverse(lambda instr: instr.operation):
        counts[classify_il_operation(operation.name)] += 1
    return counts


# --- native-instruction classification (shared with the other 4 backends
# via pipeline/native_classify.py) --------------------------------------

_BINJA_ARCH_KEY = {
    "x86": ("x86", 32),
    "x86_64": ("x86", 64),
    "armv7": ("arm", 32),
    "armv7eb": ("arm", 32),
    "thumb2": ("arm", 32),
    "thumb2eb": ("arm", 32),
    "aarch64": ("arm", 64),
    "mipsel32": ("mips", 32),
    "mips32": ("mipseb", 32),
    "mipsel64": ("mips", 64),
    "mips64": ("mipseb", 64),
}


def binja_arch_key(arch_name):
    """(arch, bits, thumb) in this project's metadata.json convention (see
    pipeline/native_classify.py) for a Binary Ninja architecture name (e.g.
    func.arch.name -- checked per function, not once per binary, since ARM
    functions can be Thumb-interworked independently of the rest of the
    binary), or None if unsupported.
    """
    key = _BINJA_ARCH_KEY.get(arch_name)
    if key is None:
        return None
    return (*key, arch_name.startswith("thumb"))


def classify_native_instructions(func, bv):
    """{"arithmetic": n, "control": n, "memory": n, "other": n} over every
    native instruction in `func`, re-disassembling each via capstone (see
    pipeline/native_classify.py) from raw bytes read out of `bv` -- Binary
    Ninja doesn't hand out capstone objects the way angr does, so this is
    decoded independently rather than reusing Binary Ninja's own analysis.
    Replicates the (address, length)-tracking loop Function.instructions
    uses internally (it discards the address/length pair after using it to
    print tokens; those are exactly what's needed here to read the
    matching raw bytes).
    """
    counts = {"arithmetic": 0, "control": 0, "memory": 0, "other": 0}
    key = binja_arch_key(func.arch.name)
    if key is None:
        return counts
    arch, bits, thumb = key
    cs_arch = cs_arch_for(arch, bits)
    family = arch_family(arch, bits)
    md = make_disassembler(arch, bits, thumb=thumb)
    if md is None:
        return counts

    for block in func.basic_blocks:
        addr = block.start
        for _tokens, length in block:
            data = bv.read(addr, length)
            decoded = next(classify_bytes(md, data, addr, family), None)
            if decoded is not None:
                counts[decoded[3]] += 1
            addr += length
    return counts
