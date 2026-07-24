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

Two classification granularities are provided, both used by expansion-ratio
metrics (pipeline/metrics/blocks/expansion_ratio.py):

  - classify_il_function_ops: one label per top-level instruction only
    (`il_func.instructions`) -- the same counting unit as
    num_{llil,mlil,hlil}_instructions.
  - classify_il_function_ast: walks the *entire* expression tree (every
    instruction and every sub-expression, via `il_func.traverse()` -- the
    same traversal pipeline/metrics/binja_invariance.py already uses for
    opcode-frequency histograms). This matters most at HLIL: a single
    top-level `if (a + b > c)` is one HLIL_IF instruction, but the addition
    and comparison it contains are real operations that would otherwise
    vanish from the count entirely -- undercounting is exactly the failure
    mode being measured (IR verbosity/expansion), so it can't be skipped.

The two aren't meant to sum to the same total -- they're deliberately
different units, reported side by side.

Type-conversion/reinterpret ops (SX, ZX, LOW_PART, FLOAT_CONV, FLOAT_TO_INT,
INT_TO_FLOAT, BOOL_TO_INT, ROUND_TO_INT, FLOOR, CEIL, FTRUNC) are
deliberately left uncategorized ("other"), mirroring the same call made for
VEX's analogous width-cast Iop_ ops in angr_lift.py -- they move/reinterpret
a value rather than compute a new one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binaryninja import VariableSourceType

from pipeline.native_classify import arch_family, classify_bytes, cs_arch_for, make_disassembler

# LLIL temp registers/flags are encoded with their high bit set (see
# binaryninja.lowlevelil.ILRegister.temp/ILFlag.temp); MLIL/HLIL have no
# temp_reg_count-style helper of their own, but Binary Ninja promotes an
# unresolved LLIL temp into a full Variable that keeps that same encoding in
# its .storage field so MLIL/HLIL temps are detected via that bit
# instead of a level-specific API.
_TEMP_STORAGE_BIT = 0x80000000


def is_bnil_temp_var(v):
    """True if MLIL/HLIL Variable `v` is Binary Ninja's promoted form of an
    unresolved LLIL temp register (temp#N), not a real named/recovered local
    or parameter. See module-level comment above for how this is encoded.
    """
    return v.source_type == VariableSourceType.RegisterVariableSourceType and bool(v.storage & _TEMP_STORAGE_BIT)

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

# Pure register/variable/flag assignment forms -- SET_REG, SET_VAR, SET_FLAG,
# HLIL_ASSIGN, and their split/field/SSA variants. Their own top-level
# operation carries no information about what's being computed (every one of
# them would otherwise land in "other"); the real operation is nested in
# their *source expression* (`.src`), e.g. `esp = esp - 8` is one
# LLIL_SET_REG instruction wrapping a nested SUB. Mirrors angr_lift.py's
# treatment of VEX's Ist_WrTmp/Ist_Put, which inherit from stmt.data for
# exactly the same reason. Deliberately excludes the _MEM_SSA forms
# (ASSIGN_MEM_SSA, ASSIGN_UNPACK_MEM_SSA) already in _MEMORY_SUFFIXES above
# -- a memory *write* is a memory op by its own destination, not by its
# source, matching VEX's Ist_Store staying "memory" regardless of what it
# stores.
_ASSIGN_SUFFIXES = {
    "SET_REG", "SET_REG_SPLIT", "SET_REG_SSA", "SET_REG_SPLIT_SSA", "SET_REG_SSA_PARTIAL",
    "SET_REG_STACK_REL", "SET_REG_STACK_REL_SSA", "SET_REG_STACK_ABS_SSA",
    "SET_FLAG", "SET_FLAG_SSA",
    "SET_VAR", "SET_VAR_FIELD", "SET_VAR_SPLIT", "SET_VAR_SPLIT_SSA",
    "SET_VAR_SSA", "SET_VAR_SSA_FIELD", "SET_VAR_ALIASED", "SET_VAR_ALIASED_FIELD",
    "ASSIGN", "ASSIGN_UNPACK",
}

# Escape-valve metric: INTRINSIC is BNIL's own generic fallback for
# instructions Binary Ninja's own IL can't express as primitive operations
# and instead models as an opaque call to a named architecture-specific
# intrinsic (e.g. x86 CPUID, ARM NEON/coprocessor ops) -- the closest BNIL
# analog to P-code's CALLOTHER / VEX's Ist_Dirty. See
# pipeline/metrics/blocks/robustness.py's escape_fraction.
_ESCAPE_SUFFIXES = {"INTRINSIC", "INTRINSIC_SSA", "MEMORY_INTRINSIC_SSA", "MEMORY_INTRINSIC_OUTPUT_SSA"}

_LEVEL_PREFIXES = ("LLIL_", "MLIL_", "HLIL_")


def _il_suffix(operation_name):
    """`operation.name` (e.g. "LLIL_ADD") with its level prefix stripped."""
    for prefix in _LEVEL_PREFIXES:
        if operation_name.startswith(prefix):
            return operation_name[len(prefix):]
    return operation_name


def classify_il_operation(operation_name):
    """`operation.name` (e.g. "LLIL_ADD", "HLIL_DEREF") -> "arithmetic" /
    "control" / "memory" / "other".
    """
    suffix = _il_suffix(operation_name)
    if suffix in _CONTROL_SUFFIXES:
        return "control"
    if suffix in _MEMORY_SUFFIXES:
        return "memory"
    if suffix in _ARITH_SUFFIXES:
        return "arithmetic"
    return "other"


def classify_il_instruction_ops(instr):
    """One top-level IL instruction -> "arithmetic"/"control"/"memory"/
    "other", the ops-level / flat granularity. Pure assignment forms
    (see _ASSIGN_SUFFIXES) inherit from their source expression's own
    top-level operation (`instr.src.operation.name`) instead of their own --
    otherwise `esp = esp - 8` would classify as "other" via SET_REG rather
    than "arithmetic" via the nested SUB it wraps.
    """
    suffix = _il_suffix(instr.operation.name)
    if suffix in _ASSIGN_SUFFIXES:
        return classify_il_operation(instr.src.operation.name)
    return classify_il_operation(instr.operation.name)


def is_il_instruction_escape(instr):
    """One top-level IL instruction -> True if it's (or wraps, via the same
    assign-inheritance as classify_il_instruction_ops) BNIL's INTRINSIC
    escape-valve operation -- e.g. `var = __intrinsic(...)` is one
    LLIL_SET_REG instruction wrapping a nested LLIL_INTRINSIC, the same
    inheritance classify_il_instruction_ops already applies for arithmetic/
    control/memory.
    """
    suffix = _il_suffix(instr.operation.name)
    if suffix in _ASSIGN_SUFFIXES:
        suffix = _il_suffix(instr.src.operation.name)
    return suffix in _ESCAPE_SUFFIXES


def classify_il_function_escape(il_func):
    """Count of top-level instructions in `il_func` that are BNIL's
    INTRINSIC escape valve -- the ops-level granularity, same counting unit
    as num_{llil,mlil,hlil}_instructions (see classify_il_function_ops).
    """
    return sum(1 for instr in il_func.instructions if is_il_instruction_escape(instr))


def classify_il_function_ops(il_func):
    """{"arithmetic": n, "control": n, "memory": n, "other": n} over every
    top-level instruction in `il_func` (LLIL/MLIL/HLIL) only -- the ops-level
    / flat granularity, one label per instruction, not descending into
    sub-expressions. See module docstring for how this differs from
    classify_il_function_ast.
    """
    counts = {"arithmetic": 0, "control": 0, "memory": 0, "other": 0}
    for instr in il_func.instructions:
        counts[classify_il_instruction_ops(instr)] += 1
    return counts


def classify_il_function_ast(il_func):
    """{"arithmetic": n, "control": n, "memory": n, "other": n} over every
    instruction and sub-expression in `il_func` (LLIL/MLIL/HLIL), via the
    same il_func.traverse() binja_invariance.py already uses.
    """
    counts = {"arithmetic": 0, "control": 0, "memory": 0, "other": 0}
    for operation in il_func.traverse(lambda instr: instr.operation):
        counts[classify_il_operation(operation.name)] += 1
    return counts


# Structural (nested statement *block*) operand names, not expression
# operands -- matches HighLevelILInstruction.traverse()'s own default
# (shallow=True) blacklist (see binaryninja/highlevelil.py). Only relevant
# for HLIL, which has structured control flow (WHILE/DO_WHILE/FOR/SWITCH/
# IF-as-block); LLIL/MLIL's IF/GOTO targets are plain basic-block-index
# ints, never nested instruction lists, so this is a no-op for them.
# Without this, a HLIL_WHILE's "body" operand (its entire loop body, often
# dozens of unrelated statements) would count as expression nesting inside
# the while-statement's own depth.
_STRUCTURAL_OPERAND_NAMES = {"true", "false", "body", "cases", "default"}


def bnil_instruction_depth(instr):
    """How many levels deep `instr`'s own expression tree nests -- 1 for a
    leaf/flat instruction with no nested instruction-valued operand, +1 for
    each level of embedded sub-expression (e.g. "eax = 4" is depth 1, "eax =
    ebx + 4" is depth 2: the SET_REG/SET_VAR/VAR_INIT's own node plus the
    ADD nested inside it).
    """
    max_child_depth = 0
    for name, op, _ in instr.detailed_operands:
        if name in _STRUCTURAL_OPERAND_NAMES:
            continue
        if hasattr(op, "detailed_operands"):
            max_child_depth = max(max_child_depth, bnil_instruction_depth(op))
        elif isinstance(op, list):
            for item in op:
                if hasattr(item, "detailed_operands"):
                    max_child_depth = max(max_child_depth, bnil_instruction_depth(item))
    return 1 + max_child_depth


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
