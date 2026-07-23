"""Shared native-instruction classifier: arithmetic / control / memory / other.

Used by every backend's lift script so the *denominator* side of the
per-category expansion ratio (see pipeline/metrics/blocks/expansion_ratio.py) is
computed by one shared rule across all five backends, instead of one
taxonomy per backend. Driven by capstone (already a transitive dependency
via angr/pyvex -- see requirements.txt) rather than each tool's own notion
of instruction semantics, so it works the same way whether the caller
already has a live capstone object (angr's block.capstone) or only raw
bytes (binja/pyghidra/r2/retdec, decoded here via make_disassembler).

Classification rule, in priority order:
  1. control: capstone's own CS_GRP_JUMP/CALL/RET/IRET/INT/BRANCH_RELATIVE,
     PLUS two gaps found by testing real ARM/MIPS binaries from this
     project's own corpus through capstone 5.0.6 (see binaries/*/arm,
     binaries/*/mips) that its groups alone don't catch:
       - MIPS `jal` (direct call): capstone tags `jalr` (register-indirect
         call) with CS_GRP_CALL but leaves plain `jal` in no control group
         at all -- confirmed by disassembling a real jal from this repo's
         mips binaries and inspecting insn.groups.
       - ARM computed jumps/returns encoded as ordinary data-processing or
         load-multiple instructions writing pc directly (e.g. `pop {r4, fp,
         pc}` as the epilogue return, or `ldr pc, [pc, r3, lsl #2]` as a
         jump-table dispatch) -- capstone reports no groups at all for
         these since, per the ARM encoding, they're not distinct
         "branch" opcodes. Any instruction with pc as a destination
         operand is control flow regardless of its mnemonic.
  2. memory: any operand capstone marks as a memory operand (covers
     load/store addressing forms across all four architectures), plus an
     explicit mnemonic check for push/pop -- their stack access is
     implicit, so capstone gives them no memory *operand* to detect.
     `lea` is excluded even though its operand syntax looks like a memory
     reference: it only computes an address, never touches memory.
  3. arithmetic: mnemonic matches a per-architecture table of ALU/logic
     opcodes (arithmetic, bitwise, shift, compare) -- capstone has no
     generic "this is an ALU op" group the way it does for control flow.
  4. other: everything else (data movement, nop, csel/cmov, misc).

Only x86/ARM/ARM64/MIPS(EB) at 32/64-bit are covered, matching this
project's binaries/ corpus (BinKit-style: x86, arm, mips, mipseb).
"""

import capstone as cs

_ARCH_MODE = {
    ("x86", 32): (cs.CS_ARCH_X86, cs.CS_MODE_32),
    ("x86", 64): (cs.CS_ARCH_X86, cs.CS_MODE_64),
    ("arm", 32): (cs.CS_ARCH_ARM, cs.CS_MODE_ARM),
    ("arm", 64): (cs.CS_ARCH_ARM64, cs.CS_MODE_ARM),
    ("mips", 32): (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS32),
    ("mips", 64): (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS64),
    ("mipseb", 32): (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS32),
    ("mipseb", 64): (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS64),
}

# BinKit encodes MIPS endianness in the architecture name itself ("mips" ==
# mipsel/little, "mipseb" == big) rather than as a separate field; x86/arm
# in this corpus are always little-endian.
_BIG_ENDIAN_ARCHES = {"mipseb"}

_MEM_OPERAND_TYPE = {
    cs.CS_ARCH_X86: cs.x86.X86_OP_MEM,
    cs.CS_ARCH_ARM: cs.arm.ARM_OP_MEM,
    cs.CS_ARCH_ARM64: cs.arm64.ARM64_OP_MEM,
    cs.CS_ARCH_MIPS: cs.mips.MIPS_OP_MEM,
}
_REG_OPERAND_TYPE = {
    cs.CS_ARCH_X86: cs.x86.X86_OP_REG,
    cs.CS_ARCH_ARM: cs.arm.ARM_OP_REG,
    cs.CS_ARCH_ARM64: cs.arm64.ARM64_OP_REG,
    cs.CS_ARCH_MIPS: cs.mips.MIPS_OP_REG,
}

CONTROL_GROUPS = {
    cs.CS_GRP_JUMP,
    cs.CS_GRP_CALL,
    cs.CS_GRP_RET,
    cs.CS_GRP_IRET,
    cs.CS_GRP_INT,
    cs.CS_GRP_BRANCH_RELATIVE,
}

_MEM_MNEMONICS = {"push", "pop", "pusha", "pushad", "popa", "popad"}
# ARM load/store-multiple: capstone models their register list as plain REG
# operands (no MEM-typed operand the way ldr/str get), and their own group
# doesn't overlap CONTROL_GROUPS or a "memory access" group -- confirmed by
# disassembling a real `ldm lr!, {r0-r3}` from this repo's arm binaries and
# finding every operand typed ARM_OP_REG. push/pop (STMDB/LDMIA aliases on
# sp) already match _MEM_MNEMONICS by their printed mnemonic, so this only
# needs to catch the non-aliased forms.
_MEM_MNEMONIC_PREFIXES = ("ldm", "stm")
_MEM_EXCLUDE_MNEMONICS = {"lea"}

_ARITH_PREFIXES = {
    "x86": (
        "add", "adc", "sub", "sbb", "inc", "dec", "neg",
        "mul", "imul", "div", "idiv",
        "and", "or", "xor", "not",
        "shl", "shr", "sar", "sal", "rol", "ror", "rcl", "rcr",
        "cmp", "test",
        # SSE/SSE2 scalar float arithmetic/compare -- e.g. `ucomiss` (float
        # compare) surfaced as a real gap when validated against a real x86-64
        # binary from this repo (r2 tags it "cmp", matching integer cmp's
        # category, but it isn't covered by the integer mnemonics above).
        "addss", "addsd", "subss", "subsd", "mulss", "mulsd", "divss", "divsd",
        "ucomiss", "ucomisd", "comiss", "comisd",
    ),
    "arm": (
        "add", "adc", "sub", "subs", "sbc", "rsb", "rsc",
        "mul", "mla", "mls", "smull", "umull", "smlal", "umlal", "sdiv", "udiv",
        "and", "orr", "orn", "eor", "bic", "mvn",
        "lsl", "lsr", "asr", "ror", "rrx",
        "cmp", "cmn", "tst", "teq",
        "clz", "rev",
    ),
    "arm64": (
        "add", "adc", "sub", "sbc", "madd", "msub",
        "mul", "mneg", "smull", "umull", "sdiv", "udiv",
        "and", "orr", "orn", "eor", "eon", "mvn", "bic",
        "lsl", "lsr", "asr", "ror",
        "cmp", "cmn", "tst",
        "clz", "rev", "neg",
    ),
    "mips": (
        "add", "addu", "addi", "addiu",
        "sub", "subu",
        "mul", "mult", "multu", "div", "divu",
        "and", "andi", "or", "ori", "xor", "xori", "nor",
        "sll", "srl", "sra", "sllv", "srlv", "srav", "rotr",
        "slt", "slti", "sltu", "sltiu",
        "clz", "clo", "neg", "not",
        # MIPS64 doubleword ALU ops -- a distinct "d"-prefixed mnemonic
        # family (not just a same-root suffix like the ARM condition codes
        # above), so startswith() on the 32-bit names above doesn't reach
        # them. Found missing by validating against a real mips/64 binary
        # from this repo (`daddiu`, `dsubu`, `ddivu`, ... all fell through
        # to "other" before this).
        "dadd", "daddu", "daddi", "daddiu",
        "dsub", "dsubu",
        "dmul", "dmult", "dmultu", "ddiv", "ddivu",
        "dsll", "dsrl", "dsra", "dsllv", "dsrlv", "dsrav", "drotr",
        "dclz", "dclo", "dneg",
    ),
}
# ARM64 shares the "arm" family for control/pc-write purposes only where
# applicable (see classify_insn) -- its arithmetic table is separate above
# since mnemonics/opcodes genuinely differ (e.g. madd/msub, no rsb/rsc).


def make_disassembler(arch, bits, thumb=False):
    """A ready-to-use `capstone.Cs` for one (arch, bits) pair from this
    project's metadata.json convention (arch in {x86,arm,mips,mipseb}, bits
    in {32,64}), or None if unsupported. `thumb=True` switches ARM32 to
    Thumb mode for callers that know a given function/block is Thumb (e.g.
    r2's per-function "bits": 16); ARM64 has no such per-instruction-set
    ambiguity, and x86/MIPS ignore the flag.
    """
    key = (arch, bits)
    if key not in _ARCH_MODE:
        return None
    cs_arch, cs_mode = _ARCH_MODE[key]
    if cs_arch == cs.CS_ARCH_ARM and thumb:
        cs_mode = cs.CS_MODE_THUMB
    cs_mode |= cs.CS_MODE_BIG_ENDIAN if arch in _BIG_ENDIAN_ARCHES else cs.CS_MODE_LITTLE_ENDIAN
    md = cs.Cs(cs_arch, cs_mode)
    md.detail = True
    return md


def cs_arch_for(arch, bits):
    """The CS_ARCH_* constant for one (arch, bits) pair, or None if
    unsupported -- for callers (e.g. angr_lift.py) that already have
    correctly-decoded capstone instructions of their own (angr's
    block.capstone) and only need the CS_ARCH_* id to call classify_insn,
    not a whole make_disassembler().
    """
    key = (arch, bits)
    return _ARCH_MODE[key][0] if key in _ARCH_MODE else None


def arch_family(arch, bits):
    """Normalized key into _ARITH_PREFIXES / the ARM pc-write check: "arm64"
    for 64-bit arm, else the (arch, bits)-independent family name.
    """
    if arch == "arm" and bits == 64:
        return "arm64"
    if arch in ("mips", "mipseb"):
        return "mips"
    return arch


def _writes_pc(insn, cs_arch):
    """True if any operand is the architecture's program-counter register
    -- catches ARM epilogue returns / computed jumps encoded as ordinary
    data-processing or load-multiple instructions (see module docstring).
    Only meaningful for 32-bit ARM; a no-op (always False) elsewhere since
    other architectures don't expose pc as a directly nameable operand in
    compiled code.
    """
    if cs_arch != cs.CS_ARCH_ARM:
        return False
    reg_type = _REG_OPERAND_TYPE[cs_arch]
    for op in insn.operands:
        if op.type == reg_type and insn.reg_name(op.value.reg) == "pc":
            return True
    return False


def classify_insn(insn, cs_arch, family):
    """Classify one already-decoded capstone CsInsn (detail=True) into
    "arithmetic" / "control" / "memory" / "other". `cs_arch` is the
    CS_ARCH_* the instruction was decoded with; `family` is the
    arch_family() key selecting the arithmetic-mnemonic table.
    """
    groups = set(insn.groups)
    mnemonic = insn.mnemonic.lower()

    if groups & CONTROL_GROUPS:
        return "control"
    if family == "mips" and mnemonic in ("jal", "break"):
        # capstone MIPS gaps found by validating against real mips/32
        # binaries from this repo: `jal` (direct call) gets no CS_GRP_CALL,
        # and `break` (the compiler-inserted trap for div-by-zero/bounds
        # checks) gets no CS_GRP_INT, unlike x86/ARM's equivalents.
        return "control"
    if _writes_pc(insn, cs_arch):
        return "control"

    # Arithmetic is checked before memory: on CISC (x86) an ALU op can take
    # a memory operand directly (`cmp dword [ebp-4], 0`, `add [mem], eax`)
    # without ceasing to be an arithmetic/compare op -- confirmed against
    # r2's own type field, which likewise calls these "cmp"/"add"/"sub" and
    # not "load"/"store", on real x86 binaries from this repo. "memory" is
    # reserved for instructions whose primary job *is* moving data, not any
    # instruction that happens to touch memory as a side channel.
    #
    # startswith(), not exact match: ARM/ARM64 glue a condition code and/or
    # flag-setting "s" directly onto the base mnemonic with no separator
    # (`andeq`, `adds`, `cmpeq`, `orrs`) -- confirmed these all fell through
    # to "other" under exact matching, against a real arm/32 binary from
    # this repo. This is safe to do only because control flow is already
    # fully resolved above via capstone groups/pc-write, independent of
    # mnemonic text -- otherwise "b"-prefix matching would misfire on
    # unrelated arithmetic mnemonics like "bic".
    if mnemonic.startswith(_ARITH_PREFIXES.get(family, ())):
        return "arithmetic"

    if mnemonic in _MEM_MNEMONICS or mnemonic.startswith(_MEM_MNEMONIC_PREFIXES):
        return "memory"
    if mnemonic not in _MEM_EXCLUDE_MNEMONICS:
        mem_type = _MEM_OPERAND_TYPE.get(cs_arch)
        if mem_type is not None and any(op.type == mem_type for op in insn.operands):
            return "memory"

    return "other"


def classify_bytes(md, data, addr, family):
    """Decode and classify every instruction in `data` (starting at `addr`)
    with disassembler `md` (from make_disassembler). Yields
    (address, size, mnemonic, category) per decoded instruction; undecodable
    trailing bytes are silently dropped (mirrors capstone's own disasm()
    behavior).
    """
    cs_arch = md.arch
    for insn in md.disasm(data, addr):
        category = classify_insn(insn, cs_arch, family)
        yield insn.address, insn.size, insn.mnemonic, category
