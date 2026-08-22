"""Read raw instruction bytes straight out of an on-disk ELF by virtual
address, with no live decompiler/lifter session required.

Every backend's own native-instruction classification (used by expansion_ratio
and temporaries) only runs for functions it *successfully* lifted -- the byte
-reading happens inside each backend's already-open session (Ghidra's
listing, angr's block loader, Binary Ninja's bv.read, r2's op bytes, IDA's
ida_bytes.get_bytes), which by construction is never reached for a function
whose IR lift raised before getting there.

The robustness block's error-rate-by-category metric needs native category
counts for *every* function a backend attempted, including the ones that
failed -- so it can't reuse those per-backend paths. This module reads the
same bytes a different way: straight from the ELF's own section table
(virtual address -> file offset), independent of whether any lifting was
attempted at all. Uses pyelftools (already an angr transitive dependency,
see requirements.txt) rather than re-deriving ELF layout by hand.
"""

from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile


class ElfByteReader:
    """Opens one ELF once and answers read(va, size) by locating the
    SHF_ALLOC section containing va and seeking to its file offset. Sections
    are collected once at construction time (a handful per binary), not
    re-scanned per read.
    """

    def __init__(self, binary_path):
        self._f = open(binary_path, "rb")
        elf = ELFFile(self._f)
        self._sections = [
            (s["sh_addr"], s["sh_addr"] + s["sh_size"], s["sh_offset"], s["sh_type"])
            for s in elf.iter_sections()
            if s["sh_flags"] & SH_FLAGS.SHF_ALLOC and s["sh_addr"] != 0
        ]

    def read(self, va, size):
        """Bytes at [va, va+size), or None if va isn't covered by any
        loadable section, the section has no real file content (SHT_NOBITS,
        e.g. .bss), or size is 0. Silently truncates to a section's end if
        [va, va+size) crosses out of it -- native_classify.classify_bytes
        already drops undecodable trailing bytes the same way capstone's own
        disasm() does, so a short read isn't a correctness problem.
        """
        if not size:
            return None
        for start, end, offset, sh_type in self._sections:
            if start <= va < end:
                if sh_type == "SHT_NOBITS":
                    return None
                self._f.seek(offset + (va - start))
                return self._f.read(min(size, end - va))
        return None

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
