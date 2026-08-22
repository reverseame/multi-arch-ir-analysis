"""Shared helpers for the lifting pipeline's backend scripts.

Each backend script (pipeline/backends/*_lift.py) is a standalone CLI tool
that takes a --binary and --outdir, lists the binary's functions, lifts them
to that backend's IR, and writes functions.json / summary.json /
whole_binary.<level>.txt under --outdir. This module holds the pieces that
are identical across backends.
"""

import argparse
import json
import re
import time
from pathlib import Path

ELF_MAGIC = b"\x7fELF"

# BinKit-style dataset naming: <project>-<pver>_<compiler>-<cver>_<arch>_<bits>_<opt>_<name>.<ext>
# e.g. coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf
_BINKIT_RE = re.compile(
    r"^(?P<project>[A-Za-z0-9]+)-(?P<project_version>[0-9][0-9.]*)_"
    r"(?P<compiler>[A-Za-z0-9]+)-(?P<compiler_version>[0-9][0-9.]*)_"
    r"(?P<arch>[A-Za-z0-9]+)_(?P<bits>\d+)_"
    r"(?P<opt>O[0-9sz]+)_(?P<name>.+)\.(?P<ext>[A-Za-z0-9]+)$"
)


def parse_binary_metadata(path):
    """Best-effort parse of the dataset's filename convention.

    Falls back to just the file stem/arch="unknown" when a binary doesn't
    follow the convention (e.g. a one-off test binary), so discovery never
    fails just because a name doesn't match.
    """
    path = Path(path)
    match = _BINKIT_RE.match(path.name)
    if match:
        meta = match.groupdict()
        meta["bits"] = int(meta["bits"])
    else:
        meta = {
            "project": None,
            "project_version": None,
            "compiler": None,
            "compiler_version": None,
            "arch": "unknown",
            "bits": None,
            "opt": None,
            "name": path.stem,
            "ext": path.suffix.lstrip("."),
        }
    meta["path"] = str(path)
    meta["filename"] = path.name
    return meta


def discover_binaries(root):
    """Recursively find ELF files under root by checking the magic bytes,
    not the extension -- the dataset mixes .elf suffixes with none at all.
    """
    root = Path(root)
    found = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            with open(candidate, "rb") as f:
                if f.read(4) == ELF_MAGIC:
                    found.append(candidate)
        except OSError:
            continue
    return found


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_function_entry(name, address, size=None, external=None, thunk=None, extra=None):
    """Normalize a function record to a common schema across backends.
    address may be an int or a string; always stored as a hex string.
    """
    if isinstance(address, int):
        address = hex(address)
    entry = {
        "name": name,
        "address": address,
        "size": size,
        "external": external,
        "thunk": thunk,
    }
    if extra:
        entry.update(extra)
    return entry


class Timer:
    """Context manager measuring wall-clock duration of a block, in seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.duration_s = None
        return self

    def __exit__(self, *exc_info):
        self.duration_s = time.perf_counter() - self._start
        return False


def build_backend_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--binary", required=True, type=Path, help="Path to the ELF binary to lift")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write functions.json/summary.json/whole_binary.<level>.txt")
    parser.add_argument("--limit", type=int, default=None, help="Only lift the first N functions (debugging/smoke tests)")
    return parser


def write_whole_binary_dump(outdir, filename, chunks):
    """Concatenate per-function IR text into a single whole-binary IR file,
    ordered by address (chunks may arrive in whatever order the backend's
    own function iteration used, which isn't always address order).

    chunks: iterable of (address_int, text).
    """
    ordered = sorted(chunks, key=lambda c: c[0])
    path = Path(outdir) / filename
    # Written as many small write() calls rather than one path.write_text()
    # with the whole joined string to avoid writting errors
    with path.open("w") as f:
        first = True
        for _, text in ordered:
            if not first:
                f.write("\n")
            f.write(text)
            first = False
    return path


def write_summary(outdir, backend, binary, num_functions, lifted_records, duration_s, fatal_error=None):
    """Write summary.json for a backend run and return the summary dict.

    lifted_records: list of {"function", "address", "status", "error"}.
    status is "ok", "error", or "skipped" (e.g. external/library functions that
    were never expected to be lifted -- not counted as errors).
    """
    num_ok = sum(1 for r in lifted_records if r["status"] == "ok")
    num_error = sum(1 for r in lifted_records if r["status"] == "error")
    num_skipped = sum(1 for r in lifted_records if r["status"] == "skipped")
    summary = {
        "backend": backend,
        "binary": str(binary),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_functions": num_functions,
        "num_lifted_ok": num_ok,
        "num_lifted_error": num_error,
        "num_lifted_skipped": num_skipped,
        "duration_s": duration_s,
        "fatal_error": fatal_error,
    }
    write_json(Path(outdir) / "summary.json", summary)
    return summary
