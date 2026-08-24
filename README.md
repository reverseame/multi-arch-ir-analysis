<h1 align="center">Multi-Architecture IR Analysis</h1>
<h3 align="center"><sub><i>Comparative Evaluation of Intermediate Representations for Hardware Architecture Abstraction Capability</i></sub></h3>

This project compares intermediate representations (IR) produced by eight backends
(pyghidra/P-code, angr/VEX, radare2/ESIL, Binary Ninja LLIL/MLIL/HLIL,
RetDec/LLVM IR, IDA Pro/Hex-Rays microcode) across multiple CPU
architectures, for a set of metrics (cost, expansion ratio, temporaries,
robustness, agnosticism, nesting depth, SSA operations, verbosity). Built
for a Master's thesis (TFM) evaluating how architecture-agnostic each
backend's IR actually is.

The pipeline runs in two phases:

1. **Lifting** (`pipeline.run_lifting`) — lift every (binary, backend) pair
   and write each backend's raw IR + per-function status to `results/` (or the specified path).
2. **Metrics** (`pipeline.metrics.run_metrics`) — read `results/` (or the specified path), compute
   the metric blocks under `pipeline/metrics/blocks/`, and write aggregated
   JSON/CSV per backend.


## Repository layout

```
pipeline/
  run_lifting.py           phase 1 orchestrator
  common.py                shared helpers: BinKit filename parsing, binary discovery, IR dump helpers
  native_classify.py       classifies native (disassembled) instructions into arithmetic/control/memory/other
  binja_il_classify.py     BNIL-specific IR classification/traversal helpers shared by the 3 binja backends
  elf_bytes.py             reads raw instruction bytes straight from the ELF, independent of any backend
  backends/                one standalone CLI script per backend
    pyghidra_lift.py
    angr_lift.py
    r2_lift.py
    binja_llil_lift.py
    binja_mlil_lift.py
    binja_hlil_lift.py
    retdec_lift.py
    ida_lift.py                            
  metrics/
    run_metrics.py          phase 2 orchestrator
    registry.py             @register_block registration used by every file under blocks/
    formulas.py             shared aggregate-stats helpers (median/IQR/etc.)
    blocks/                 metric scripts  
      expansion_ratio.py
      temporaries.py
      nesting_depth.py
      ssa_operations.py
      cost.py
      robustness.py
      agnosticism.py
results/                   default output path (gitignored)
requirements.txt           pinned Python dependencies
```

## Requirements

- Python 3.12 (pipeline was developed/run against 3.12.3).
- A virtual environment is recommended:
  ```bash
  python3 -m venv IRanalysis
  source IRanalysis/bin/activate
  pip install -r requirements.txt
  ```
  If you don't activate the venv, pass `--python IRanalysis/bin/python3` to
  `run_lifting.py` instead (see Usage below) — each backend runs as its own
  subprocess using whichever Python executable you point it at.

### External backend tools

Four of the eight backends are not pip packages and must be installed and
licensed separately. The pipeline will error clearly (per-backend, per-binary,
not aborting the whole run) if one isn't reachable.

| Backend | Requirement |
|---|---|
| `pyghidra` | A Ghidra install, with either `GHIDRA_INSTALL_DIR` set to point at it, or Ghidra discoverable some other way `pyghidra` supports. Requires a JDK (developed against Temurin 21). |
| `r2` | The `radare2` CLI on `PATH` (`r2pipe` just talks to it over a pipe). |
| `binja_llil` / `binja_mlil` / `binja_hlil` | A licensed Binary Ninja install with its headless Python API installed into your interpreter (via Binary Ninja's own installer/`install_api.py`) — commercial license required for headless/API use. |
| `ida` | A licensed IDA Pro install with idalib activated: from the IDA install directory, `pip install idalib/python`, then run that install's `py-activate-idalib.py` once. `import idapro` must happen before any other `ida_*` import, which is why this backend runs as its own subprocess. |
| `retdec` | The `retdec-decompiler` CLI, found via `--retdec-bin`, the `RETDEC_BIN` env var, or `PATH` (checked in that order). |

`angr` and `pyelftools` (used directly by `pipeline/elf_bytes.py` and the
robustness metric block) are ordinary pip dependencies and need no extra setup.

### Docker (free backends only)

A prebuilt image provides the environment for the four free/open-source
backends (`pyghidra`, `angr`, `r2`, `retdec`) — Ghidra, radare2, RetDec,
Temurin JDK, and the pinned Python dependencies. Binary Ninja and IDA Pro
are **not** included (both require a paid, per-seat commercial license —
see the table above); run those two separately on a licensed host.

The image ships tools only, not this repo's code — mount your own checkout
of `pipeline/` in at run time so `git pull` alone picks up code updates,
no image rebuild needed:

```bash
docker pull ghcr.io/gitdebaro/multi-arch-ir-analysis:latest
# or build locally from the Dockerfile:
#   docker build -t multi-arch-ir-analysis .

docker run --rm -it ghcr.io/gitdebaro/multi-arch-ir-analysis:latest
```


## Corpus setup

Binaries are expected to follow the BinKit naming convention:
`<project>-<pver>_<compiler>-<cver>_<arch>_<bits>_<opt>_<name>.<ext>`
(e.g. `coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf`), staged under
`binaries/<tool>/<arch>/<bits>/<opt>/<filename>`. `pipeline/common.py`
parses arch/bits/opt/compiler straight from the filename; binaries that
don't match the pattern are still lifted, just without that metadata.

The binaries folder and all its content, Binkit subset with the 9 compiler versions (4.2 GB unzipped), can be obtained in the following link:

https://drive.google.com/file/d/1jMf_w0qf2cdomKUgRsNWDYZPOEe9tblF/view?usp=sharing

## Usage

### Phase 1 — lifting

```bash
# Everything under binaries/, all 8 backends
python -m pipeline.run_lifting --binaries-dir binaries

# A subset: one binary, two backends
python -m pipeline.run_lifting --binary binaries/ls/x86/32/O0/coreutils-8.29_gcc-8.2.0_x86_32_O0_ls.elf \
    --backends r2,angr

# Filter by architecture/bits/optimization level (comma-separated, matched against the BinKit filename)
python -m pipeline.run_lifting --binaries-dir binaries --arch arm,x86 --bits 32 --opt O0,O2
```

Useful flags: 
```
--results-dir (default ./results) 
--max-parallel (concurrent binary x backend subprocesses, default = CPU count)
--max-mem-mb (soft scheduling budget)
--timeout (per-run subprocess timeout, default 1800s)
--skip-existing / --resume (for continuing a large corpus run across sessions — see manifest.json / pause_state.json under --results-dir)

Run `python -m pipeline.run_lifting --help` for thefull list.
```

### Phase 2 — metrics

```bash
python -m pipeline.metrics.run_metrics --results-dir results

# or just a subset of metric blocks:
python -m pipeline.metrics.run_metrics --results-dir results --blocks cost,robustness
```

Reads `results/manifest.json` (written by phase 1) plus each binary's
`metadata.json`, then runs every registered block in
`pipeline/metrics/blocks/` (or only `--blocks`) and writes
`<outdir>/<block>.json` + `<block>.csv` (default outdir:
`<results-dir>/metrics`).


## Output layout

Under `--results-dir` (default `results/`):

- `manifest.json` — one entry per (binary, backend) run: status, output
  dir, timing/`rusage` stats.
- `<binary_stem>/<backend>/` — per-run output:
  - `functions.json` — every function `run_lifting` found for that binary.
  - `lift_records.json` — one record per attempted function: `status`
    (`"ok"` / `"error"` / `"skipped"`), IR-size counts, and on `"error"`,
    the exception type/message.
  - `summary.json` — counts (`num_lifted_ok`/`_error`/`_skipped`),
    `duration_s`, `fatal_error` if the whole run aborted.
  - `whole_binary.<level>.txt` — concatenated per-function IR text, address-ordered.
