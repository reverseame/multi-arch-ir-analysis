"""PoC: inter-architecture invariance metrics (possible_metrics.txt) for a
single Binary Ninja IL level, computed across >=2 architecture builds of the
same source binary.

For each architecture build, extracts per function (matched by name across
all builds):
    - op-type frequency histogram, via il_func.traverse() over every
      instruction AND sub-instruction (not just top-level instructions --
      gives a real opcode vocabulary distribution, e.g. LLIL_ADD, LLIL_LOAD).
    - IR size: top-level instruction count.
    - cyclomatic complexity: E - N + 2, from the IL function's own basic
      blocks/edges (not the native disassembly CFG).

Then computes, per matched function (pairwise across builds, averaged when
more than 2 builds are given):
    - weighted_jaccard         (higher = better, [0,1])
    - jsd_similarity           (higher = better, [0,1])
    - ir_size_cv               (lower = better, [0,inf), across ALL builds at once)
    - cyclomatic_complexity_delta (lower = better, [0,inf))

Function match rate (precondition metric) is reported per binary: matched
functions / total functions in that binary.

Standalone CLI:
    python -m pipeline.metrics.binja_invariance --binary <path> --binary <path> [--binary <path> ...] \
        --level {llil,mlil,hlil} --outdir <dir>

python3 -m pipeline.metrics.binja_invariance \
    --binary binaries/ls/coreutils-8.29_gcc-6.4.0_x86_32_O1_ls.elf \
    --binary binaries/ls/coreutils-8.29_gcc-4.9.4_arm_32_O0_ls.elf \
    --level llil --outdir results/invariance/ls_llil
"""

import argparse
import csv
import itertools
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common import parse_binary_metadata, write_json  # noqa: E402
from pipeline.metrics.formulas import (  # noqa: E402
    coefficient_of_variation,
    cyclomatic_complexity,
    jensen_shannon_similarity,
    weighted_jaccard,
)

from binaryninja import load  # noqa: E402

LEVELS = ("llil", "mlil", "hlil")


def binary_label(path):
    meta = parse_binary_metadata(path)
    if meta["arch"] == "unknown":
        return Path(path).stem
    return f"{meta['arch']}_{meta['bits']}_{meta['opt']}"


def extract_function_metrics(bv, level):
    """name -> {op_histogram, ir_size, cyclomatic}. Functions with no IL
    body (e.g. external stubs with zero instructions) are dropped -- matching
    two empty functions across architectures is a trivial, misleading 1.0.
    """
    functions = {}
    for func in bv.functions:
        try:
            il_func = getattr(func, level)
            if il_func is None:
                continue

            ir_size = sum(1 for _ in il_func.instructions)
            if ir_size == 0:
                continue

            histogram = Counter(op.name for op in il_func.traverse(lambda instr: instr.operation))

            basic_blocks = list(il_func.basic_blocks)
            num_nodes = len(basic_blocks)
            num_edges = sum(len(bb.outgoing_edges) for bb in basic_blocks)
            cc = cyclomatic_complexity(num_nodes, num_edges)
        except Exception:
            continue

        functions[func.name] = {
            "op_histogram": histogram,
            "ir_size": ir_size,
            "cyclomatic": cc,
        }
    return functions


def run(binary_paths, level, outdir, limit):
    outdir.mkdir(parents=True, exist_ok=True)

    per_binary = {}
    binary_info = {}
    for path in binary_paths:
        label = binary_label(path)
        meta = parse_binary_metadata(path)
        print(f"Loading {path} (arch={meta['arch']}, label={label})...")
        with load(str(path)) as bv:
            per_binary[label] = extract_function_metrics(bv, level)
        binary_info[label] = {"path": str(path), **meta}

    labels = list(per_binary.keys())
    name_sets = [set(fns) for fns in per_binary.values()]
    matched_names = sorted(set.intersection(*name_sets)) if name_sets else []
    num_matched_total = len(matched_names)
    evaluated_names = matched_names[:limit] if limit is not None else matched_names

    pairs = list(itertools.combinations(labels, 2))
    rows = []
    for name in evaluated_names:
        row = {"function": name}
        sizes = [per_binary[l][name]["ir_size"] for l in labels]
        row["ir_size_cv"] = coefficient_of_variation(sizes)
        for l in labels:
            row[f"ir_size__{l}"] = per_binary[l][name]["ir_size"]
            row[f"cyclomatic__{l}"] = per_binary[l][name]["cyclomatic"]

        wj_vals, jsd_vals, ccd_vals = [], [], []
        for a, b in pairs:
            hist_a = per_binary[a][name]["op_histogram"]
            hist_b = per_binary[b][name]["op_histogram"]
            wj = weighted_jaccard(hist_a, hist_b)
            jsd = jensen_shannon_similarity(hist_a, hist_b)
            ccd = abs(per_binary[a][name]["cyclomatic"] - per_binary[b][name]["cyclomatic"])
            wj_vals.append(wj)
            jsd_vals.append(jsd)
            ccd_vals.append(ccd)
            if len(pairs) > 1:
                row[f"weighted_jaccard__{a}_vs_{b}"] = wj
                row[f"jsd_similarity__{a}_vs_{b}"] = jsd
                row[f"cyclomatic_delta__{a}_vs_{b}"] = ccd

        row["weighted_jaccard"] = statistics.mean(wj_vals)
        row["jsd_similarity"] = statistics.mean(jsd_vals)
        row["cyclomatic_complexity_delta"] = statistics.mean(ccd_vals)
        rows.append(row)

    def agg(key):
        values = [r[key] for r in rows if r[key] is not None]
        if not values:
            return None
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    summary = {
        "level": level,
        "binaries": {
            l: {
                **binary_info[l],
                "total_functions": len(per_binary[l]),
                "matched_functions": num_matched_total,
                "function_match_rate": (num_matched_total / len(per_binary[l])) if per_binary[l] else 0.0,
            }
            for l in labels
        },
        "num_matched_functions": num_matched_total,
        "num_functions_evaluated": len(evaluated_names),
        "metrics": {
            "weighted_jaccard_similarity": {"direction": "higher_is_better", "range": "[0,1]", **(agg("weighted_jaccard") or {})},
            "jsd_similarity": {"direction": "higher_is_better", "range": "[0,1]", **(agg("jsd_similarity") or {})},
            "ir_size_coefficient_of_variation": {"direction": "lower_is_better", "range": "[0,inf)", **(agg("ir_size_cv") or {})},
            "cyclomatic_complexity_delta": {"direction": "lower_is_better", "range": "[0,inf)", **(agg("cyclomatic_complexity_delta") or {})},
        },
    }
    write_json(outdir / "summary.json", summary)

    csv_path = outdir / "per_function_metrics.csv"
    if rows:
        fieldnames = ["function", "weighted_jaccard", "jsd_similarity", "ir_size_cv", "cyclomatic_complexity_delta"]
        fieldnames += sorted(k for k in rows[0] if k not in fieldnames and k != "function")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")

    print(f"\nLevel: {level}")
    for l in labels:
        b = summary["binaries"][l]
        print(f"  {l}: {b['total_functions']} functions, match_rate={b['function_match_rate']:.2%}")
    print(f"Matched functions: {num_matched_total} (evaluated: {len(evaluated_names)})")
    for metric_name, stats in summary["metrics"].items():
        if stats.get("n"):
            print(f"  {metric_name} ({stats['direction']}): mean={stats['mean']:.4f} median={stats['median']:.4f}")
        else:
            print(f"  {metric_name}: no data")
    print(f"\nWrote {outdir / 'summary.json'}")
    print(f"Wrote {csv_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Compute inter-architecture invariance metrics for a Binary Ninja IL level")
    parser.add_argument("--binary", action="append", required=True, type=Path,
                         help="Path to an architecture build's ELF binary (repeat for each architecture, need >=2)")
    parser.add_argument("--level", choices=LEVELS, default="llil", help="Binary Ninja IL level to compare (default: llil)")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write summary.json/per_function_metrics.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N matched functions (debugging/smoke tests)")
    args = parser.parse_args()

    if len(args.binary) < 2:
        parser.error("need at least 2 --binary paths to compute inter-architecture invariance")

    return run(args.binary, args.level, args.outdir, args.limit)


if __name__ == "__main__":
    sys.exit(main())
