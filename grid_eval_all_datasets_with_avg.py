#!/usr/bin/env python3
"""
Build 3x3 acc@0.5 tables (test formats 2/3/4 × train formats 2/3/4)
for *each dataset* using the corresponding pretrained bundle, and also an
AVERAGE grid across all datasets.

Assumptions (matched to your screenshots):
- CSVs:
  /home/ddn1/Documents/GitHub/SAPLMA/data/capital_true_false_instruct/format{2,3,4}/embeddings_with_labels_data/{dataset}LLAMA7_12_rmv_period.csv
- Bundles:
  pretrained_saplma/instruct/format_{2,3,4}/saplma_checkpoints_LLAMA7/{dataset}_rep9_bundle
- eval script: eval_saplma.py (in the same directory as this file)

Outputs (per dataset):
saplma_tests_results/matrix/<dataset>/matrix.csv
saplma_tests_results/matrix/<dataset>/matrix.md

Global summary:
saplma_tests_results/matrix/summary.csv

Average grid across all datasets:
saplma_tests_results/matrix/average/matrix.csv
saplma_tests_results/matrix/average/matrix.md
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

# -------------------------
# CONFIG — EDIT IF NEEDED
# -------------------------

DATASET_NAMES: List[str] = [
    "animals", "capitals", "companies", "elements", "facts", "inventions"
]

TRAIN_FORMATS = [2, 3, 4]
TEST_FORMATS  = [2, 3, 4]

# Label names for pretty headers
FMT_LABEL = {
    2: "empty",          # format 2
    3: "tell_me_true",   # format 3
    4: "tell_me_false",  # format 4
}

# Path roots (match your tree)
DATASET_ROOT = Path("/home/ddn1/Documents/GitHub/SAPLMA/data/capital_true_false_instruct")
EMB_DIRNAME  = "embeddings_with_labels_data"
CSV_SUFFIX   = "LLAMA7_12_rmv_period.csv"   # same for all datasets in your screenshot

BUNDLE_ROOT  = Path("pretrained_saplma/instruct")
BUNDLE_MID   = "saplma_checkpoints_LLAMA7"
BUNDLE_SUFFIX = "rep9_bundle"               # change if you use another replica

OUT_ROOT = Path("saplma_tests_results/matrix")

# eval script (assume same directory)
EVAL_SCRIPT = Path(__file__).with_name("eval_saplma.py")

# -------------------------
# Helpers
# -------------------------

def dataset_csv_for(dataset: str, test_fmt: int) -> Path:
    # /.../format{fmt}/embeddings_with_labels_data/{dataset}LLAMA7_12_rmv_period.csv
    return DATASET_ROOT / f"format{test_fmt}" / EMB_DIRNAME / f"{dataset}{CSV_SUFFIX}"

def bundle_for(dataset: str, train_fmt: int) -> Path:
    # pretrained_saplma/instruct/format_{fmt}/saplma_checkpoints_LLAMA7/{dataset}_rep9_bundle
    return BUNDLE_ROOT / f"format_{train_fmt}" / BUNDLE_MID / f"{dataset}_{BUNDLE_SUFFIX}"

def out_dir_for(dataset: str, train_fmt: int, test_fmt: int) -> Path:
    return OUT_ROOT / dataset / f"train_fmt{train_fmt}_test_fmt{test_fmt}"

def run_eval(bundle: Path, csv_path: Path, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--bundle", str(bundle),
        "--csv", str(csv_path),
        "--label-col", "label",
        "--out-dir", str(out_dir),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] eval failed for bundle={bundle} csv={csv_path}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return False

def read_acc_at_05(out_dir: Path) -> float:
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    if "accuracy_at_0p5" not in metrics:
        raise RuntimeError(f"No 'accuracy_at_0p5' in {metrics_path}")
    return float(metrics["accuracy_at_0p5"])

def pct(p: float) -> str:
    return f"{p * 100:.2f}"

def _compute_widths(col_headers: List[str], rows: List[List[str]]) -> List[int]:
    widths = [len(h) for h in col_headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)
    return widths

def _sep_line(widths: List[int]) -> None:
    total = sum(widths) + 3 * (len(widths) - 1)
    print("-" * total)

def draw_ascii_table(title: str, left_header: str, col_keys: List[int], row_keys: List[int],
                     cell_text: Dict[Tuple[int,int], str]) -> None:
    col_headers = [left_header] + [FMT_LABEL[c] for c in col_keys]
    rows = []
    for r in row_keys:
        rows.append([FMT_LABEL[r]] + [cell_text[(r, c)] for c in col_keys])

    widths = _compute_widths(col_headers, rows)

    print(f"\n{title}\n")
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(col_headers)))
    _sep_line(widths)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
        _sep_line(widths)

def write_markdown_table(md_path: Path, left_header: str,
                         col_keys: List[int], row_keys: List[int],
                         cell_text: Dict[Tuple[int,int], str]) -> None:
    headers = [left_header] + [FMT_LABEL[c] for c in col_keys]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in row_keys:
        row = [FMT_LABEL[r]] + [cell_text[(r, c)] for c in col_keys]
        lines.append("| " + " | ".join(row) + " |")

    md_path.write_text("\n".join(lines))

def write_matrix_csv(csv_path: Path, col_keys: List[int], row_keys: List[int],
                     cell_vals: Dict[Tuple[int,int], float]) -> None:
    # rows = test_fmt; columns = train_fmt; values = acc@0.5 (0..1)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test\\train"] + [FMT_LABEL[c] for c in col_keys])
        for r in row_keys:
            w.writerow([FMT_LABEL[r]] + [f"{cell_vals.get((r,c), float('nan')):.6f}" for c in col_keys])

# -------------------------
# Main
# -------------------------

def main():
    if not EVAL_SCRIPT.exists():
        raise SystemExit(f"eval_saplma.py not found at: {EVAL_SCRIPT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect a long-form summary for all datasets and an aggregator for averages
    summary_rows = []  # dataset, test_fmt, train_fmt, acc_0p5
    agg = defaultdict(list)  # (test_fmt, train_fmt) -> list[acc]

    for dataset in DATASET_NAMES:
        # Prepare matrix containers (text for print/md, numeric for csv)
        cell_text: Dict[Tuple[int,int], str] = {}
        cell_vals: Dict[Tuple[int,int], float] = {}

        # Sanity: check CSVs exist for all test formats
        for tf in TEST_FORMATS:
            csv_path = dataset_csv_for(dataset, tf)
            if not csv_path.exists():
                raise SystemExit(f"[{dataset}] CSV not found for test format {tf}: {csv_path}")

        # Run all pairs
        for test_fmt in TEST_FORMATS:
            csv_path = dataset_csv_for(dataset, test_fmt)
            for train_fmt in TRAIN_FORMATS:
                bundle = bundle_for(dataset, train_fmt)
                if not bundle.exists():
                    print(f"[WARN] Bundle missing for {dataset}, train_fmt {train_fmt}: {bundle}")

                out_dir = out_dir_for(dataset, train_fmt, test_fmt)
                ok = run_eval(bundle, csv_path, out_dir)
                if not ok:
                    cell_text[(test_fmt, train_fmt)] = "acc@0.5: N/A"
                    continue

                try:
                    acc = read_acc_at_05(out_dir)
                    cell_vals[(test_fmt, train_fmt)] = acc
                    cell_text[(test_fmt, train_fmt)] = f"acc@0.5: {pct(acc)}"
                    summary_rows.append([dataset, test_fmt, train_fmt, f"{acc:.6f}"])
                    agg[(test_fmt, train_fmt)].append(acc)  # accumulate for averages
                except Exception as e:
                    print(f"[WARN] {e}")
                    cell_text[(test_fmt, train_fmt)] = "acc@0.5: N/A"

        # Print table for this dataset
        draw_ascii_table(
            title=f"{dataset.capitalize()}:",
            left_header=f"test({dataset}) \\ train({dataset})",
            col_keys=TRAIN_FORMATS,
            row_keys=TEST_FORMATS,
            cell_text=cell_text
        )

        # Save markdown + csv matrix
        ds_out_dir = OUT_ROOT / dataset
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        write_markdown_table(
            ds_out_dir / "matrix.md",
            left_header=f"test({dataset}) \\ train({dataset})",
            col_keys=TRAIN_FORMATS,
            row_keys=TEST_FORMATS,
            cell_text=cell_text
        )
        write_matrix_csv(
            ds_out_dir / "matrix.csv",
            col_keys=TRAIN_FORMATS,
            row_keys=TEST_FORMATS,
            cell_vals=cell_vals
        )

    # Global summary CSV
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "test_format", "train_format", "accuracy_at_0p5"])
        w.writerows(summary_rows)

    # ---------- Average grid across datasets ----------
    avg_cell_vals: Dict[Tuple[int,int], float] = {}
    avg_cell_text: Dict[Tuple[int,int], str] = {}
    for r in TEST_FORMATS:
        for c in TRAIN_FORMATS:
            vals = agg.get((r, c), [])
            if len(vals) == 0:
                avg_cell_text[(r,c)] = "acc@0.5: N/A"
                continue
            avg = sum(vals) / len(vals)
            avg_cell_vals[(r,c)] = avg
            avg_cell_text[(r,c)] = f"acc@0.5: {pct(avg)}"

    # Print average grid
    draw_ascii_table(
        title="Average across datasets:",
        left_header="test(avg) \\ train(avg)",
        col_keys=TRAIN_FORMATS,
        row_keys=TEST_FORMATS,
        cell_text=avg_cell_text
    )

    # Save average grid (md + csv)
    avg_out_dir = OUT_ROOT / "average"
    avg_out_dir.mkdir(parents=True, exist_ok=True)

    write_markdown_table(
        avg_out_dir / "matrix.md",
        left_header="test(avg) \\ train(avg)",
        col_keys=TRAIN_FORMATS,
        row_keys=TEST_FORMATS,
        cell_text=avg_cell_text
    )
    write_matrix_csv(
        avg_out_dir / "matrix.csv",
        col_keys=TRAIN_FORMATS,
        row_keys=TEST_FORMATS,
        cell_vals=avg_cell_vals
    )

    print(f"\n[OK] Wrote per-dataset matrices under: {OUT_ROOT}/<dataset>/")
    print(f"[OK] Global summary: {OUT_ROOT/'summary.csv'}")
    print(f"[OK] Average grid:   {avg_out_dir/'matrix.md'} and {avg_out_dir/'matrix.csv'}")
    print("\nLegend: empty = format 2, tell_me_true = format 3, tell_me_false = format 4")

if __name__ == "__main__":
    main()
