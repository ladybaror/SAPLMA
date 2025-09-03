#!/usr/bin/env python3
"""
Run eval_saplma.py over a 3x3 grid (test formats 2/3/4 vs train formats 2/3/4)
for the Animals dataset and print a table with acc@0.5 values.

Adjust the paths in BUNDLES and DATASET_ROOT/CSV_NAME to your layout.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

# -------------------------
# CONFIG — EDIT THESE PATHS
# -------------------------

# Where your pretrained bundles live (one per training format).
# Replace animals_rep9_bundle with your actual bundle names if different.
BUNDLES: Dict[int, str] = {
    2: "pretrained_saplma/instruct/format_2/saplma_checkpoints_LLAMA7/animals_rep9_bundle",
    3: "pretrained_saplma/instruct/format_3/saplma_checkpoints_LLAMA7/animals_rep9_bundle",
    4: "pretrained_saplma/instruct/format_4/saplma_checkpoints_LLAMA7/animals_rep9_bundle",
}

# Root folder that contains format{2,3,4}/... for the Animals CSVs
DATASET_ROOT = Path("/home/ddn1/Documents/GitHub/SAPLMA/data/capital_true_false_instruct")

# The CSV file name under each format*/embeddings_with_labels_data/
CSV_NAME = "embeddings_with_labels_data/animalsLLAMA7_12_rmv_period.csv"

# Where to drop all eval outputs for this grid
OUT_ROOT = Path("saplma_tests_results/animals_grid")

# eval_saplma.py path (assumes this script is next to it)
EVAL_SCRIPT = Path(__file__).with_name("eval_saplma.py")

# Column/row labels
FMT_LABEL = {
    2: "empty",          # format 2
    3: "tell_me_true",   # format 3
    4: "tell_me_false",  # format 4
}

# -------------------------
# Helpers
# -------------------------

def dataset_csv_for_format(fmt: int) -> Path:
    return DATASET_ROOT / f"format{fmt}" / CSV_NAME

def out_dir_for_pair(train_fmt: int, test_fmt: int) -> Path:
    # e.g., saplma_tests_results/animals_grid/train_fmt4_test_fmt2/
    return OUT_ROOT / f"train_fmt{train_fmt}_test_fmt{test_fmt}"

def run_eval(bundle: str, csv_path: Path, out_dir: Path) -> Tuple[bool, str]:
    """Runs eval_saplma.py. Returns (ok, message)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--bundle", bundle,
        "--csv", str(csv_path),
        "--label-col", "label",
        "--out-dir", str(out_dir),
    ]
    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, cp.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, f"[ERR] {e}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"

def read_acc_at_05(out_dir: Path) -> float:
    """Load accuracy_at_0p5 from metrics.json. Returns float in [0,1]."""
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    if "accuracy_at_0p5" not in metrics:
        raise RuntimeError(f"No 'accuracy_at_0p5' in {metrics_path}")
    return float(metrics["accuracy_at_0p5"])

def format_percent(p: float) -> str:
    return f"{p * 100:.2f}"

def draw_ascii_table(cell_text: Dict[Tuple[int,int], str]) -> None:
    """
    cell_text[(row_fmt, col_fmt)] -> e.g. 'acc@0.5: 77.28'
    Rows/Cols are 2,3,4 with labels in FMT_LABEL.
    """
    col_keys = [2,3,4]
    row_keys = [2,3,4]
    header0 = "test(animals) \\ train(All but ani.)"

    # Compute column widths
    col_headers = [header0] + [FMT_LABEL[c] for c in col_keys]
    rows = []
    for r in row_keys:
        row = [FMT_LABEL[r]] + [cell_text[(r, c)] for c in col_keys]
        rows.append(row)

    widths = [len(h) for h in col_headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def sep_line():
        total = sum(widths) + 3 * (len(widths) - 1)
        print("-" * total)

    # Print
    print("Animals:\n")
    # header
    header = " | ".join(h.ljust(widths[i]) for i, h in enumerate(col_headers))
    print(header)
    sep_line()
    for row in rows:
        line = " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))
        print(line)
        sep_line()

def print_markdown_table(cell_text: Dict[Tuple[int,int], str]) -> None:
    col_keys = [2,3,4]
    row_keys = [2,3,4]
    headers = ["test(animals) \\ train(All but ani.)"] + [FMT_LABEL[c] for c in col_keys]
    print("\nMarkdown:\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in row_keys:
        row = [FMT_LABEL[r]] + [cell_text[(r, c)] for c in col_keys]
        print("| " + " | ".join(row) + " |")

# -------------------------
# Main
# -------------------------

def main():
    # sanity checks
    if not EVAL_SCRIPT.exists():
        raise SystemExit(f"eval_saplma.py not found at: {EVAL_SCRIPT}")
    for fmt, bundle in BUNDLES.items():
        if not Path(bundle).exists():
            print(f"[WARN] Bundle for train format {fmt} does not exist yet: {bundle}")

    # Grid run
    cell_text: Dict[Tuple[int,int], str] = {}
    for test_fmt in (2,3,4):
        csv_path = dataset_csv_for_format(test_fmt)
        if not csv_path.exists():
            raise SystemExit(f"CSV not found for test format {test_fmt}: {csv_path}")

        for train_fmt in (2,3,4):
            bundle = BUNDLES[train_fmt]
            out_dir = out_dir_for_pair(train_fmt, test_fmt)

            ok, msg = run_eval(bundle, csv_path, out_dir)
            if not ok:
                print(msg)
                cell_text[(test_fmt, train_fmt)] = "acc@0.5: N/A"
                continue

            try:
                acc = read_acc_at_05(out_dir)
                cell_text[(test_fmt, train_fmt)] = f"acc@0.5: {format_percent(acc)}"
            except Exception as e:
                print(f"[WARN] {e}")
                cell_text[(test_fmt, train_fmt)] = "acc@0.5: N/A"

    # Output
    draw_ascii_table(cell_text)
    print_markdown_table(cell_text)

if __name__ == "__main__":
    main()
