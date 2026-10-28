"""Generate Table OA.1: RMS pricing errors across the WAND panel.

This script produces a Chung-Shih (2007) Table 3-style robustness table:

    RMS errors averaged over the 8 WAND parameter combinations at
    each selected n, per scheme. European calls only.

Rows: n ∈ {50, 100, 500, 1000, 5000} benchmark values.
Columns per scheme: RMS error over the 8 WAND cases.
Schemes: CRR, GCRR-GC, Tian (1999), Joshi (2009).

WAND panel per Section 5.1:
    S = 40, r = 0.06, K ∈ {35, 45}, sigma ∈ {0.20, 0.40}, T ∈ {0.25, 0.50}.

Joshi handling: Joshi's construction requires odd n. RMS values for
Joshi at each row are computed from Joshi's prices at the adjacent
odd n_eff = n + 1. Footnoted in the output.

Inputs: data/wand_results.csv (from wand_robustness_data.py).

Outputs (in tables/):
    table_1_appendix_rms.csv
    table_1_appendix_rms.md
    table_1_appendix_rms.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd


SELECTED_N = [50, 100, 500, 1000, 5000]
SCHEMES_ORDER = ["crr", "gcrr_gc", "tian_1999", "joshi_2009"]
SCHEME_DISPLAY = {
    "crr":        "CRR",
    "gcrr_gc":    "GCRR-GC",
    "tian_1999":  "Tian (1999)",
    "joshi_2009": "Joshi (2009)",
}

INPUT_CSV = Path(__file__).parent.parent / "data" / "wand_results.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "tables"


def compute_rms_row(df, scheme, target_n):
    """Compute RMS of errors across the 8 WAND cases at (scheme, target_n).

    For Joshi, look up prices at n_eff = target_n + 1 (adjacent odd n).
    """
    if scheme == "joshi_2009":
        lookup_n = target_n + 1
    else:
        lookup_n = target_n

    subset = df[(df["scheme"] == scheme) & (df["n_eff"] == lookup_n)]
    if subset.empty:
        return None
    if len(subset) != 8:
        # Sanity: should be 8 cases per (scheme, n_eff)
        print(f"  WARNING: {scheme} at n_eff={lookup_n}: found {len(subset)} rows, expected 8")

    errors = subset["error"].values
    rms = float(np.sqrt(np.mean(errors ** 2)))
    return rms


def build_table(df):
    """Build the appendix table as a list of row dicts."""
    rows = []
    for target_n in SELECTED_N:
        row = {"n": target_n}
        for scheme in SCHEMES_ORDER:
            rms = compute_rms_row(df, scheme, target_n)
            row[scheme] = rms
        rows.append(row)
    return rows


def write_outputs(rows, out_dir):
    """Write CSV, MD, XLSX outputs."""
    out_dir.mkdir(exist_ok=True)

    # Build a display-ready DataFrame
    display_rows = []
    for r in rows:
        display_row = {"n": int(r["n"])}
        for scheme in SCHEMES_ORDER:
            display_row[SCHEME_DISPLAY[scheme]] = r[scheme]
        display_rows.append(display_row)
    df = pd.DataFrame(display_rows)

    csv_path = out_dir / "table_1_appendix_rms.csv"
    df.to_csv(csv_path, index=False, float_format="%.4e")
    print(f"Saved {csv_path}")

    xlsx_path = out_dir / "table_1_appendix_rms.xlsx"
    df.to_excel(xlsx_path, index=False, float_format="%.4e")
    print(f"Saved {xlsx_path}")

    md_path = out_dir / "table_1_appendix_rms.md"
    _write_markdown(df, md_path)
    print(f"Saved {md_path}")


def _write_markdown(df, path):
    """Markdown output with caption and footnote."""
    lines = []
    lines.append("# Appendix Table: RMS Pricing Errors across the WAND Panel")
    lines.append("")
    lines.append("Approximation RMS error for European calls, averaged over the 8 WAND-panel parameter combinations:")
    lines.append("$K \\in \\{35, 45\\}$, $\\sigma \\in \\{0.20, 0.40\\}$, $T \\in \\{0.25, 0.50\\}$")
    lines.append("(with $S_0 = 40$, $r = 0.06$).")
    lines.append("")

    header_cells = ["n"] + [SCHEME_DISPLAY[s] for s in SCHEMES_ORDER]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

    for _, row in df.iterrows():
        cells = [str(int(row["n"]))]
        for scheme in SCHEMES_ORDER:
            val = row[SCHEME_DISPLAY[scheme]]
            cells.append(f"{val:.4e}" if pd.notna(val) else "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("*Note: Joshi (2009) values are computed from Joshi prices at the adjacent odd n (n + 1) since the construction requires odd n.*")
    lines.append("")

    path.write_text("\n".join(lines))


def main():
    print("Building appendix Table 1 (RMS across WAND panel)...")

    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} rows from {INPUT_CSV.name}")

    rows = build_table(df)
    write_outputs(rows, OUTPUT_DIR)

    print(f"\nGenerated appendix Table: {len(rows)} rows (n = {SELECTED_N}).")


if __name__ == "__main__":
    main()
