"""Generate main-text Table 1: convergence for K = 105 and K = 95.

Table 1 reports prices and signed errors across schemes. This script
produces a Chung-Shih (2007) Table 2-style two-panel table:

    Upper panel: K = 105 (OTM), the primary non-ATM case in Section 5.1
    Lower panel: K =  95 (ITM), the symmetric-moneyness robustness case

Rows per panel: n ∈ {50, 100, 500, 1000, 5000}, matching the benchmark
n values used in WAND (2002) and Chung-Shih (2007). Each panel ends
with a Black-Scholes reference row.

Columns per scheme: price (C_n) and error (C_n - C_BS). Four schemes
reported: CRR, GCRR-GC, Tian (1999), Joshi (2009).

Joshi handling: Joshi's construction requires odd n. At each requested
even n we report Joshi's price at the adjacent odd n_eff = n + 1.
This is footnoted in the table output.

Fixed parameters (Section 5.1 primary non-ATM case):
    S_0 = 100, T = 0.5, r = 0.06, sigma = 0.20.

Outputs (in tables/):
    table_1_main.csv
    table_1_main.md
    table_1_main.xlsx
"""

from pathlib import Path

import pandas as pd

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.pricer import binomial_price

# --- Fixed parameters (Section 5.1 primary non-ATM cases) ---
S = 100.0
T = 0.5
R = 0.06
SIGMA = 0.20

# --- Two-panel structure ---
K_UPPER = 105.0  # OTM primary
K_LOWER = 95.0  # ITM symmetric-moneyness robustness

# --- Benchmark n values (WAND 2002 / Chung-Shih 2007) ---
SELECTED_N = [50, 100, 500, 1000, 5000]

# --- Scheme registry ---
SCHEMES = {
    "crr": (schemes.crr, "CRR"),
    "gcrr_gc": (schemes.gcrr_gc, "GCRR-GC"),
    "tian_1999": (schemes.tian_1999, "Tian (1999)"),
    "joshi_2009": (schemes.joshi_2009, "Joshi (2009)"),
}

OUTPUT_DIR = Path(__file__).parent.parent / "tables"


def price_and_error(scheme_fn, S, K, T, r, sigma, target_n):
    """Compute price and error for one (scheme, n) cell.

    For Joshi, requests at even n internally map to the adjacent odd n_eff,
    so the returned n_eff can differ from target_n. Returns (n_eff, price,
    error).
    """
    params = scheme_fn(S, K, T, r, sigma, target_n)
    n_eff = params.n
    price = binomial_price(S, K, T, r, params, "call", "european")
    bs = bs_call(S, K, T, r, sigma)
    error = price - bs
    return n_eff, price, error


def build_panel(K):
    """Build one panel (K = specified value) as a list of row dicts."""
    rows = []
    bs = bs_call(S, K, T, R, SIGMA)

    for n_req in SELECTED_N:
        row = {"n": n_req}
        for scheme_key, (scheme_fn, _) in SCHEMES.items():
            n_eff, price, error = price_and_error(scheme_fn, S, K, T, R, SIGMA, n_req)
            row[f"{scheme_key}_price"] = price
            row[f"{scheme_key}_error"] = error
            # Track n_eff for Joshi footnote; other schemes match n_req
            if scheme_key == "joshi_2009":
                row[f"{scheme_key}_n_eff"] = n_eff
        rows.append(row)

    # BS reference row
    bs_row = {"n": "BS"}
    for scheme_key in SCHEMES:
        bs_row[f"{scheme_key}_price"] = bs
        bs_row[f"{scheme_key}_error"] = None
        if scheme_key == "joshi_2009":
            bs_row[f"{scheme_key}_n_eff"] = None
    rows.append(bs_row)

    return rows, bs


def rows_to_dataframe(rows_upper, rows_lower):
    """Combine both panels into a single DataFrame with a panel indicator."""
    all_rows = []
    for r in rows_upper:
        r = {**r, "panel": f"K = {K_UPPER:g}"}
        all_rows.append(r)
    for r in rows_lower:
        r = {**r, "panel": f"K = {K_LOWER:g}"}
        all_rows.append(r)

    # Reorder columns: panel, n, then per-scheme (price, error) pairs
    columns = ["panel", "n"]
    for scheme_key in SCHEMES:
        columns.append(f"{scheme_key}_price")
        columns.append(f"{scheme_key}_error")

    df = pd.DataFrame(all_rows)[columns]
    return df


def write_outputs(df, out_dir):
    """Write CSV, MD, XLSX outputs."""
    out_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = out_dir / "table_1_main.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"Saved {csv_path}")

    # XLSX
    xlsx_path = out_dir / "table_1_main.xlsx"
    df.to_excel(xlsx_path, index=False, float_format="%.6g")
    print(f"Saved {xlsx_path}")

    # Markdown (human-readable)
    md_path = out_dir / "table_1_main.md"
    _write_markdown(df, md_path)
    print(f"Saved {md_path}")


def _write_markdown(df, path):
    """Custom markdown writer with panel headers."""
    lines = []
    lines.append("# Table 1: Convergence of Binomial Schemes at Primary Non-ATM Cases")
    lines.append("")
    lines.append(
        f"Parameters: $S_0 = {S:g}$, $T = {T}$, $r = {R}$, $\\sigma = {SIGMA}$."
    )
    lines.append("")

    for panel_name in df["panel"].unique():
        lines.append(f"## Panel: {panel_name}")
        lines.append("")

        panel_df = df[df["panel"] == panel_name].drop(columns=["panel"])

        # Build header
        header_cells = ["n"]
        for scheme_key, (_, scheme_label) in SCHEMES.items():
            header_cells.append(f"{scheme_label} price")
            header_cells.append(f"{scheme_label} error")

        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

        for _, row in panel_df.iterrows():
            cells = [str(row["n"])]
            for scheme_key in SCHEMES:
                price = row[f"{scheme_key}_price"]
                error = row[f"{scheme_key}_error"]
                cells.append(f"{price:.6f}" if pd.notna(price) else "-")
                cells.append(f"{error:+.4e}" if pd.notna(error) else "-")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    lines.append(
        "*Note: Joshi (2009) prices are reported at the adjacent odd n (n + 1) since the construction requires odd n.*"
    )
    lines.append("")

    path.write_text("\n".join(lines))


def main():
    print("Building main-text Table 1...")
    print(f"  Parameters: S = {S}, T = {T}, r = {R}, sigma = {SIGMA}")
    print(f"  n grid: {SELECTED_N}")
    print(f"  Upper panel: K = {K_UPPER}")
    print(f"  Lower panel: K = {K_LOWER}")

    rows_upper, _ = build_panel(K_UPPER)
    rows_lower, _ = build_panel(K_LOWER)
    df = rows_to_dataframe(rows_upper, rows_lower)

    write_outputs(df, OUTPUT_DIR)

    print(
        f"\nGenerated Table 1 (main-text): {len(df)} rows (2 panels of {len(SELECTED_N) + 1} rows each)."
    )


if __name__ == "__main__":
    main()
