"""Generate Table 2 (phase-gap diagnostic) for the GCRR-GC paper.

Per Chen Section 9.5: "Table 2 should report the fitted phase-wise
constants A_0 and A_1, the empirical phase gap A_1 - A_0, the
closed-form prediction (1/2) S_0 phi(d_1) sigma sqrt(T), and show
that the even and odd limits are stable but not generally equal."

Inputs:  data/wand_results.csv (from wand_robustness_data.py).
Outputs:
    tables/table_2_phase_gap.csv  -- CSV for import to Word via Excel
    tables/table_2_phase_gap.md   -- Markdown for readability

Columns:
    Case (K, sigma, T) | A_0_hat | A_1_hat | empirical gap | predicted gap | |residual|

Fitting method: large-n mean of scaled errors (n_eff >= 1000). Verified
in scripts/verify_phase_gap_wand.py to match the Corollary 7.2
prediction within 2e-4 across all 8 WAND cases.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


LARGE_N_THRESHOLD = 1000
INPUT_CSV = Path(__file__).parent.parent / "data" / "wand_results.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "tables"


def phase_gap_prediction(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Closed-form A_1 - A_0 from Corollary 7.2."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return 0.5 * S * norm.pdf(d1) * sigma * np.sqrt(T)


def build_table_rows(df: pd.DataFrame) -> list[dict]:
    """For each WAND case, compute fitted constants and predicted gap.

    Returns a list of dicts, one per case, with columns matching the
    final table output.
    """
    gc = df[df["scheme"] == "gcrr_gc"]

    rows = []
    for case_id in sorted(gc["case_id"].unique()):
        case = gc[gc["case_id"] == case_id]
        S = case["S"].iloc[0]
        K = case["K"].iloc[0]
        sigma = case["sigma"].iloc[0]
        T = case["T"].iloc[0]
        r = case["r"].iloc[0]

        large_n = case[case["n_eff"] >= LARGE_N_THRESHOLD]
        A_0 = large_n[large_n["eta"] == 0]["scaled_error"].mean()
        A_1 = large_n[large_n["eta"] == 1]["scaled_error"].mean()
        emp_gap = A_1 - A_0

        prediction = phase_gap_prediction(S, K, T, r, sigma)
        residual = abs(emp_gap - prediction)

        rows.append({
            "K": K,
            "sigma": sigma,
            "T": T,
            "A_0_hat": A_0,
            "A_1_hat": A_1,
            "emp_gap": emp_gap,
            "predicted": prediction,
            "abs_residual": residual,
        })
    return rows


def write_csv(rows: list[dict], path: Path):
    """Write table as CSV (importable to Word via Excel paste)."""
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, float_format="%.6g")
    print(f"Saved {path}")


def write_markdown(rows: list[dict], path: Path):
    """Write table as Markdown for human review."""
    header = (
        "| Case (K, σ, T)        | $\\hat{A}_0$ | $\\hat{A}_1$ | "
        "$\\hat{A}_1 - \\hat{A}_0$ | Predicted | |Residual| |\n"
        "|-----------------------|-------------|-------------|"
        "--------------------------|-----------|------------|\n"
    )
    body_lines = []
    for r in rows:
        case_str = f"(K={r['K']:.0f}, σ={r['sigma']:.2f}, T={r['T']:.2f})"
        body_lines.append(
            f"| {case_str:21s} | "
            f"{r['A_0_hat']:+.4f}    | "
            f"{r['A_1_hat']:+.4f}    | "
            f"{r['emp_gap']:+.4f}                 | "
            f"{r['predicted']:+.4f}  | "
            f"{r['abs_residual']:.2e}  |"
        )
    table_md = header + "\n".join(body_lines) + "\n"

    # Include a brief caption for readability
    caption = (
        "**Table 2.** Empirical phase gap $\\hat{A}_1 - \\hat{A}_0$ versus the "
        "Corollary 7.2 prediction $\\frac{1}{2} S_0 \\phi(d_1) \\sigma \\sqrt{T}$, "
        "across the WAND robustness panel ($S_0=40$, $r=0.06$). Fits use the "
        "mean of $n(C_n - C_{BS})$ over $n_{eff} \\geq 1000$ for GCRR-GC.\n\n"
    )

    with open(path, "w") as f:
        f.write(caption)
        f.write(table_md)
    print(f"Saved {path}")


def main():
    df = pd.read_csv(INPUT_CSV)
    rows = build_table_rows(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(rows, OUTPUT_DIR / "table_2_phase_gap.csv")
    write_markdown(rows, OUTPUT_DIR / "table_2_phase_gap.md")

    print(f"\nGenerated Table 2 with {len(rows)} cases.")
    max_residual = max(r["abs_residual"] for r in rows)
    print(f"Maximum residual: {max_residual:.2e}")


if __name__ == "__main__":
    main()
