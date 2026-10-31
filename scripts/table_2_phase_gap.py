"""Table 2: phase-wise constants and phase gap across the WAND panel.

Tests Theorem 3.2 and Corollary 3.3 at the coefficient level. For each of
the eight WAND-style cases (Section 5.1), the phase constants A_0 and A_1
are estimated by least squares,

    n (C_n - C_BS) = A + B n^{-1/2} + C n^{-1},   n = base + eta,
    base in FIT_N_BASES = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5),

with GCRR-GC prices from the exact two-tail representation (Proposition
3.1), which agrees with backward induction to 1e-10 (tests/test_pricer.py).
Fitted values are compared with eq. (8) for A_0, A_1 and eq. (9) for the
gap. Residuals are of order 1e-7 to 1e-6, depending on platform floating
point.

Outputs (tables/):
    table_2_phase_gap.csv, .xlsx   full precision
    table_2_phase_gap.md           rounded, with caption
"""

from itertools import product
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

from gcrr.errors import fit_leading_constant
from gcrr.high_precision import bs_call_mp, gcrr_gc_call_mp
from gcrr.theory import A_eta, phase_gap

mp.mp.dps = 40  # fit uses n up to 1e5, where n amplifies double-precision rounding

# WAND-style panel (Section 5.1)
S0, RATE = 40.0, 0.06
K_VALUES = (35.0, 45.0)
SIGMA_VALUES = (0.20, 0.40)
T_VALUES = (0.25, 0.50)

# Step counts for fitting A_eta; n = base + eta in each phase
FIT_N_BASES = (1000, 2000, 5000, 10000, 20000, 50000, 100000)

OUTPUT_DIR = Path(__file__).parent.parent / "tables"


def fit_phase_constant(S, K, T, r, sigma, eta: int) -> float:
    """Fitted A_eta: least-squares fit of n (C_n - C_BS) = A + B n^{-1/2} + C n^{-1}
    over n = base + eta, base in FIT_N_BASES. Prices are 40-digit GCRR-GC
    prices from the two-tail representation (Proposition 3.1), so the fit
    does not depend on the machine's floating-point libraries.
    """
    n_values = np.array([base + eta for base in FIT_N_BASES], dtype=int)

    C_BS = bs_call_mp(S, K, T, r, sigma)
    scaled_errors = []
    for n in n_values:
        n = int(n)
        C_n = gcrr_gc_call_mp(S, K, T, r, sigma, n)
        scaled_errors.append(float(n * (C_n - C_BS)))

    return fit_leading_constant(n_values, scaled_errors)


def build_table_rows() -> list[dict]:
    """For each WAND case, compute fitted constants and predicted gap.

    Returns a list of dicts, one per case, with columns matching the
    final table output.
    """
    rows = []

    for K, sigma, T in product(K_VALUES, SIGMA_VALUES, T_VALUES):
        # Empirical fitted coefficients
        A0_hat = fit_phase_constant(S0, K, T, RATE, sigma, 0)
        A1_hat = fit_phase_constant(S0, K, T, RATE, sigma, 1)

        # Theoretical predictions from Theorem 3.2
        A0_theory = A_eta(S0, K, T, RATE, sigma, 0)
        A1_theory = A_eta(S0, K, T, RATE, sigma, 1)

        # Empirical and theoretical phase gaps
        gap_hat = A1_hat - A0_hat
        gap_theory = phase_gap(S0, K, T, RATE, sigma)

        # Largest discrepancy among the three theoretical comparisons
        max_abs_resid = max(
            abs(A0_hat - A0_theory), abs(A1_hat - A1_theory), abs(gap_hat - gap_theory)
        )

        rows.append(
            {
                "K": K,
                "sigma": sigma,
                "T": T,
                "A0_hat": A0_hat,
                "A0_theory": A0_theory,
                "A1_hat": A1_hat,
                "A1_theory": A1_theory,
                "gap_hat": gap_hat,
                "gap_theory": gap_theory,
                "max_abs_resid": max_abs_resid,
            }
        )

    return rows


def write_csv(rows: list[dict], path: Path):
    """Write table as CSV at full precision."""
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, float_format="%.10g")
    print(f"Saved {path}")


def write_xlsx(rows: list[dict], path: Path):
    """Write table as XLSX at full precision (for pasting into Word)."""
    table = pd.DataFrame(rows)
    table.to_excel(path, index=False, float_format="%.10g")
    print(f"Saved {path}")


def write_markdown(rows: list[dict], path: Path):
    """Write table as Markdown with caption, rounded for reading."""
    caption = (
        "**Table 2.** Phase-wise constants and phase gap for GCRR-GC across "
        "the WAND panel ($S_0=40$, $r=0.06$). $\\hat{A}_\\eta$ are least-squares "
        "fits of $n(C_n - C_{BS}) = A + Bn^{-1/2} + Cn^{-1}$ over "
        "$n \\in [10^3, 10^5]$ in each phase, using 40-digit prices; predictions are from eqs. (8) "
        "and (9). Max resid. is the largest of the three absolute differences.\n\n"
    )

    header = (
        "| Case (K, σ, T) | $\\hat{A}_0$ | $A_0$ (8) | $\\hat{A}_1$ | $A_1$ (8) "
        "| $\\hat{A}_1 - \\hat{A}_0$ | $A_1 - A_0$ (9) | Max resid. |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )

    body_lines = []
    for r in rows:
        case_str = f"(K={r['K']:.0f}, σ={r['sigma']:.2f}, T={r['T']:.2f})"
        body_lines.append(
            f"| {case_str} "
            f"| {r['A0_hat']:+.4f} | {r['A0_theory']:+.4f} "
            f"| {r['A1_hat']:+.4f} | {r['A1_theory']:+.4f} "
            f"| {r['gap_hat']:+.4f} | {r['gap_theory']:+.4f} "
            f"| {r['max_abs_resid']:.1e} |"
        )

    with open(path, "w") as f:
        f.write(caption)
        f.write(header)
        f.write("\n".join(body_lines) + "\n")
    print(f"Saved {path}")


def main():
    rows = build_table_rows()

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(rows, OUTPUT_DIR / "table_2_phase_gap.csv")
    write_xlsx(rows, OUTPUT_DIR / "table_2_phase_gap.xlsx")
    write_markdown(rows, OUTPUT_DIR / "table_2_phase_gap.md")

    print(f"\nGenerated Table 2 with {len(rows)} cases.")
    max_residual = max(r["max_abs_resid"] for r in rows)
    print(f"Maximum residual: {max_residual:.2e}")


if __name__ == "__main__":
    main()
