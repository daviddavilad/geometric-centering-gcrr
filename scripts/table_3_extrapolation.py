"""Table 3: Phase-aware extrapolation and phase alignment (Section 3.3).

Compares five signed pricing errors for GCRR-GC European calls:

    raw_even     C_n - C_BS, with even n
    raw_odd      C_{n+1} - C_BS
    same_phase   ((n+2)*C_{n+2} - n*C_n)/2 - C_BS
    adjacent     (n+1)*C_{n+1} - n*C_n - C_BS
    adjacent_pa  Adjacent Richardson after phase alignment

The phase adjustment is
    C_n^PA = C_n - (n % 2)*gap/n,
where gap = A_1 - A_0 from Corollary 3.3.

All tree parameters, binomial tails, prices, and extrapolation
calculations use 40-digit mpmath arithmetic. Richardson
extrapolation amplifies rounding errors by approximately n,
making ordinary double precision unreliable at large n.

Panels:
    A: Out-of-the-money, K=105
    B: In-the-money, K=95

Outputs:
    tables/table_3_extrapolation.csv
    tables/table_3_extrapolation.xlsx
    tables/table_3_extrapolation.md
"""

from pathlib import Path

import mpmath as mp
import pandas as pd

from gcrr.extrapolation import (
    phase_adjust,
    richardson_adjacent,
    richardson_same_phase,
)
from gcrr.high_precision import bs_call_mp, gcrr_gc_call_mp
from gcrr.theory import phase_gap

# Keep precision active for all calculations, not only pricing.
mp.mp.dps = 40

S, T, RATE, SIGMA = 100.0, 0.5, 0.06, 0.20

PANELS = (("A", 105.0), ("B", 95.0))

N_GRID = (50, 100, 500, 1000, 5000)

OUTPUT_DIR = Path(__file__).parent.parent / "tables"

COLUMNS = (
    "panel",
    "K",
    "n",
    "raw_even",
    "raw_odd",
    "same_phase",
    "adjacent",
    "adjacent_pa",
)


def build_table_rows() -> list[dict]:
    """Compute five signed extrapolation errors per panel and n."""
    rows = []
    assert all(n % 2 == 0 for n in N_GRID), f"N_GRID must contain only even n: {N_GRID}"

    for panel, K in PANELS:
        # Reference value and theoretical phase gap.
        C_BS = bs_call_mp(S, K, T, RATE, SIGMA)
        gap = mp.mpf(phase_gap(S, K, T, RATE, SIGMA))

        for n in N_GRID:
            # Compute each required tree price only once.
            C_n = gcrr_gc_call_mp(S, K, T, RATE, SIGMA, n)
            C_n1 = gcrr_gc_call_mp(S, K, T, RATE, SIGMA, n + 1)
            C_n2 = gcrr_gc_call_mp(S, K, T, RATE, SIGMA, n + 2)

            # Raw pricing errors.
            raw_even = C_n - C_BS
            raw_odd = C_n1 - C_BS

            # Same-phase Richardson: n and n+2.
            C_same = richardson_same_phase(C_n, C_n2, n)
            same_phase = C_same - C_BS

            # Ordinary adjacent Richardson: n and n+1.
            C_adj = richardson_adjacent(C_n, C_n1, n)
            adjacent = C_adj - C_BS

            # Phase-align prices before adjacent extrapolation.
            C_n_pa = phase_adjust(C_n, n, gap)
            C_n1_pa = phase_adjust(C_n1, n + 1, gap)

            C_adj_pa = richardson_adjacent(C_n_pa, C_n1_pa, n)
            adjacent_pa = C_adj_pa - C_BS

            # Convert to float only for the final output.
            rows.append(
                {
                    "panel": panel,
                    "K": int(K),
                    "n": n,
                    "raw_even": float(raw_even),
                    "raw_odd": float(raw_odd),
                    "same_phase": float(same_phase),
                    "adjacent": float(adjacent),
                    "adjacent_pa": float(adjacent_pa),
                }
            )

    return rows


def write_csv(rows: list[dict], path: Path):
    """Write numerical results as CSV with seven significant digits."""
    table = pd.DataFrame(rows, columns=COLUMNS)
    table.to_csv(path, index=False, float_format="%.6e")
    print(f"Saved {path}")


def write_xlsx(rows: list[dict], path: Path):
    """Write numerical results as an Excel workbook."""
    table = pd.DataFrame(rows, columns=COLUMNS)
    table.to_excel(path, index=False, float_format="%.6e")
    print(f"Saved {path}")


def write_markdown(rows: list[dict], path: Path):
    """Write a readable table with separate OTM and ITM panels."""

    caption = (
        "**Table 3.** Extrapolation errors for GCRR-GC European calls "
        r"($S_0=100$, $T=0.5$, $r=0.06$, $\sigma=0.20$). "
        r"Raw errors are $C_n - C_{BS}$ at the even count $n$ and "
        r"$C_{n+1} - C_{BS}$ at the adjacent odd count. "
        r"Same-phase Richardson uses "
        r"$[(n+2)C_{n+2}-nC_n]/2$, while adjacent Richardson uses "
        r"$(n+1)C_{n+1}-nC_n$. "
        r"The phase-adjusted column applies adjacent Richardson "
        r"after $C_n^{PA}=C_n-\eta\Delta A/n$, where "
        r"$\Delta A=A_1-A_0$ is given by Corollary 3.3. "
        "All prices and extrapolations are evaluated using "
        "40-digit arithmetic.\n\n"
    )

    sections = [caption]

    for panel, K in PANELS:
        description = "Out-of-the-Money" if K > S else "In-the-Money"

        sections.append(f"### Panel {panel}: {description} (K={K:.0f})\n\n")

        sections.append(
            "| n | Raw even | Raw odd | Same phase | "
            "Adjacent | Adjacent PA |\n"
            "|---:|---:|---:|---:|---:|---:|\n"
        )

        for row in rows:
            if row["panel"] != panel:
                continue

            sections.append(
                f"| {row['n']} "
                f"| {row['raw_even']:+.3e} "
                f"| {row['raw_odd']:+.3e} "
                f"| {row['same_phase']:+.3e} "
                f"| {row['adjacent']:+.3e} "
                f"| {row['adjacent_pa']:+.3e} |\n"
            )

        sections.append("\n")

    path.write_text("".join(sections), encoding="utf-8")
    print(f"Saved {path}")


def main():
    rows = build_table_rows()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(rows, OUTPUT_DIR / "table_3_extrapolation.csv")
    write_xlsx(rows, OUTPUT_DIR / "table_3_extrapolation.xlsx")
    write_markdown(rows, OUTPUT_DIR / "table_3_extrapolation.md")

    print(f"\nGenerated Table 3 with {len(rows)} rows.")

    for panel, K in PANELS:
        last = next(
            row for row in rows if row["panel"] == panel and row["n"] == N_GRID[-1]
        )
        print(
            f"Panel {panel} (K={K:.0f}): same-phase error at n={N_GRID[-1]} = {last['same_phase']:+.4e}"
        )


if __name__ == "__main__":
    main()
