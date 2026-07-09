"""WAND-style robustness data pipeline for Tables 1 and 2.

Computes binomial pricing errors and scaled errors across the WAND-style
non-ATM parameter panel specified in Chen Section 9.4 (revised draft):

    S = 40, r = 0.06, X in {35, 45}, sigma in {0.20, 0.40}, T in {0.25, 0.50}.

This is 8 parameter combinations (2 strikes x 2 vols x 2 maturities). The
panel is directly comparable to the benchmark panels in Widdicks et al.
(2002) and Chung and Shih (2007), and avoids the ATM collapse by keeping
X/S in {0.875, 1.125}.

For each (case, scheme) pair, the script computes binomial prices and
errors over a log-spaced n grid from 200 to 5000, per Chen Section 9.4's
"200 <= n <= 5000 for asymptotic constants" specification.

Output: data/wand_results.csv in long format. One row per
(case_id, scheme, n_eff). Columns:
    case_id, S, K, T, r, sigma, scheme, n_req, n_eff, eta,
    price, bs_price, error, scaled_error.

Downstream: Tables 1 and 2 will be generated from this CSV. The fitted
phase-wise constants A_0, A_1, the empirical phase gap, and the
closed-form prediction (Corollary 7.2) will be computed in a separate
table-generation script.

Schemes computed:
    CRR (baseline), GCRR-GC (the proposed method), Tian (1999),
    Joshi (2009). GCRR-fixed and GCRR-XPC excluded; they are control
    schemes documented in the figure scripts but not part of the main
    benchmark comparison.

Runtime: approximately 3-5 minutes. Each binomial price at n=5000 is
roughly O(n^2) = 25M operations in NumPy.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.pricer import binomial_price


# --- WAND-style parameter panel (Chen Section 9.4) ---
WAND_S = 40.0
WAND_R = 0.06
WAND_K_VALUES = [35.0, 45.0]
WAND_SIGMA_VALUES = [0.20, 0.40]
WAND_T_VALUES = [0.25, 0.50]


# --- n grid: log-spaced for efficient compute coverage ---
def _build_n_grid() -> np.ndarray:
    """Log-spaced n grid covering both parities.

    Each anchor n yields two grid points (n, n+1) so every scheme has
    both even-phase and odd-phase data. Density: coarse at low n
    (where WAND-standard selected values live), denser at moderate n,
    sparse at high n.

    Range covers n = 50 to 5001, including Chen §9.4's selected values
    {50, 100, 500, 1000, 5000} (WAND (2002) and Chung-Shih (2007)
    benchmark points).

    Joshi's odd-only construction maps consecutive (n, n+1) pairs to
    the same odd n_eff, so its row count after dedup is half the others'.
    """
    very_low = np.array([50, 100, 150])                 # 3 anchors: covers n=50, 100
    low = np.arange(200, 500, 50)                       # 6 anchors: 200, 250, ..., 450
    mid = np.arange(500, 1000, 100)                     # 5 anchors: 500, 600, ..., 900
    high = np.arange(1000, 5001, 200)                   # 21 anchors: 1000, 1200, ..., 5000

    anchors = np.concatenate([very_low, low, mid, high])  # 35 even anchors

    # For each anchor, include both n and n+1 to cover even and odd phases.
    grid = np.concatenate([anchors, anchors + 1])
    grid.sort()
    return grid


N_GRID = _build_n_grid()


# --- Schemes to evaluate (subset of full library) ---
SCHEMES = {
    "crr":        schemes.crr,
    "gcrr_gc":    schemes.gcrr_gc,
    "tian_1999":  schemes.tian_1999,
    "joshi_2009": schemes.joshi_2009,
}


def build_cases() -> list[dict]:
    """Enumerate the 8 WAND parameter combinations."""
    cases = []
    for case_id, (K, sigma, T) in enumerate(
        product(WAND_K_VALUES, WAND_SIGMA_VALUES, WAND_T_VALUES), start=1
    ):
        cases.append({
            "case_id": case_id,
            "S": WAND_S,
            "K": K,
            "T": T,
            "r": WAND_R,
            "sigma": sigma,
        })
    return cases


def compute_case_scheme_rows(case: dict, scheme_name: str, scheme_fn) -> list[dict]:
    """For one (case, scheme), compute prices and errors across N_GRID.

    Returns a list of row dicts ready for DataFrame insertion. Handles
    Joshi's odd-n bumping by recording the effective n (params.n) rather
    than the requested n, and deduplicating any collisions.
    """
    S, K, T, r, sigma = case["S"], case["K"], case["T"], case["r"], case["sigma"]
    bs_price = bs_call(S, K, T, r, sigma)

    rows = []
    seen_n_eff = set()
    for n_req in N_GRID:
        params = scheme_fn(S, K, T, r, sigma, int(n_req))
        n_eff = params.n
        # Deduplicate: Joshi at consecutive even/odd n_req can map to the
        # same odd n_eff. Keep first occurrence.
        if n_eff in seen_n_eff:
            continue
        seen_n_eff.add(n_eff)

        price = binomial_price(S, K, T, r, params, "call", "european")
        error = price - bs_price
        scaled_error = n_eff * error
        eta = n_eff % 2

        rows.append({
            "case_id": case["case_id"],
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
            "scheme": scheme_name,
            "n_req": int(n_req),
            "n_eff": int(n_eff),
            "eta": int(eta),
            "price": price,
            "bs_price": bs_price,
            "error": error,
            "scaled_error": scaled_error,
        })
    return rows


def main():
    cases = build_cases()
    print(f"WAND-style panel: {len(cases)} cases x {len(SCHEMES)} schemes "
          f"x ~{len(N_GRID)} n values")
    print(f"Total: ~{len(cases) * len(SCHEMES) * len(N_GRID)} binomial computations")
    print()

    t_start = time.time()
    all_rows = []
    for case in cases:
        print(f"Case {case['case_id']}/{len(cases)}: "
              f"K={case['K']}, sigma={case['sigma']}, T={case['T']}")
        for scheme_name, scheme_fn in SCHEMES.items():
            t_scheme = time.time()
            rows = compute_case_scheme_rows(case, scheme_name, scheme_fn)
            all_rows.extend(rows)
            elapsed = time.time() - t_scheme
            print(f"  {scheme_name:12s} {len(rows):3d} rows  ({elapsed:5.1f}s)")

    df = pd.DataFrame(all_rows)
    print()
    print(f"Total rows: {len(df)}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    # Sanity checks
    assert df["price"].notna().all(), "NaN in prices"
    assert df["bs_price"].notna().all(), "NaN in BS prices"
    assert (df["price"] > 0).all(), "Non-positive prices found"
    assert df["case_id"].nunique() == 8, "Expected 8 cases"
    assert df["scheme"].nunique() == 4, "Expected 4 schemes"

    # Verify GCRR-GC converges to BS at largest n for each case (sanity check)
    for case_id in df["case_id"].unique():
        gc_largest = df[(df["case_id"] == case_id) & (df["scheme"] == "gcrr_gc")] \
                        .sort_values("n_eff").iloc[-1]
        assert abs(gc_largest["error"]) < 0.01, (
            f"GCRR-GC at large n in case {case_id} too far from BS: "
            f"error={gc_largest['error']:.6f}"
        )

    # Save
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "wand_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(f"Schema: {list(df.columns)}")


if __name__ == "__main__":
    main()
