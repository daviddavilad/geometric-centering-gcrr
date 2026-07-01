"""
Numerical verification of the phase gap A_1 - A_0 for the GCRR-GC call price.

Theorem 7.1 (with the corrected A_eta) predicts that
the two phase-wise asymptotic constants differ by

    A_1 - A_0 = (1/2) * S * phi(d_1) * sigma * sqrt(T),

since the rho_S - rho_B and cubic-difference contributions to A_eta are
eta-independent and cancel, leaving only the (2*eta - 1)*(d_1 - d_2)/4
term.

This script verifies the closed form empirically using the in-repo GCRR-GC
pricer:

  1. For each phase eta in {0, 1}, compute the GCRR-GC call price at
     increasing n via gcrr.schemes.gcrr_gc + gcrr.pricer.binomial_price.
  2. Fit n * (C_{N,eta} - C_BS) -> A_eta with a small higher-order
     correction.
  3. Subtract to get the empirical phase gap A_1 - A_0.
  4. Compare to the analytical prediction.

Usage: uv run python scripts/verify_phase_gap.py
"""

import numpy as np
from scipy.stats import norm

from gcrr import schemes
from gcrr.pricer import binomial_price
from gcrr.black_scholes import bs_call, d1_d2


def predicted_phase_gap(S, K, T, r, sigma):
    """A_1 - A_0 = (1/2) * S * phi(d_1) * sigma * sqrt(T)."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return 0.5 * S * norm.pdf(d1) * sigma * np.sqrt(T)


def fit_A_eta(eta, S, K, T, r, sigma, n_values=None):
    """
    Fit n * (C_{N,eta} - C_BS) = A_eta + B/sqrt(n) + C/n via least squares
    and return A_eta. The expansion is C_{N,eta} = C_BS + A_eta/n + O(n^{-3/2}),
    so the leading constant of n*(C - C_BS) is exactly A_eta.
    """
    if n_values is None:
        base = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        # Adjust each n to match the requested phase.
        n_values = [n + (eta - n % 2) % 2 for n in base]

    C_BS = bs_call(S, K, T, r, sigma)

    n_list, scaled_errors = [], []
    for n in n_values:
        if n % 2 != eta:
            continue
        params = schemes.gcrr_gc(S, K, T, r, sigma, n)
        C = binomial_price(S, K, T, r, params, option_type="call", exercise="european")
        n_list.append(n)
        scaled_errors.append(n * (C - C_BS))

    # Fit scaled_error = A + B/sqrt(n) + C/n.
    A_mat = np.array([[1.0, 1.0 / np.sqrt(n), 1.0 / n] for n in n_list])
    b_vec = np.array(scaled_errors)
    coef, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    return coef[0]  # A_eta


def main():
    # Five benchmark cases spanning ATM/ITM/OTM, varied rates and tenors.
    benchmarks = [
        # (label, S, K, T, r, sigma)
        ("ATM, r=5%, sigma=20%, T=1y",   100.0, 100.0, 1.00, 0.05, 0.20),
        ("ATM, r=0,  sigma=30%, T=1y",   100.0, 100.0, 1.00, 0.00, 0.30),
        ("ITM, K=90,  r=5%, sigma=20%",  100.0,  90.0, 1.00, 0.05, 0.20),
        ("OTM, K=110, r=5%, sigma=20%",  100.0, 110.0, 1.00, 0.05, 0.20),
        ("short-dated T=0.25",           100.0, 100.0, 0.25, 0.05, 0.20),
    ]

    print("=" * 88)
    print(" Numerical verification of the phase gap A_1 - A_0 (GCRR-GC, in-repo pricer)")
    print("=" * 88)
    print()
    print(" Predicted form: A_1 - A_0 = (1/2) * S * phi(d_1) * sigma * sqrt(T)")
    print()

    header = (
        f"{'case':<32} | {'A_0 fit':>10} {'A_1 fit':>10} "
        f"| {'gap fit':>10} {'gap pred':>10} {'err':>10}"
    )
    print(header)
    print("-" * len(header))

    for label, S, K, T, r, sigma in benchmarks:
        A0 = fit_A_eta(0, S, K, T, r, sigma)
        A1 = fit_A_eta(1, S, K, T, r, sigma)
        gap_fit = A1 - A0
        gap_pred = predicted_phase_gap(S, K, T, r, sigma)
        err = abs(gap_fit - gap_pred)
        print(
            f"{label:<32} | {A0:>10.4f} {A1:>10.4f} "
            f"| {gap_fit:>10.4f} {gap_pred:>10.4f} {err:>10.2e}"
        )

    print()
    print("Interpretation:")
    print("  'gap fit' is the empirical A_1 - A_0 from the in-repo GCRR-GC pricer.")
    print("  'gap pred' is the analytical (1/2) * S * phi(d_1) * sigma * sqrt(T).")
    print("  Matching to within fit precision verifies the closed form across")
    print("  the parameter cases tested.")


if __name__ == "__main__":
    main()