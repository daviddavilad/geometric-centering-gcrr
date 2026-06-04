"""
Numerical verification of the 1/n coefficient in Proposition B.1
of the GCRR-GC draft.

Setup: For Bin(n, p_n) with p_n = 1/2 + a/(2*sqrt(n)) and n = 2N + eta,
we compute the upper tail Psi = P(B_n >= N+1) exactly using scipy,
then extract the empirical 1/n coefficient and compare to two candidate
analytical predictions:

    DRAFT (as written):   rho + a^3       + (2*eta - 1) * a/4
    DRAFT (typo-fixed):   rho + a^3 / 4   + (2*eta - 1) * a/4

The expansion under test is

    Psi = Phi(a) - (2*chi_eta / sqrt(n)) * phi(a)
                 + (C(a, eta, rho) / n) * phi(a) + O(n^{-3/2})

where chi_eta = (1 - eta)/2.

For each (a, eta), we evaluate Psi at increasing n, subtract the predicted
Phi(a) - 2*chi_eta/sqrt(n)*phi(a) part, divide by phi(a)/n, and look at
where the empirical residual settles.

To remove the O(1/sqrt(n)) sub-leading bias we fit a three-term model
    residual = C/n + D/n^{1.5} + E/n^2
and report the C coefficient (least squares).

Usage: python3 scripts/check_proposition_B1.py
"""

import numpy as np
from scipy.stats import binom, norm


def exact_upper_tail(n, eta, a, rho=0.0):
    """
    Exact P(B_n >= N+1) using scipy, for n = 2N + eta and
    p_n = 1/2 + a/(2 sqrt(n)) + rho/(2 n^{3/2}).
    """
    if (n - eta) % 2 != 0:
        raise ValueError(f"n={n} has wrong parity for eta={eta}")
    N = (n - eta) // 2
    p = 0.5 + a / (2.0 * np.sqrt(n)) + rho / (2.0 * n ** 1.5)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p_n = {p} out of (0,1) at n={n}, a={a}")
    # P(B_n >= N+1) = 1 - F(N) where F is the binomial CDF.
    return 1.0 - binom.cdf(N, n, p)


def empirical_1_over_n_coefficient(a, eta, rho=0.0, n_values=None):
    """
    Fit the residual after subtracting the leading and 1/sqrt(n) terms
    to a three-term model C/n + D/n^{1.5} + E/n^2 and return C.
    """
    if n_values is None:
        base = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        n_values = [n + (eta - n % 2) % 2 for n in base]

    chi = (1 - eta) / 2.0
    Phi_a = norm.cdf(a)
    phi_a = norm.pdf(a)

    n_list, residuals = [], []
    for n in n_values:
        if (n - eta) % 2 != 0:
            continue
        psi = exact_upper_tail(n, eta, a, rho)
        residual = (psi - Phi_a + (2.0 * chi / np.sqrt(n)) * phi_a) / phi_a
        n_list.append(n)
        residuals.append(residual)

    A = np.array([[1.0 / n, 1.0 / n ** 1.5, 1.0 / n ** 2] for n in n_list])
    b = np.array(residuals)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef[0]


def draft_prediction_as_written(a, eta, rho=0.0):
    """C = rho + a^3 + (2*eta - 1) * a/4"""
    return rho + a ** 3 + (2 * eta - 1) * a / 4.0


def draft_prediction_typo_fixed(a, eta, rho=0.0):
    """C = rho + a^3/4 + (2*eta - 1) * a/4"""
    return rho + a ** 3 / 4.0 + (2 * eta - 1) * a / 4.0


def main():
    print("=" * 78)
    print(" Numerical verification of the 1/n coefficient in Proposition B.1")
    print("=" * 78)
    print()
    print(" For each (a, eta), we extract the empirical 1/n coefficient from")
    print(" Psi - Phi(a) + (2 chi_eta / sqrt n) phi(a), divided by phi(a)/n,")
    print(" and compare to the two candidate analytical predictions.")
    print()

    header = (
        f"{'a':>6} {'eta':>4} | {'empirical C':>14} "
        f"| {'draft as written':>16} {'err':>10} "
        f"| {'draft typo-fixed':>16} {'err':>10}"
    )
    print(header)
    print("-" * len(header))

    a_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, -0.5, -1.0]

    for a in a_values:
        for eta in [0, 1]:
            C_empirical = empirical_1_over_n_coefficient(a, eta)
            C_as_written = draft_prediction_as_written(a, eta)
            C_typo_fixed = draft_prediction_typo_fixed(a, eta)

            err_as_written = abs(C_empirical - C_as_written)
            err_typo_fixed = abs(C_empirical - C_typo_fixed)

            print(
                f"{a:>6.2f} {eta:>4} | {C_empirical:>14.8f} "
                f"| {C_as_written:>16.8f} {err_as_written:>10.2e} "
                f"| {C_typo_fixed:>16.8f} {err_typo_fixed:>10.2e}"
            )

    print()
    print("Interpretation:")
    print("  The 'empirical C' column is what the exact binomial CDF actually")
    print("  produces in the limit. Whichever analytical prediction matches it")
    print("  to high precision (errors ~ 1e-6 or smaller) is the correct form.")
    print()
    print("  The 'err' columns are |empirical - prediction|. The smaller column")
    print("  identifies the correct analytical form.")


if __name__ == "__main__":
    main()