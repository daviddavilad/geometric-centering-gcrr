"""Phase-separated pricing error infrastructure.

Computes the binomial call price C_n via backward induction for each n in
a given sweep, subtracts the Black-Scholes reference, and returns the
errors separated by parity (n even vs n odd).

Phase separation matters because the GCRR-GC paper's main convergence
theorem produces *phase-wise* coefficients A_0 and A_1 in
    C_{N,eta} = C_BS + A_eta / n + O(n^{-3/2}),
where eta = n mod 2. Plotting unscaled |C_n - C_BS| against n on log-log
axes mixes the two phases; separating them is the right diagnostic.

The "scaled error" n * (C_n - C_BS) is the natural quantity: under
phase-wise smooth convergence, this scales to A_eta as n grows. A flat
line at level A_eta (one for each parity) is the visual signature of
the GCRR-GC paper's main theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_call, bs_put
from .pricer import binomial_price


@dataclass(frozen=True)
class PhaseSeparatedErrors:
    """Errors of a scheme across a sweep of n values, separated by parity.

    Attributes:
        n_even, n_odd: Arrays of the n values, partitioned by parity.
        err_even, err_odd: Signed errors C_n - C_BS at those n values.
        scaled_err_even, scaled_err_odd: n * err, the diagnostic quantity
            for visualizing phase-wise smooth convergence.
    """

    n_even: np.ndarray
    n_odd: np.ndarray
    err_even: np.ndarray
    err_odd: np.ndarray
    scaled_err_even: np.ndarray
    scaled_err_odd: np.ndarray


def phase_separated_errors(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    scheme,
    n_values: np.ndarray,
    option_type: str = "call",
) -> PhaseSeparatedErrors:
    """Compute pricing errors across a sweep of n, separated by parity.

    Uses the *effective* n actually consumed by the pricer for both the
    scaled-error coefficient AND the parity partition. This matters for
    schemes that enforce a parity (e.g. Joshi 2009 forces n odd by
    bumping even requests to n+1): without this, even-n requests to
    Joshi would appear under "even" with a scaled error computed at the
    underlying odd n, producing misleading duplicate points.

    Trade-off: if the user passes n_values = arange(50, 500), and the
    scheme bumps every even n by 1, the returned n_odd will contain
    duplicates (each even n bumped to the same odd n+1 as the next odd
    request). We de-duplicate after the fact by keeping the first
    occurrence of each effective n.
    """
    if option_type == "call":
        bs_ref = bs_call(S, K, T, r, sigma)
    elif option_type == "put":
        bs_ref = bs_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    # Price each requested n and record the effective n that the scheme used.
    n_eff_list = []
    err_list = []
    for n in n_values:
        params = scheme(S, K, T, r, sigma, int(n))
        price = binomial_price(S, K, T, r, params, option_type, "european")
        n_eff_list.append(params.n)
        err_list.append(price - bs_ref)

    n_eff = np.asarray(n_eff_list)
    errs = np.asarray(err_list)

    # De-duplicate: keep first occurrence of each effective n.
    _, unique_idx = np.unique(n_eff, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original order
    n_eff = n_eff[unique_idx]
    errs = errs[unique_idx]

    even_mask = n_eff % 2 == 0
    odd_mask = ~even_mask

    return PhaseSeparatedErrors(
        n_even=n_eff[even_mask],
        n_odd=n_eff[odd_mask],
        err_even=errs[even_mask],
        err_odd=errs[odd_mask],
        scaled_err_even=n_eff[even_mask] * errs[even_mask],
        scaled_err_odd=n_eff[odd_mask] * errs[odd_mask],
    )


def fit_leading_constant(n_values, values) -> float:
    """Least-squares fit of y_n = A + B*n^(-1/2) + C/n.

    Returns the fitted leading asymptotic constant A.
    """
    n = np.asarray(n_values, dtype=float)
    y = np.asarray(values, dtype=float)

    X = np.column_stack(
        [
            np.ones_like(n),
            n ** (-0.5),
            1.0 / n,
        ]
    )

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    return float(coef[0])
