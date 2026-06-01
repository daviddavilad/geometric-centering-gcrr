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
    scheme,  # callable: (S, K, T, r, sigma, n) -> TreeParameters
    n_values: np.ndarray,
    option_type: str = "call",
) -> PhaseSeparatedErrors:
    """Compute pricing errors across a sweep of n, separated by parity.

    Args:
        S, K, T, r, sigma: Market and contract parameters.
        scheme: A scheme function from gcrr.schemes.
        n_values: 1D array of n's at which to price.
        option_type: 'call' or 'put'.

    Returns:
        PhaseSeparatedErrors dataclass.

    Notes:
        - Joshi (2009) bumps even n to n+1 internally; we use the *requested*
          n for parity assignment (a request at n=200 lands in n_even with
          err computed at the effective n=201, which is a small but worth-
          knowing convention).
        - For Black-Scholes reference: dividend yield q=0 throughout
          (consistent with the schemes that don't take q).
    """
    if option_type == "call":
        bs_ref = bs_call(S, K, T, r, sigma)
    elif option_type == "put":
        bs_ref = bs_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    errs = np.empty(len(n_values), dtype=float)
    for i, n in enumerate(n_values):
        params = scheme(S, K, T, r, sigma, int(n))
        price = binomial_price(S, K, T, r, params, option_type, "european")
        errs[i] = price - bs_ref

    n_arr = np.asarray(n_values)
    even_mask = (n_arr % 2 == 0)
    odd_mask = ~even_mask

    return PhaseSeparatedErrors(
        n_even=n_arr[even_mask],
        n_odd=n_arr[odd_mask],
        err_even=errs[even_mask],
        err_odd=errs[odd_mask],
        scaled_err_even=n_arr[even_mask] * errs[even_mask],
        scaled_err_odd=n_arr[odd_mask] * errs[odd_mask],
    )
