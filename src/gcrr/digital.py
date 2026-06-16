"""Digital option pricing on binomial trees.

A strict digital call pays 1 if S_T > K, 0 otherwise. Its binomial price
is the discounted probability that the terminal index j exceeds the
strict upper threshold N+1 (in the GCRR-GC strict-threshold sense):

    D^>_{N,eta} = e^{-rT} * P(Bin(n, p_n^B) >= N+1)

where n = 2N + eta, p_n^B is the bond-numeraire up-probability of the
underlying tree, and the threshold N+1 is the unified strict threshold
that GCRR-GC produces for both parities (Section 9.2 of the draft).

This module is intentionally separate from pricer.py:
    - The call/put pricer uses backward induction (O(n^2) time)
    - The strict digital reduces to a single binomial tail evaluation
      via scipy.stats.binom.sf, which is O(n) and numerically stable
      for large n.

The half-node corrected digital (Section 9.3, Figure 5) will live here
as well once implemented.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .pricer import TreeParameters


def digital_strict_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: TreeParameters,
) -> float:
    """Strict digital call price on a binomial tree.

    The payoff is 1 if S_T > K (strict inequality), 0 otherwise. Priced as
        e^{-rT} * P(Bin(n, p) >= N+1)
    where p is the bond-numeraire probability and N = floor(n/2). For n
    even (eta = 0), N = n/2 and the threshold j >= N+1 is one above the
    central index. For n odd (eta = 1), N = (n-1)/2 and the threshold is
    again j >= N+1 (the geometric-midpoint placement of GCRR-GC).

    Args:
        S: Spot (positive). Not used in this calculation directly but
            kept for API symmetry with binomial_price.
        K: Strike (positive). Not used in this calculation directly
            (the strike enters via params.p, which depends on K through
            the scheme). Kept for API symmetry.
        T: Time to expiry in years. Used for the discount factor.
        r: Continuously compounded risk-free rate. Used for the discount
            factor.
        params: TreeParameters with (u, d, p, n, dt, R, lam).

    Returns:
        The discounted strict digital price.

    Notes:
        For GCRR-GC, N+1 is the unified strict threshold (Section 9.2 of
        the GCRR-GC draft) -- meaning the strike is at the geometric
        center in odd phase or on the central node in even phase, and
        the strict tail j >= N+1 captures S_T > K in both cases.
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")

    n, p = params.n, params.p
    N = n // 2  # n = 2N+eta => n//2 = N for both parities
    # P(Bin(n, p) >= N+1) = sf(N, n, p) where sf(k) = P(X > k)
    psi = binom.sf(N, n, p)
    return float(np.exp(-r * T) * psi)