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


def digital_half_node_corrected_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: TreeParameters,
) -> float:
    """Half-node corrected digital call price on a binomial tree.

    The strict digital D^>_{N,eta} = e^{-rT} Psi_{N,eta}(p_n^B) suffers
    an O(1/sqrt(n)) "central-atom discrepancy" in the even phase: when
    the strike sits exactly on the central terminal node, the strict
    upper threshold j >= N+1 misses the half-mass of the central atom
    that "should" be assigned to S_T > K.

    The half-node correction adds back half the central atom for even
    phase (eta = 0) and leaves the odd phase unchanged:

        Psi^*_{N,eta}(p) = Psi_{N,eta}(p) + ((1-eta)/2) * C(2N, N) p^N (1-p)^N

    The price is then D^*_{N,eta} = e^{-rT} * Psi^*_{N,eta}(p_n^B).

    By construction, the corrected digital satisfies the exact symmetry
        Psi^*_{N,0}(1/2) = Psi^*_{N,1}(1/2) = 1/2,
    which removes the leading O(1/sqrt(n)) error and is expected to
    produce O(1/n) convergence in the even phase.

    Args:
        S, K, T, r, params: As in digital_strict_price.

    Returns:
        The discounted half-node corrected digital price.

    Notes:
        For odd n, the correction term is zero, so this function is
        equivalent to digital_strict_price in the odd phase. The
        difference appears only in the even phase.
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")

    n, p = params.n, params.p
    N = n // 2
    eta = n % 2

    # Strict tail Psi_{N,eta}(p) = P(Bin(n, p) >= N+1)
    psi = binom.sf(N, n, p)

    # Half-node correction: ((1-eta)/2) * C(2N, N) p^N (1-p)^N.
    # This is zero for odd n (eta=1), 1/2 of the central atom for even n.
    # For odd n the central atom binom.pmf(N, 2N, p) does not match
    # binom.pmf(N, n, p) (since n = 2N+1, not 2N), but the (1-eta)/2 = 0
    # factor zeros out the entire correction term in that case anyway.
    if eta == 0:
        # Even phase: n = 2N, central atom = P(Bin(2N, p) = N) = C(2N, N) p^N (1-p)^N
        central_atom = binom.pmf(N, n, p)
        correction = 0.5 * central_atom
    else:
        correction = 0.0

    return float(np.exp(-r * T) * (psi + correction))