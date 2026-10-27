"""Digital (cash-or-nothing) call prices on centered binomial trees (Section 4).

    D^str_{N,eta} = e^{-rT} Psi_{N,eta}(p_n^B),       payoff 1{S_T > K}
    D^hn_{N,eta}  = e^{-rT} Psi^hn_{N,eta}(p_n^B),    payoff 1{S_T > K} + (1/2) 1{S_T = K}
    Psi^hn_{N,eta}(p) = Psi_{N,eta}(p) + ((1 - eta)/2) C(2N, N) p^N (1 - p)^N   (eq. 11)

with n = 2N + eta and Psi_{N,eta}(p) = P(Bin(n, p) >= N + 1). Both prices
assume the strict exercise threshold is m_n = N + 1, which holds for GCRR-GC
(Proposition 2.1) and for the odd-step centered benchmark; the half-node
price also assumes the even-phase strike sits exactly on node N. Both
conditions are checked, and other trees raise ValueError.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .pricer import TreeParameters
from .diagnostics import strike_location

def _check_centered(S, K, params, half_node: bool = False) -> None:
    """Raise ValueError unless m_n = N + 1 and, for the half-node price at
    even n, the strike lies exactly on node N (gamma_n = N)."""
    n = params.n
    N = n // 2
    eta = n % 2

    loc = strike_location(S, K, params)

    # Condition 1: Unified strict exercise threshold
    if loc.m != N + 1:
        raise ValueError(
            f"Centered-tree condition failed: "
            f"strict threshold m={loc.m}, "
            f"expected N+1={N + 1} for n={n}."
        )

    # Condition 2: Even-phase central-node strike alignment
    if half_node and eta == 0 and loc.gamma != N:
        raise ValueError(
            f"Half-node condition failed: "
            f"gamma={loc.gamma}, expected N={N} "
            f"for even n={n}."
        )


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
        S: Spot (positive). Used by the centered-tree check.
        K: Strike (positive). Used by the centered-tree check; it also
            enters through params.p via the scheme.
        T: Time to expiry in years. Used for the discount factor.
        r: Continuously compounded risk-free rate. Used for the discount
            factor.
        params: TreeParameters with (u, d, p, n, dt, R, lam).

    Returns:
        The discounted strict digital price.

    Notes:
        For GCRR-GC, N+1 is the unified strict threshold (Proposition 2.1) 
        - meaning the strike is at the geometric center in odd phase or 
        on the central node in even phase, and the strict tail j >= N+1
        captures S_T > K in both cases.
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time T must be positive, got {T}")

    _check_centered(S, K, params)

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
    """Half-node corrected digital call price on a binomial tree [Section 4, eqs. (11)–(12)].

    The strict digital D^str_{N,eta} = e^{-rT} Psi_{N,eta}(p_n^B) suffers
    an O(1/sqrt(n)) "central-atom discrepancy" in the even phase: when
    the strike sits exactly on the central terminal node, the strict
    upper threshold j >= N+1 misses the half-mass of the central atom
    that "should" be assigned to S_T > K.

    The half-node correction adds back half the central atom for even
    phase (eta = 0) and leaves the odd phase unchanged:

        Psi^hn_{N,eta}(p) = Psi_{N,eta}(p) + ((1-eta)/2) * C(2N, N) p^N (1-p)^N

    The price is then D^hn_{N,eta} = e^{-rT} * Psi^hn_{N,eta}(p_n^B).

    By construction, the corrected digital satisfies the exact symmetry
        Psi^hn_{N,0}(1/2) = Psi^hn_{N,1}(1/2) = 1/2,
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
    
    _check_centered(S, K, params, half_node=True)

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