"""Strike-location diagnostics for binomial trees (Section 2.1).

Pure geometric quantities describing where the strike sits in the terminal
log-lattice. They depend only on (S, K, n, u, d), not on p, r, or sigma.

    gamma_n  continuous terminal index of the strike, the real j with
             S u^j d^(n-j) = K:  gamma_n = (log(K/S) - n log d) / log(u/d).
    J_n      weak upper index ceil(gamma_n): smallest j with S u^j d^(n-j) >= K.
    m_n      strict exercise threshold, eq. (1): smallest j with S u^j d^(n-j) > K.
             Equals J_n unless the strike is exactly on a node, where m_n = J_n + 1.
    theta_n  J_n - gamma_n in [0, 1): fractional distance to the node above.
    Delta_n  1 - 2 theta_n in (-1, 1]: Delta = 1 on a node, 0 at the
             geometric midpoint of two adjacent nodes.

For GCRR-GC (Proposition 2.1), gamma_n = n/2 in both phases, so Delta_n = 1
for even n, Delta_n = 0 for odd n, and m_n = N + 1 for every n.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pricer import TreeParameters


@dataclass(frozen=True)
class StrikeLocation:
    """Strike-location coordinates for one tree."""

    gamma: float  # continuous strike index
    J: int  # weak upper index ceil(gamma_n)
    m: int  # strict exercise threshold m_n, eq. (1)
    theta: float  # J - gamma, in [0, 1)
    Delta: float  # 1 - 2*theta, in (-1, 1]


def strike_location(S: float, K: float, params: TreeParameters) -> StrikeLocation:
    """Compute (gamma, J, theta, Delta) for the given tree.

    Args:
        S: Spot price (positive).
        K: Strike price (positive).
        params: TreeParameters with u > d > 0.

    Returns:
        StrikeLocation dataclass.

    Note on floating-point handling:
        When the construction places the strike *exactly* on a terminal
        node (gamma exactly integer), the operations ceil(gamma) and
        gamma - J can become discontinuous under FP rounding: gamma
        computed as n/2 + 1e-15 versus n/2 - 1e-15 yields different J
        values and therefore radically different theta. To make the
        diagnostic robust at exact-alignment cases, gamma is snapped to
        the nearest integer when it lies within machine epsilon of one.
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")

    log_u, log_d = math.log(params.u), math.log(params.d)
    log_ratio = log_u - log_d  # = log(u/d), positive since u > d

    gamma_raw = (math.log(K / S) - params.n * log_d) / log_ratio

    # Snap gamma to nearest integer if within FP precision of one.
    # Without this, exact strike-on-node placements (GCRR-GC even n,
    # GCRR-XPC, Tian (1999), etc.) can produce theta = 0 or theta ~ 1
    # depending on the sign of accumulated rounding error in log(K/S) -
    # n*log(d). The two are geometrically identical; the snap restores
    # that identity.
    nearest_int = round(gamma_raw)
    if abs(gamma_raw - nearest_int) < 1e-10:
        gamma = float(nearest_int)
    else:
        gamma = gamma_raw

    J = math.ceil(gamma)
    theta = J - gamma
    Delta = 1.0 - 2.0 * theta

    # Strict upper neighbor: smallest j with S * u^j * d^(n-j) > K (strict
    # inequality). Equals J when the strike is strictly between nodes
    # (gamma non-integer). When the strike sits exactly on a node (gamma
    # integer, which is what the snap-to-integer above detects), the node
    # at j = gamma satisfies S * u^j * d^(n-j) = K, not > K, so the
    # strict threshold steps up to gamma + 1.
    if gamma == math.floor(gamma):  # gamma is integer => strike on node
        m = int(gamma) + 1
    else:
        m = J

    return StrikeLocation(gamma=gamma, J=J, m=m, theta=theta, Delta=Delta)


def strike_location_sweep(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    scheme,  # callable: (S, K, T, r, sigma, n) -> TreeParameters
    n_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute strike-location coordinates across a sweep of n values.

    Args:
        S, K, T, r, sigma: Market and contract parameters.
        scheme: A scheme function from gcrr.schemes (e.g. schemes.crr,
            schemes.gcrr_gc, schemes.joshi_2009).
        n_values: 1D array of n's at which to evaluate.

    Returns:
        Dictionary with arrays 'gamma', 'J', 'm', 'theta', 'Delta', each
        the same length as n_values.

    Notes:
        Some schemes have parity restrictions (GCRR-XPC requires even n,
        Joshi forces odd n internally). For Joshi, the returned coordinates
        correspond to the *effective* n (n or n+1), and len(output) still
        matches len(n_values).
    """
    gammas = np.empty(len(n_values), dtype=float)
    Js = np.empty(len(n_values), dtype=int)
    ms = np.empty(len(n_values), dtype=int)
    thetas = np.empty(len(n_values), dtype=float)
    Deltas = np.empty(len(n_values), dtype=float)

    for i, n in enumerate(n_values):
        params = scheme(S, K, T, r, sigma, int(n))
        loc = strike_location(S, K, params)
        gammas[i] = loc.gamma
        Js[i] = loc.J
        ms[i] = loc.m
        thetas[i] = loc.theta
        Deltas[i] = loc.Delta

    return {"gamma": gammas, "J": Js, "m": ms, "theta": thetas, "Delta": Deltas}
