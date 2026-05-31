"""Strike-location diagnostics for binomial trees.

Pure geometric quantities characterizing where the strike sits in the
terminal log-lattice. Depend only on (S, K, n, u, d) -- not on p, r, sigma.

Definitions (GCRR-GC draft Section 1):
    gamma_n(X)  = continuous strike index, the real-valued j such that
                  S * u^j * d^(n-j) = K. Equivalently,
                  gamma_n = (log(K/S) - n*log(d)) / log(u/d).

    J_n(X)      = weak upper neighboring index = ceil(gamma_n).
                  The smallest integer j with S * u^j * d^(n-j) >= K.

    theta_n(X)  = J_n - gamma_n in [0, 1). Fractional log-distance from
                  the strike to the next node above, normalized by the
                  lattice spacing.

    Delta_n(X)  = 1 - 2*theta_n in (-1, 1]. Centered version of theta.
                  Delta = 0 <=> strike at geometric midpoint of two nodes;
                  Delta = 1 <=> strike exactly on a node (theta = 0);
                  Delta -> -1 <=> strike just above a node (theta -> 1).

For the GCRR-GC construction:
    even n (eta=0): gamma_n = n/2 exactly, theta_n = 0, Delta_n = 1
    odd n  (eta=1): gamma_n = n/2,         theta_n = 1/2, Delta_n = 0
This is the parity-free property: |Delta_n| stays at extreme values for
both parities, instead of oscillating pseudo-randomly as in CRR.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .pricer import TreeParameters


@dataclass(frozen=True)
class StrikeLocation:
    """Strike-location coordinates for one tree."""
    gamma: float   # continuous strike index
    J: int         # weak upper neighboring index = ceil(gamma)
    theta: float   # J - gamma, in [0, 1)
    Delta: float   # 1 - 2*theta, in (-1, 1]


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

    return StrikeLocation(gamma=gamma, J=J, theta=theta, Delta=Delta)


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
        Dictionary with arrays 'gamma', 'J', 'theta', 'Delta', each
        the same length as n_values.

    Notes:
        Some schemes have parity restrictions (GCRR-XPC requires even n,
        Joshi forces odd n internally). For Joshi, the returned coordinates
        correspond to the *effective* n (n or n+1), and len(output) still
        matches len(n_values).
    """
    gammas = np.empty(len(n_values), dtype=float)
    Js = np.empty(len(n_values), dtype=int)
    thetas = np.empty(len(n_values), dtype=float)
    Deltas = np.empty(len(n_values), dtype=float)

    for i, n in enumerate(n_values):
        params = scheme(S, K, T, r, sigma, int(n))
        loc = strike_location(S, K, params)
        gammas[i] = loc.gamma
        Js[i] = loc.J
        thetas[i] = loc.theta
        Deltas[i] = loc.Delta

    return {"gamma": gammas, "J": Js, "theta": thetas, "Delta": Deltas}
