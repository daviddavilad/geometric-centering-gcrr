"""Binomial scheme parameterizations.

Every function returns a TreeParameters for the generic pricer.

The GCRR family shares the jump structure
    u = exp(lambda * sigma * sqrt(dt))
    d = exp(-(1/lambda) * sigma * sqrt(dt))
    p = (R - d) / (u - d),   R = exp(r*dt)
and differs only in how lambda is chosen:
    - crr:           lambda = 1
    - gcrr_fixed:    lambda = const
    - gcrr_xpc:      lambda chosen so a terminal node sits ON the strike (even n)
    - gcrr_gc:       lambda chosen by the parity-free geometric-center condition

Verify each lambda formula against your handwritten derivations before
trusting numerical output.
"""
from __future__ import annotations

import numpy as np

from .pricer import TreeParameters


def _gcrr_from_lambda(
    lam: float, S: float, K: float, T: float, r: float, sigma: float, n: int
) -> TreeParameters:
    """Build a GCRR TreeParameters from a given lambda."""
    dt = T / n
    R = np.exp(r * dt)
    u = np.exp(lam * sigma * np.sqrt(dt))
    d = np.exp(-(1.0 / lam) * sigma * np.sqrt(dt))
    p = (R - d) / (u - d)
    return TreeParameters(u=u, d=d, p=p, n=n, dt=dt, R=R, lam=lam)


def crr(S, K, T, r, sigma, n) -> TreeParameters:
    """Standard CRR: the GCRR special case lambda = 1."""
    return _gcrr_from_lambda(1.0, S, K, T, r, sigma, n)


def gcrr_fixed(S, K, T, r, sigma, n, lam=1.1) -> TreeParameters:
    """GCRR with a fixed (constant) stretch parameter. Control scheme."""
    return _gcrr_from_lambda(lam, S, K, T, r, sigma, n)


def gcrr_xpc(S, K, T, r, sigma, n) -> TreeParameters:
    """GCRR-XPC (Chung-Shih 2007): strike on the central terminal node.

    Requires n even. The construction places the central terminal node
    S * u^(n/2) * d^(n/2) exactly on the strike K.

    Taking logs of that condition and using h = T/n yields
        lambda - 1/lambda = 2 log(K/S) / (sigma * sqrt(nT))  =:  A_n,
    whose unique positive root is
        lambda = (A_n + sqrt(A_n^2 + 4)) / 2.

    This coincides algebraically with the GCRR-GC lambda at eta = 0 (even
    phase). Kept as a separate function for intellectual provenance
    (Chung-Shih 2007) and to enforce the even-n requirement.
    """
    if n % 2 != 0:
        raise ValueError(f"GCRR-XPC requires even n, got {n}")
    x = np.log(K / S)
    A_n = 2.0 * x / (sigma * np.sqrt(n * T))
    lam = (A_n + np.sqrt(A_n**2 + 4.0)) / 2.0
    return _gcrr_from_lambda(lam, S, K, T, r, sigma, n)


def gcrr_gc(S, K, T, r, sigma, n) -> TreeParameters:
    """GCRR-GC (Chen-Davila): parity-free geometric-center condition.

    From draft Proposition 4.1:
        A_n    = 2 ln(X/S) / (sigma * sqrt(n*T))
        lambda = ( A_n + sqrt(A_n^2 + 4) ) / 2

    Works for both even and odd n. NOTE: verify against your derivation.
    """
    x = np.log(K / S)
    A_n = 2.0 * x / (sigma * np.sqrt(n * T))
    lam = (A_n + np.sqrt(A_n**2 + 4.0)) / 2.0
    return _gcrr_from_lambda(lam, S, K, T, r, sigma, n)


def _tian_flexible_from_lambda(
    lam: float, S: float, K: float, T: float, r: float, sigma: float, n: int
) -> TreeParameters:
    """Build Tian (1999) flexible-binomial TreeParameters from a given tilt.

    Implements Tian (1999) eq. (6):
        u = exp(sigma * sqrt(dt) + lam * sigma^2 * dt)
        d = exp(-sigma * sqrt(dt) + lam * sigma^2 * dt)
        p = (e^{r*dt} - d) / (u - d)

    Note: Tian's "lambda" is a *tilt* parameter (additive shift in log-space),
    structurally different from the GCRR *stretch* parameter (multiplicative
    scaling of sqrt(dt) in the exponent). We store None in the TreeParameters
    lam field to avoid confusion.
    """
    dt = T / n
    sqrt_dt = np.sqrt(dt)
    sigma2_dt = sigma * sigma * dt
    R = np.exp(r * dt)
    u = np.exp(sigma * sqrt_dt + lam * sigma2_dt)
    d = np.exp(-sigma * sqrt_dt + lam * sigma2_dt)
    p = (R - d) / (u - d)
    return TreeParameters(u=u, d=d, p=p, n=n, dt=dt, R=R, lam=None)


def tian_1999(S, K, T, r, sigma, n) -> TreeParameters:
    """Tian (1999) strike-aligned flexible binomial.

    Places the terminal node (n, j_0) exactly at the strike K by solving
    Tian (1999) eqs. (11) and (13) for the tilt parameter lambda. Yields
    smooth (essentially monotone) O(1/n) convergence and admits Richardson
    extrapolation to O(1/n^2).

    Procedure:
        1. CRR baseline factors u_0 = e^{sigma*sqrt(dt)}, d_0 = 1/u_0.
        2. Real-valued ideal up-count
               eta = (log(K/S) - n*log(d_0)) / log(u_0/d_0)
           (eq. 10), rounded to nearest integer j_0 (eq. 11).
        3. Solve for the strike-aligning lambda (eq. 13):
               lam = (log(K/S) - (2*j_0 - n)*sigma*sqrt(dt)) / (n*sigma^2*dt)
        4. Return flexible parameters with that lambda.

    Verified against Tian (1999) Tables I and II in the public option-pricing
    repository (28 paper-reproduction tests pass to 4-6 decimal places).
    """
    dt = T / n
    sqrt_dt = np.sqrt(dt)

    # CRR baseline (eq. 8)
    u_0 = np.exp(sigma * sqrt_dt)
    d_0 = np.exp(-sigma * sqrt_dt)

    # Ideal up-count (eq. 10) and nearest integer j_0 (eq. 11)
    log_K_over_S = np.log(K / S)
    eta = (log_K_over_S - n * np.log(d_0)) / np.log(u_0 / d_0)
    j_0 = int(round(eta))

    # Strike-aligning tilt (eq. 13)
    lam_tilt = (log_K_over_S - (2 * j_0 - n) * sigma * sqrt_dt) / (n * sigma**2 * dt)

    return _tian_flexible_from_lambda(lam_tilt, S, K, T, r, sigma, n)
