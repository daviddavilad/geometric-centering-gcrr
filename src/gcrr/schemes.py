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
    """GCRR-XPC (Chung-Shih 2007): strike on the central node. Requires n even.

    lambda(n) from Chung-Shih eq. (23):
        A      = 2 ln(X/S) / (sigma * sqrt(n*T))     [note: their A_n form]
        lambda = ln(X/S)/(sigma*sqrt(n*T)) + sqrt( (2 ln(X/S)/(sigma*sqrt(n*T)))^2 + 4 ) / 2

    NOTE: verify this against your handwritten derivation of eq. (23).
    """
    if n % 2 != 0:
        raise ValueError(f"GCRR-XPC requires even n, got {n}")
    x = np.log(K / S)
    a = x / (sigma * np.sqrt(n * T))
    lam = a + np.sqrt((2 * a) ** 2 + 4) / 2.0  # placeholder form -- CONFIRM
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
