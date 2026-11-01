"""Binomial scheme parameterizations.

Every function returns a TreeParameters for the generic pricer.

The GCRR family shares the jump structure
    u = exp(lambda * sigma * sqrt(dt))
    d = exp(-(1/lambda) * sigma * sqrt(dt))
    p = (R - d) / (u - d),   R = exp(r*dt)
and differs only in how lambda is chosen:
    - crr:           lambda = 1
    - gcrr_fixed:    lambda = const
    - gcrr_xpc:      lambda chosen so the central node sits on the strike (even n)
    - gcrr_gc:       lambda chosen by the parity-free geometric-center condition

tian_1999 and joshi_2009 use Tian's flexible form instead,
    u, d = exp(+/- sigma * sqrt(dt) + lambda * sigma^2 * dt),
with a tilt lambda in place of a stretch. Tian puts a terminal node on
the strike; Joshi (odd n only) puts the strike midway between the two
central nodes.
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
        lambda - 1/lambda = 2 log(K/S) / (sigma * sqrt(nT))  =:  kappa_n,
    whose unique positive root is
        lambda = (kappa + sqrt(kappa^2 + 4)) / 2.

    This coincides algebraically with the GCRR-GC lambda at eta = 0 (even
    phase). Kept as a separate function for intellectual provenance
    (Chung-Shih 2007) and to enforce the even-n requirement.
    """
    if n % 2 != 0:
        raise ValueError(f"GCRR-XPC requires even n, got {n}")
    x = np.log(K / S)
    kappa = 2.0 * x / (sigma * np.sqrt(n * T))
    lam = (kappa + np.sqrt(kappa**2 + 4.0)) / 2.0
    return _gcrr_from_lambda(lam, S, K, T, r, sigma, n)


def gcrr_gc(S, K, T, r, sigma, n) -> TreeParameters:
    """GCRR-GC: parity-free geometric-center condition (Proposition 2.1).

    Imposes K^2 = S_{n,N} S_{n,N+eta}, which places the strike on the central
    node for even n and at the geometric midpoint of the two central nodes
    for odd n. The unique positive stretch is
        kappa_n  = 2 log(K/S) / (sigma sqrt(nT)),
        lambda_n = (kappa_n + sqrt(kappa_n^2 + 4)) / 2.
    For even n this coincides with gcrr_xpc (Chung and Shih, 2007).
    """
    x = np.log(K / S)
    kappa = 2.0 * x / (sigma * np.sqrt(n * T))
    lam = (kappa + np.sqrt(kappa**2 + 4.0)) / 2.0
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
    Tian (1999) eqs. (11) and (13) for the tilt parameter lambda. The
    strike node is not the central one, and the error jumps wherever
    j_0 - n // 2 changes, that is, wherever the strike moves one node
    further from the center (see Figure 3 and tests/test_errors.py).

    Procedure:
        1. CRR baseline factors u_0 = e^{sigma*sqrt(dt)}, d_0 = 1/u_0.
        2. Real-valued ideal up-count
               eta = (log(K/S) - n*log(d_0)) / log(u_0/d_0)
           (eq. 10), rounded to nearest integer j_0 (eq. 11).
        3. Solve for the strike-aligning lambda (eq. 13):
               lam = (log(K/S) - (2*j_0 - n)*sigma*sqrt(dt)) / (n*sigma^2*dt)
        4. Return flexible parameters with that lambda.
    """
    dt = T / n
    sqrt_dt = np.sqrt(dt)

    # CRR baseline (eq. 8)
    u_0 = np.exp(sigma * sqrt_dt)
    d_0 = np.exp(-sigma * sqrt_dt)

    # Ideal up-count (eq. 10) and nearest integer j_0 (eq. 11)
    log_K_over_S = np.log(K / S)
    eta = (log_K_over_S - n * np.log(d_0)) / np.log(u_0 / d_0)
    j_0 = int(np.rint(eta))

    # Strike-aligning tilt (eq. 13)
    lam_tilt = (log_K_over_S - (2 * j_0 - n) * sigma * sqrt_dt) / (n * sigma**2 * dt)

    return _tian_flexible_from_lambda(lam_tilt, S, K, T, r, sigma, n)


def joshi_2009(S, K, T, r, sigma, n) -> TreeParameters:
    """Joshi (2009) odd-step centered tree.

    Tian's flexible tree with tilt lambda = mu / sigma^2: the drift mu
    shifts every step by the same amount, and the up and down log-steps
    stay symmetric at +/- sigma*sqrt(dt). Where Tian puts a terminal node
    on the strike, Joshi centers the tree on it.

    Construction (Joshi 2009, Sections 3-4):
        mu = log(K/S0) / T              # places strike at log-center
        u  = exp(mu*dt + sigma*sqrt(dt))
        d  = exp(mu*dt - sigma*sqrt(dt))
        p  = (e^{r*dt} - d) / (u - d)

    Requires n odd: with n = 2N+1 there are 2N+2 terminal nodes, and the
    strike sits exactly between indices N and N+1. Joshi defines the tree
    for odd n only; we use n directly when odd and n+1 when even.

    The resulting tree admits a *complete* asymptotic expansion in 1/n
    (no oscillating term), so Richardson extrapolation works cleanly.
    Convergence rate: O(1/n) un-extrapolated, O(1/n^2) after one
    Richardson step.

    Reference: Joshi (2009), "Achieving Smooth Asymptotics for the
    Prices of European Options in Binomial Trees."
    """
    # Enforce n odd; Joshi requires 2N+1 steps
    n_eff = n if n % 2 == 1 else n + 1

    dt = T / n_eff
    sqrt_dt = np.sqrt(dt)
    R = np.exp(r * dt)

    # Strike-centering drift (Joshi Section 4)
    mu = np.log(K / S) / T

    # Symmetric step around the mu-drift center
    u = np.exp(mu * dt + sigma * sqrt_dt)
    d = np.exp(mu * dt - sigma * sqrt_dt)
    p = (R - d) / (u - d)

    return TreeParameters(u=u, d=d, p=p, n=n_eff, dt=dt, R=R, lam=None)
