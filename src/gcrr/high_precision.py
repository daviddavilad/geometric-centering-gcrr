"""High-precision GCRR-GC call prices (mpmath) for extrapolation diagnostics.

Richardson-type combinations such as ((n+2) C_{n+2} - n C_n) / 2 cancel the
leading error but amplify rounding by ~n. In double precision this
contaminates extrapolated errors below ~1e-8 (19% at n = 5000). These
functions evaluate Proposition 3.1 with every quantity, including the tree,
in mpmath arithmetic.

Callers must set mpmath.mp.dps (e.g. 40) BEFORE calling, and keep it set
for all arithmetic on the results: mpmath numbers keep their digits, but
operations run at the current global precision.
"""
from __future__ import annotations

import mpmath as mp

def bs_call_mp(S, K, T, r, sigma):
    """Black-Scholes call in mpmath: S Phi(d1) - K e^{-rT} Phi(d2)."""

    S, K, T, r, sigma = (mp.mpf(str(v)) for v in (S, K, T, r, sigma))

    s = sigma * mp.sqrt(T)

    d1 = (mp.log(S / K) + (r + sigma**2 / 2) * T) / s
    d2 = d1 - s

    return S * mp.ncdf(d1) - K * mp.exp(-r * T) * mp.ncdf(d2)

def gcrr_gc_call_mp(S, K, T, r, sigma, n: int):
    """GCRR-GC European call via Proposition 3.1, all in mpmath:
        kappa = 2 log(K/S) / (sigma sqrt(nT)),  lambda = (kappa + sqrt(kappa^2 + 4)) / 2,
        h = T/n,  u = exp(lambda sigma sqrt(h)),  d = exp(-sigma sqrt(h) / lambda),
        p^B = (e^{rh} - d) / (u - d),  p^S = p^B u e^{-rh},  m = N + 1,
        C = S Psi(p^S) - K e^{-rT} Psi(p^B),
        Psi(p) = P(Bin(n, p) >= m) = I_p(m, n - m + 1)   (regularized incomplete beta).
    """

    S, K, T, r, sigma = (mp.mpf(str(v)) for v in (S, K, T, r, sigma))

    # Step size
    h = T / n

    # Geometric-center stretch parameter
    kappa = 2 * mp.log(K / S) / (sigma * mp.sqrt(n * T))
    lam = (kappa + mp.sqrt(kappa**2 + 4)) / 2

    # Up and down multipliers
    u = mp.exp(lam * sigma * mp.sqrt(h))
    d = mp.exp(-sigma * mp.sqrt(h) / lam)

    # Bond- and stock-numeraire probabilities
    pB = (mp.exp(r * h) - d) / (u - d)
    pS = pB * u * mp.exp(-r * h)

    # Unified strict exercise threshold
    m = n // 2 + 1

    # Regularized incomplete beta gives binomial upper tails
    tail_S = mp.betainc(m, n - m + 1, 0, pS, regularized=True)
    tail_B = mp.betainc(m, n - m + 1, 0, pB, regularized=True)

    return S * tail_S - K * mp.exp(-r * T) * tail_B