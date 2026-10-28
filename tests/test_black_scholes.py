"""Tests for pure Black-Scholes identities and reference-price behavior.

These tests are scheme-agnostic: they verify identities of the
Black-Scholes formula itself, not the convergence of any binomial
scheme to Black-Scholes (those live in test_schemes.py).
"""
from math import exp, log, sqrt

from scipy.stats import norm


def test_bs_phi_identity():
    """Verify S_0 * phi(d_1) = X * exp(-rT) * phi(d_2) across parameters.

    This is a fundamental Black-Scholes identity relating the two
    Gaussian densities evaluated at d_1 and d_2.

    The identity follows from d_1 - d_2 = sigma * sqrt(T) and direct
    algebraic manipulation of the log-normal density; it plays a key
    role in the delta/digital relationship and in the phase-gap
    identity of Corollary 3.3.
    """
    cases = [
        # (S, X, T, r, sigma) — spans ATM/OTM/ITM and the WAND panel extremes
        (100.0, 100.0, 1.00, 0.05, 0.20),   # ATM
        (100.0, 105.0, 0.50, 0.06, 0.20),   # OTM, primary non-ATM per Section 5.1
        (100.0,  95.0, 0.50, 0.06, 0.20),   # ITM, primary non-ATM per Section 5.1
        ( 40.0,  35.0, 0.25, 0.06, 0.20),   # WAND case 1
        ( 40.0,  45.0, 0.50, 0.06, 0.40),   # WAND case 8
    ]

    for S, X, T, r, sigma in cases:
        d1 = (log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        lhs = S * norm.pdf(d1)
        rhs = X * exp(-r * T) * norm.pdf(d2)

        assert abs(lhs - rhs) < 1e-12, (
            f"BS identity failed at S={S}, X={X}, T={T}, r={r}, sigma={sigma}: "
            f"S*phi(d1)={lhs}, X*exp(-rT)*phi(d2)={rhs}"
        )
