"""Closed-form asymptotic coefficients from the paper.

Implements the theoretical predictions that the numerical section tests:

    rho_B, rho_S        Appendix A.3, n^{-3/2} coefficients in eq. (4)
    A_eta               Theorem 3.2, eq. (8): phase-wise 1/n constant
    phase_gap           Corollary 3.3, eq. (9): A_1 - A_0
    digital_*           Section 4 and Appendix C.3: digital coefficients

Notation follows Appendix A.3:
    R = r T,  s = sigma sqrt(T),  x = log(K / S_0),  b = R - x,
    E = x^2 / (2 s^2) + s^2 / 6.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .black_scholes import d1_d2

def _appendix_a_quantities(S, K, T, r, sigma):
    """Return (R, s, x, b, E) as defined in Appendix A.3."""
    
    R = r * T
    s = sigma * np.sqrt(T)
    x = np.log(K / S)
    b = R - x
    E = (x**2 / (2 * s**2)) + s**2 / 6

    return R, s, x, b, E

def rho_B(S, K, T, r, sigma) -> float:
    """Appendix A.3:  rho_B = (1/s) [ (b^2 - x^2)/2 - s^4/24 - (b - s^2/2) E ]."""
    R, s, x, b, E = _appendix_a_quantities(S, K, T, r, sigma)

    # Compute and return rho_B
    return ((b**2 - x**2) / 2 - s**4 / 24 - (b - s**2 / 2) * E) / s

def rho_S(S, K, T, r, sigma) -> float:
    """Appendix A.3:  rho_S = (1/s) [ (x^2 - b^2)/2 + s^4/24 - (b + s^2/2) E ]."""
    R, s, x, b, E = _appendix_a_quantities(S, K, T, r, sigma)

    # Compute and return rho_S
    return ((x**2 - b**2) / 2 + s**4 / 24 - (b + s**2 / 2) * E) / s

def A_eta(S, K, T, r, sigma, eta: int) -> float:
    """Theorem 3.2, eq. (8):
    A_eta = S phi(d1) [ rho_S - rho_B + (d1^3 - d2^3)/4 + (2 eta - 1)(d1 - d2)/4 ].
    """

    if eta not in (0, 1):
        raise ValueError("eta must be 0 or 1")

    d1, d2 = d1_d2(S, K, T, r, sigma)

    rho_b = rho_B(S, K, T, r, sigma)
    rho_s = rho_S(S, K, T, r, sigma)

    numeraire_correction = rho_s - rho_b
    cubic_correction = (d1**3 - d2**3) / 4
    phase_correction = (2 * eta - 1) * (d1 - d2) / 4

    return S * norm.pdf(d1) * (numeraire_correction + cubic_correction + phase_correction)

def phase_gap(S, K, T, r, sigma) -> float:
    """Corollary 3.3, eq. (9):  A_1 - A_0 = (1/2) S phi(d1) sigma sqrt(T)."""
    d1, _ = d1_d2(S, K, T, r, sigma)

    return 0.5 * S * norm.pdf(d1) * sigma * np.sqrt(T)

def digital_strict_even_coef(S, K, T, r, sigma) -> float:
    """Section 4 / Appendix C.3: n^{-1/2} coefficient of the even-phase
    strict digital error,  D^str_{N,0} - D_BS = c / sqrt(n) + O(1/n),
    with  c = -e^{-rT} phi(d2).
    """
    _, d2 = d1_d2(S, K, T, r, sigma)

    return -np.exp(-r * T) * norm.pdf(d2)

def digital_hn_coef(S, K, T, r, sigma, eta: int) -> float:
    """Appendix C.3: 1/n coefficient of the half-node digital error,
    D^hn_{N,eta} - D_BS = c_eta / n + O(n^{-3/2}),
    with  c_eta = e^{-rT} phi(d2) [ rho_B + d2^3/4 + (2 eta - 1) d2/4 ].
    """

    if eta not in (0, 1):
        raise ValueError("eta must be 0 or 1")

    _, d2 = d1_d2(S, K, T, r, sigma)
    rho_b = rho_B(S, K, T, r, sigma)


    cubic_correction = d2**3 / 4
    phase_correction = ((2 * eta - 1) * d2) / 4

    return np.exp(-r * T) * norm.pdf(d2) * (rho_b + cubic_correction + phase_correction)