"""Black-Scholes European option prices and the d1, d2 helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_digital_strict(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price of a strict digital call paying 1 if S_T > K.

    Equals e^{-rT} * Phi(d_2), since P(S_T > K) = Phi(d_2) under the
    risk-neutral measure when the asset follows GBM. The strict-vs-weak
    distinction (S_T > K vs S_T >= K) is irrelevant in continuous time
    because P(S_T = K) = 0 under GBM.
    """
    _, d2 = d1_d2(S, K, T, r, sigma)
    return float(np.exp(-r * T) * norm.cdf(d2))