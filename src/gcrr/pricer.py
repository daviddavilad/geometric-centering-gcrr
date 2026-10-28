"""Generic binomial backward-induction pricer and tree-parameter interface.

Scheme-agnostic: every scheme produces a TreeParameters, and the pricer
consumes it identically. This is the same architecture as the public
option-pricing repo, generalized to carry the stretch parameter and
strike-location bookkeeping that the GCRR family needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class TreeParameters:
    """Parameters defining one binomial tree.

    Attributes
    ----------
    u, d : up/down multiplicative factors
    p    : bond-numeraire (risk-neutral) up-probability
    n    : number of periods
    dt   : time step T/n
    R    : one-period gross risk-free return e^{r*dt}
    lam  : stretch parameter (lambda); 1.0 recovers CRR. None if not applicable.
    """

    u: float
    d: float
    p: float
    n: int
    dt: float
    R: float
    lam: float | None = None

    def __post_init__(self):
        """Verify no-arbitrage condition d < R < u and p in (0, 1)."""
        if self.u <= self.d:
            raise ValueError(f"Require u > d, got u={self.u}, d={self.d}")
        if not (self.d < self.R < self.u):
            raise ValueError(
                f"No-arbitrage violated: require d < R < u, "
                f"got d={self.d}, R={self.R}, u={self.u}"
            )
        if not 0.0 < self.p < 1.0:
            raise ValueError(f"Require p in (0, 1), got p={self.p}")
        if self.dt <= 0:
            raise ValueError(f"Require dt > 0, got dt={self.dt}")
        if self.n <= 0:
            raise ValueError(f"Require n > 0, got n={self.n}")


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: TreeParameters,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    """Price an option by backward induction on the given tree.

    Vectorized over the j-index at each time step. O(n^2) time, O(n) space.
    """
    if S <= 0:
        raise ValueError(f"Spot S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")

    u, d, p, n, R = params.u, params.d, params.p, params.n, params.R

    # Terminal stock prices: S * u^j * d^(n-j), j = 0..n
    j = np.arange(n + 1)
    ST = S * (u**j) * (d ** (n - j))

    # Terminal payoff
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    disc = 1.0 / R
    is_american = exercise == "american"

    # Backward induction
    for step in range(n - 1, -1, -1):
        V = disc * (p * V[1 : step + 2] + (1.0 - p) * V[0 : step + 1])
        if is_american:
            jj = np.arange(step + 1)
            S_nodes = S * (u**jj) * (d ** (step - jj))
            if option_type == "call":
                intrinsic = np.maximum(S_nodes - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_nodes, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def two_tail_call_price(
    S, K, T, r, params: TreeParameters, m: int | None = None
) -> float:
    """European call via the exact two-numeraire tail representation
    (Proposition 3.1):  C = S Psi(p^S) - K e^{-rT} Psi(p^B),
    Psi(p) = P(Bin(n, p) >= m),  p^S = p^B u / R.

    m is the strict exercise threshold m_n = min{j : S u^j d^(n-j) > K}
    (eq. 1). If None, it is computed from the tree geometry. For GCRR-GC,
    m = N + 1 in both phases (Proposition 2.1). O(n) via binom.sf, versus
    O(n^2) for backward induction; the two agree to 1e-10
    (tests/test_pricer.py).
    """
    from .diagnostics import strike_location

    n = params.n

    # Bond- and stock-numeraire probabilities
    pB = params.p
    pS = pB * params.u / params.R

    # Strict exercise threshold
    if m is None:
        m = strike_location(S, K, params).m

    # Binomial upper tails P(Bin(n, p) >= m); sf(k) = P(X > k)
    tail_S = binom.sf(m - 1, n, pS)
    tail_B = binom.sf(m - 1, n, pB)

    return float(S * tail_S - K * np.exp(-r * T) * tail_B)
