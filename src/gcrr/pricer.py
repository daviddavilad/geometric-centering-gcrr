"""Generic binomial backward-induction pricer and tree-parameter interface.

Scheme-agnostic: every scheme produces a TreeParameters, and the pricer
consumes it identically. This is the same architecture as the public
option-pricing repo, generalized to carry the stretch parameter and
strike-location bookkeeping that the GCRR family needs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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
    ST = S * (u ** j) * (d ** (n - j))

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
            S_nodes = S * (u ** jj) * (d ** (step - jj))
            if option_type == "call":
                intrinsic = np.maximum(S_nodes - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_nodes, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])
