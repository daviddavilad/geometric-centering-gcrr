"""Strict digital errors for GCRR-GC and Joshi (2009).

For each scheme, plots the raw error D^>_{N,eta} - D_BS against n,
separated by parity. The strict digital pays 1 if S_T > K, 0 otherwise,
priced as e^{-rT} * Psi_{N,eta}(p_n^B).

Section 9.3 of the GCRR-GC draft predicts that the strict digital
exhibits an "even-phase central-atom discrepancy of order n^{-1/2}".
The empirical pattern (verified in scripts/check_phase_gap is the
pricing analogue; here for digitals) is even stronger:

  - Even-phase GCRR-GC error: O(1/sqrt(n)) with empirical coefficient
    ~-0.378 at OTM (K=105). This is the central-atom discrepancy.
  - Odd-phase GCRR-GC error: O(1/n), about three orders of magnitude
    smaller than even-phase at n=12800.
  - Joshi (odd n only): tracks GCRR-GC odd phase, also O(1/n).
  - CRR: oscillates pseudo-randomly within each phase.

Figure 5 (forthcoming) will show the half-node corrected digital,
which removes the even-phase central-atom term and is expected to
restore O(1/n) convergence in both phases.

Generates one figure (Chen §9.4 base case):
    figures/digital_errors_atm.{pdf,png}  -- K = 100 (ATM)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.black_scholes import bs_digital_strict
from gcrr.digital import digital_strict_price
from gcrr.plotting import SCHEME_COLORS, SCHEME_LABELS, set_style


# --- Fixed parameters (Chen §9.4 base case) ---
S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
N_VALUES = np.arange(25, 1001)

SCHEMES = [
    ("joshi_2009",  schemes.joshi_2009, 10),  # background
    ("gcrr_gc",     schemes.gcrr_gc,     3),  # foreground
]
# Format: (key, fn, marker_size)


def compute_phase_separated_digital_errors(K, scheme_fn, n_values):
    """Compute digital errors at each n; partition by *effective* parity.

    Mirrors the bookkeeping in errors.phase_separated_errors: Joshi forces
    n odd internally, so we use params.n (effective n) for the parity
    partition, not the requested n. Duplicates from Joshi's even-n bumping
    are de-duplicated by keeping the first occurrence.
    """
    bs_ref = bs_digital_strict(S, K, T, r, sigma)

    n_eff_list, err_list = [], []
    for n in n_values:
        params = scheme_fn(S, K, T, r, sigma, int(n))
        price = digital_strict_price(S, K, T, r, params)
        n_eff_list.append(params.n)
        err_list.append(price - bs_ref)

    n_eff = np.asarray(n_eff_list)
    errs = np.asarray(err_list)

    # De-duplicate: keep first occurrence of each effective n.
    _, unique_idx = np.unique(n_eff, return_index=True)
    unique_idx = np.sort(unique_idx)
    n_eff = n_eff[unique_idx]
    errs = errs[unique_idx]

    even_mask = (n_eff % 2 == 0)
    odd_mask = ~even_mask

    return {
        "n_even": n_eff[even_mask],
        "n_odd": n_eff[odd_mask],
        "err_even": errs[even_mask],
        "err_odd": errs[odd_mask],
    }


def make_figure(K, moneyness_tag, out_dir):
    """Two-panel phase-separated digital-error figure at a given strike."""
    results = {
        key: compute_phase_separated_digital_errors(K, fn, N_VALUES)
        for key, fn, _ in SCHEMES
    }

    # Side-by-side panels with INDEPENDENT y-axes -- the phase asymmetry
    # (~3 orders of magnitude at large n) makes shared y-axes unreadable.
    fig, (ax_even, ax_odd) = plt.subplots(1, 2, figsize=(12, 5))

    for key, _, size in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        # Even panel: GCRR-GC only (Joshi is odd-only)
        if len(out["n_even"]) > 0:
            ax_even.scatter(
                out["n_even"], out["err_even"],
                color=color, marker="o", s=size,
                alpha=0.85, edgecolors="none",
            )
        # Odd panel: both schemes
        if len(out["n_odd"]) > 0:
            ax_odd.scatter(
                out["n_odd"], out["err_odd"],
                color=color, marker="o", s=size,
                alpha=0.85, label=label, edgecolors="none",
            )

    for ax in (ax_even, ax_odd):
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlabel(r"$n$ (periods)")

    ax_even.set_title(r"Even $n$")
    ax_odd.set_title(r"Odd $n$")
    ax_even.set_ylabel(r"$D^>_{N,\eta} - D_{BS}$")

    # Legend on the odd panel (where both schemes appear)
    ax_odd.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        f"Strict digital errors: {moneyness_tag} "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={r}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"digital_errors_{moneyness_tag.lower()}"
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def main():
    set_style()
    out_dir = Path(__file__).parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)

    make_figure(K=100.0, moneyness_tag="ATM", out_dir=out_dir)


if __name__ == "__main__":
    main()
