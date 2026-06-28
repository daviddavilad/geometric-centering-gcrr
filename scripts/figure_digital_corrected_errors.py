"""Half-node corrected digital errors for GCRR-GC and Joshi (2009).

Companion figure to scripts/figure_digital_errors.py: that figure shows
strict digital errors (with the O(1/sqrt(n)) even-phase central-atom
discrepancy visible); this figure shows that the half-node correction
removes the leading-order error and produces O(1/n) convergence in
both phases.

Per Section 9.3 of the GCRR-GC draft, the corrected upper tail is

    Psi^*_{N,eta}(p) = Psi_{N,eta}(p) + ((1-eta)/2) * C(2N, N) p^N (1-p)^N

which adds half the central atom in the even phase and leaves the odd
phase unchanged. The corrected digital satisfies the exact symmetry
Psi^*_{N,0}(1/2) = Psi^*_{N,1}(1/2) = 1/2 by construction. Empirical
behavior (verified in tests/test_digital.py):

  - Even-phase corrected error: O(1/n) with coefficient ~+0.02 at OTM
    (vs. O(1/sqrt(n)) with coefficient ~-0.38 for the strict digital).
  - Odd-phase corrected error: identical to strict (correction is zero
    for odd n), so also O(1/n) and tracking Joshi.

The visual contrast with Figure 4 is the empirical evidence for the
central-atom story of Section 9.3.

Generates two figures:
    figures/digital_corrected_errors_itm.{pdf,png}  -- K = 95  (ITM)
    figures/digital_corrected_errors_otm.{pdf,png}  -- K = 105 (OTM)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.black_scholes import bs_digital_strict
from gcrr.digital import digital_half_node_corrected_price
from gcrr.plotting import SCHEME_COLORS, SCHEME_LABELS, set_style


# --- Fixed parameters ---
S, T, r, sigma = 100.0, 0.5, 0.06, 0.20
N_VALUES = np.arange(25, 501)  # Chen §9.4: n=25..500 for figures

SCHEMES = [
    ("joshi_2009",  schemes.joshi_2009, 10),  # background
    ("gcrr_gc",     schemes.gcrr_gc,     3),  # foreground
]
# Format: (key, fn, marker_size)


def compute_phase_separated_corrected_errors(K, scheme_fn, n_values):
    """Compute corrected digital errors at each n; partition by *effective* parity.

    Mirrors compute_phase_separated_digital_errors but uses
    digital_half_node_corrected_price. For Joshi, every requested n maps
    to an odd effective n; the half-node correction is zero in the odd
    phase, so corrected_price == strict_price for Joshi.

    Note on the BS reference: the half-node corrected digital is still
    an approximation of the same Black-Scholes digital price, so we
    subtract bs_digital_strict as the reference target.
    """
    bs_ref = bs_digital_strict(S, K, T, r, sigma)

    n_eff_list, err_list = [], []
    for n in n_values:
        params = scheme_fn(S, K, T, r, sigma, int(n))
        price = digital_half_node_corrected_price(S, K, T, r, params)
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
    """Two-panel phase-separated corrected-digital-error figure at a given strike."""
    results = {
        key: compute_phase_separated_corrected_errors(K, fn, N_VALUES)
        for key, fn, _ in SCHEMES
    }

    # Side-by-side panels with INDEPENDENT y-axes -- same convention as Figure 4.
    fig, (ax_even, ax_odd) = plt.subplots(1, 2, figsize=(12, 5))

    for key, _, size in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        if len(out["n_even"]) > 0:
            ax_even.scatter(
                out["n_even"], out["err_even"],
                color=color, marker="o", s=size,
                alpha=0.85, edgecolors="none",
            )
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
    ax_even.set_ylabel(r"$D^*_{N,\eta} - D_{BS}$")

    # Legend on the odd panel (where both schemes appear)
    ax_odd.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        f"Half-node corrected digital errors: {moneyness_tag} "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={r}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"digital_corrected_errors_{moneyness_tag.lower()}"
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

    make_figure(K=95.0,  moneyness_tag="ITM", out_dir=out_dir)
    make_figure(K=105.0, moneyness_tag="OTM", out_dir=out_dir)


if __name__ == "__main__":
    main()
