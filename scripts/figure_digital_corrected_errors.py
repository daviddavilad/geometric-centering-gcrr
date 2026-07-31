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
    figures_paper/digital_corrected_errors_itm.{pdf,png}  -- K = 95  (ITM)
    figures_paper/digital_corrected_errors_otm.{pdf,png}  -- K = 105 (OTM)

Set WRITE_COLOR = True to also generate legacy color versions in
figures/ (used by talks/slides).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.black_scholes import bs_digital_strict
from gcrr.digital import digital_half_node_corrected_price
from gcrr.plotting import (
    SCHEME_COLORS,
    SCHEME_LABELS,
    set_style,
    set_style_paper,
    plot_paper_series,
)


# --- Toggle: also generate legacy color versions? ---
WRITE_COLOR = False

# --- Fixed parameters ---
S, T, r, sigma = 100.0, 0.5, 0.06, 0.20
N_VALUES = np.arange(25, 501)  # Chen §9.4: n=25..500 for figures

SCHEMES = [
    ("joshi_2009", schemes.joshi_2009),
    ("gcrr_gc",    schemes.gcrr_gc),
]


def compute_phase_separated_corrected_errors(K, scheme_fn, n_values):
    """Compute corrected digital errors at each n; partition by *effective* parity."""
    bs_ref = bs_digital_strict(S, K, T, r, sigma)

    n_eff_list, err_list = [], []
    for n in n_values:
        params = scheme_fn(S, K, T, r, sigma, int(n))
        price = digital_half_node_corrected_price(S, K, T, r, params)
        n_eff_list.append(params.n)
        err_list.append(price - bs_ref)

    n_eff = np.asarray(n_eff_list)
    errs = np.asarray(err_list)

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


def _compute_results(K):
    """Compute phase-separated corrected digital errors for each scheme."""
    return {
        key: compute_phase_separated_corrected_errors(K, fn, N_VALUES)
        for key, fn in SCHEMES
    }


def make_figure_paper(K, moneyness_tag, results, out_dir):
    """Paper-style monochrome figure: two panels with independent y-axes."""
    set_style_paper()

    fig, (ax_even, ax_odd) = plt.subplots(1, 2, figsize=(9, 4))

    for key, _ in SCHEMES:
        out = results[key]
        if len(out["n_even"]) > 0:
            plot_paper_series(ax_even, out["n_even"], out["err_even"], key, label="_nolegend_")
        if len(out["n_odd"]) > 0:
            plot_paper_series(ax_odd, out["n_odd"], out["err_odd"], key)

    for ax in (ax_even, ax_odd):
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_xlabel(r"$n$ (periods)")

    ax_even.set_title(r"Even $n$")
    ax_odd.set_title(r"Odd $n$")
    ax_even.set_ylabel(r"$D^*_{N,\eta} - D_{BS}$")

    ax_odd.legend(loc="best", frameon=True)

    fig.suptitle(
        f"Half-node corrected digital errors ({moneyness_tag}): "
        f"$S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={r}$, $\\sigma={sigma}$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"digital_corrected_errors_{moneyness_tag.lower()}"
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def make_figure_color(K, moneyness_tag, results, out_dir):
    """Legacy color figure: preserved for talks/slides."""
    set_style()

    fig, (ax_even, ax_odd) = plt.subplots(1, 2, figsize=(12, 5))

    sizes = {"joshi_2009": 10, "gcrr_gc": 3}
    for key, _ in SCHEMES:
        out = results[key]
        if len(out["n_even"]) > 0:
            ax_even.scatter(
                out["n_even"], out["err_even"],
                color=SCHEME_COLORS[key], marker="o", s=sizes[key],
                alpha=0.85, edgecolors="none",
            )
        if len(out["n_odd"]) > 0:
            ax_odd.scatter(
                out["n_odd"], out["err_odd"],
                color=SCHEME_COLORS[key], marker="o", s=sizes[key],
                alpha=0.85, label=SCHEME_LABELS[key], edgecolors="none",
            )

    for ax in (ax_even, ax_odd):
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlabel(r"$n$ (periods)")

    ax_even.set_title(r"Even $n$")
    ax_odd.set_title(r"Odd $n$")
    ax_even.set_ylabel(r"$D^*_{N,\eta} - D_{BS}$")

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
    project_root = Path(__file__).parent.parent

    # Paper (monochrome) output
    paper_dir = project_root / "figures_paper"
    paper_dir.mkdir(exist_ok=True)

    for K, tag in [(95.0, "ITM"), (105.0, "OTM")]:
        results = _compute_results(K)
        make_figure_paper(K, tag, results, paper_dir)

    # Optional: legacy color output
    if WRITE_COLOR:
        color_dir = project_root / "figures"
        color_dir.mkdir(exist_ok=True)
        for K, tag in [(95.0, "ITM"), (105.0, "OTM")]:
            results = _compute_results(K)
            make_figure_color(K, tag, results, color_dir)


if __name__ == "__main__":
    main()
