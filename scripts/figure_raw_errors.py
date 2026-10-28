"""Raw call-price errors for three GCRR-family schemes.

For each scheme, plots the raw error C_n - C_BS against n (no scaling).
Companion to scripts/figure_scaled_errors.py: the scaled-error figure
shows convergence rates by dividing by 1/n; this figure shows the
raw error magnitudes.

The contrast drawn in Section 5.2: a stretch parameter alone
does not produce smooth convergence; the stretch must be chosen by
the geometric-center condition. The fixed-stretch GCRR scheme (with
lambda = 1.1) is included as the control. CRR (lambda = 1) is the
baseline. GCRR-GC (lambda chosen by the geometric-center equation)
is the proposed construction.

Generates two figures:
    figures_paper/raw_errors_itm.{pdf,png}  -- K = 95  (ITM)
    figures_paper/raw_errors_otm.{pdf,png}  -- K = 105 (OTM)

Set WRITE_COLOR = True to also generate legacy color versions in
figures/ (used by talks/slides).

Test case: European call.
    S = 100, T = 0.5, r = 0.06, sigma = 0.20.
    (Primary non-ATM cases per Section 5.1.)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.plotting import (
    SCHEME_COLORS,
    SCHEME_LABELS,
    plot_paper_series,
    set_style,
    set_style_paper,
)
from gcrr.pricer import binomial_price

matplotlib.use("Agg")  # same layout on every OS

# --- Toggle: also generate legacy color versions? ---
WRITE_COLOR = False

# --- Fixed parameters ---
S, T, r, sigma = 100.0, 0.5, 0.06, 0.20
N_VALUES = np.arange(25, 501)  # Section 5.1: n=25..500 for figures

# In this figure: three schemes, ordered so GCRR-GC is drawn last (foreground)
SCHEMES = [
    ("crr", schemes.crr),
    ("gcrr_fixed", schemes.gcrr_fixed),
    ("gcrr_gc", schemes.gcrr_gc),
]


def _compute_errors(K: float):
    """Compute raw errors for each scheme at each n."""
    C_BS = bs_call(S, K, T, r, sigma)
    results = {}
    for key, scheme_fn in SCHEMES:
        errs = np.empty(len(N_VALUES))
        for i, n in enumerate(N_VALUES):
            params = scheme_fn(S, K, T, r, sigma, int(n))
            price = binomial_price(S, K, T, r, params, "call", "european")
            errs[i] = price - C_BS
        results[key] = errs
    return results


def make_figure_paper(K: float, moneyness_tag: str, results: dict, out_dir: Path):
    """Paper-style monochrome figure: line + shape marker, Chung-Shih-like."""
    set_style_paper()
    fig, ax = plt.subplots()

    for key, _ in SCHEMES:
        plot_paper_series(ax, N_VALUES, results[key], key)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_xlabel(r"$n$ (periods)")
    ax.set_ylabel(r"$C_n - C^{\mathrm{BS}}$")
    ax.legend(loc="best")

    # fig.suptitle removed: parameters live in the Word caption
    fig.tight_layout(rect=(0, 0, 1, 1))

    stem = f"raw_errors_{moneyness_tag.lower()}"
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def make_figure_color(K: float, moneyness_tag: str, results: dict, out_dir: Path):
    """Legacy color figure: preserved for talks/slides.

    Uses set_style() and the halo-style scatter markers of the original.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    sizes = {"crr": 20, "gcrr_fixed": 10, "gcrr_gc": 3}
    for key, _ in SCHEMES:
        ax.scatter(
            N_VALUES,
            results[key],
            color=SCHEME_COLORS[key],
            marker="o",
            s=sizes[key],
            alpha=0.85,
            label=SCHEME_LABELS[key],
            edgecolors="none",
        )

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlabel(r"$n$ (periods)")
    ax.set_ylabel(r"$C_n - C^{\mathrm{BS}}$")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"Raw call-price errors: {moneyness_tag} "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={r}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"raw_errors_{moneyness_tag.lower()}"
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
        results = _compute_errors(K)
        make_figure_paper(K, tag, results, paper_dir)

    # Optional: legacy color output
    if WRITE_COLOR:
        color_dir = project_root / "figures"
        color_dir.mkdir(exist_ok=True)
        for K, tag in [(95.0, "ITM"), (105.0, "OTM")]:
            results = _compute_errors(K)
            make_figure_color(K, tag, results, color_dir)


if __name__ == "__main__":
    main()
