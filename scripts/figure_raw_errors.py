"""Raw call-price errors for three GCRR-family schemes.

For each scheme, plots the raw error C_n - C_BS against n (no scaling).
Companion to scripts/figure_scaled_errors.py: the scaled-error figure
multiplies each error by n to expose the 1/n rate; this figure shows
the raw error magnitudes.

The contrast drawn in Section 5.2: a stretch parameter alone
does not produce smooth convergence; the stretch must be chosen by
the geometric-center condition. The fixed-stretch GCRR scheme (with
lambda = 1.1) is included as the control. CRR (lambda = 1) is the
baseline. GCRR-GC (lambda chosen by the geometric-center equation)
is the proposed construction.

Generates two figures:
    figures_paper/raw_errors_itm.{pdf,png}  -- K = 95  (ITM)
    figures_paper/raw_errors_otm.{pdf,png}  -- K = 105 (OTM)

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
    plot_paper_series_by_phase,
    set_style_paper,
)
from gcrr.pricer import binomial_price

matplotlib.use("Agg")  # same layout on every OS

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
    """Paper figure: one line per phase for each scheme."""
    set_style_paper()
    fig, ax = plt.subplots()

    for key, _ in SCHEMES:
        plot_paper_series_by_phase(ax, N_VALUES, results[key], key)

    ax.axhline(0, color="0.6", linewidth=0.5, zorder=0)
    ax.set_xlabel(r"$n$ (periods)")
    ax.set_ylabel(r"$C_n - C_{BS}$")
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


def main():
    out_dir = Path(__file__).parent.parent / "figures_paper"
    out_dir.mkdir(exist_ok=True)

    for K, tag in [(95.0, "ITM"), (105.0, "OTM")]:
        make_figure_paper(K, tag, _compute_errors(K), out_dir)


if __name__ == "__main__":
    main()
