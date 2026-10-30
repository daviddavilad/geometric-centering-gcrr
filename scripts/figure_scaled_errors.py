"""Phase-separated scaled errors for European call schemes.

For each scheme, plots n * (C_n - C_BS) against n, separated by parity
(even n vs odd n). Under phase-wise smooth convergence (Theorem 3.2),
each phase should approach a horizontal line; CRR (un-centered) shows
oscillation that doesn't flatten.

Generates two figures:
    figures_paper/scaled_errors_itm.{pdf,png}  -- K = 95  (ITM)
    figures_paper/scaled_errors_otm.{pdf,png}  -- K = 105 (OTM)

Test case: European call.
    S = 100, T = 0.5, r = 0.06, sigma = 0.20.
    (Primary non-ATM cases per Section 5.1.)

Schemes shown: CRR (gray baseline), Tian (1999), Joshi (2009), GCRR-GC.
GCRR-fixed and GCRR-XPC omitted for legibility of the headline figure.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.errors import phase_separated_errors
from gcrr.plotting import (
    plot_paper_series,
    set_style_paper,
)

matplotlib.use("Agg")  # same layout on every OS

# --- Fixed parameters ---
S, T, r, sigma = 100.0, 0.5, 0.06, 0.20
N_VALUES = np.arange(25, 501)  # Section 5.1: n=25..500 for figures

SCHEMES = [
    ("crr", schemes.crr),
    ("tian_1999", schemes.tian_1999),
    ("joshi_2009", schemes.joshi_2009),
    ("gcrr_gc", schemes.gcrr_gc),
]


def _compute_results(K: float):
    """Compute phase-separated scaled errors for each scheme."""
    return {
        key: phase_separated_errors(S, K, T, r, sigma, fn, N_VALUES, "call")
        for key, fn in SCHEMES
    }


def make_figure_paper(K: float, moneyness_tag: str, results: dict, out_dir: Path):
    """Paper figure: one panel per phase."""
    set_style_paper()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    ax_even, ax_odd = axes

    for key, _ in SCHEMES:
        out = results[key]
        plot_paper_series(ax_even, out.n_even, out.scaled_err_even, key)
        plot_paper_series(
            ax_odd, out.n_odd, out.scaled_err_odd, key, label="_nolegend_"
        )

    for ax in axes:
        ax.axhline(0, color="0.6", linewidth=0.5, zorder=0)
        ax.set_xlabel(r"$n$ (periods)")

    ax_even.set_title("Even $n$")
    ax_odd.set_title("Odd $n$")
    ax_even.set_ylabel(r"$n \cdot (C_n - C_{BS})$")

    handles, labels = ax_even.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    # fig.suptitle removed: parameters live in the Word caption
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    stem = f"scaled_errors_{moneyness_tag.lower()}"
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
        results = _compute_results(K)
        make_figure_paper(K, tag, results, out_dir)


if __name__ == "__main__":
    main()
