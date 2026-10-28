r"""Strike-location diagnostics: Delta_n and m_n - N across four schemes.

Two stacked panels showing the geometric structure of the strike's
position in the terminal log-lattice:

Top panel: Delta_n = 1 - 2*theta_n, the continuous centered location
coordinate. Reference lines at Delta = 0 (geometric midpoint) and
Delta = +/-1 (strike on a node).

Bottom panel: m_n - N, the integer-valued strict upper threshold
relative to the central index N. Proposition 2.1 implies that
GCRR-GC produces the "unified strict threshold j >= N+1" -- i.e.,
m_n - N = 1 for all n, regardless of parity.

Generates:
    figures_paper/delta_and_threshold.{pdf,png}

Set WRITE_COLOR = True to also generate legacy color version in
figures/ (used by talks/slides).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.diagnostics import strike_location_sweep
from gcrr.plotting import (
    SCHEME_LABELS,
    plot_paper_series,
    set_style,
    set_style_paper,
)

# --- Toggle: also generate legacy color version? ---
WRITE_COLOR = False

# --- Parameters (Section 5.1 primary non-ATM case) ---
S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.06, 0.20  # OTM
N_VALUES = np.arange(25, 501)  # Section 5.1: n=25..500 for figures

SCHEMES = [
    ("gcrr_gc", schemes.gcrr_gc),
    ("joshi_2009", schemes.joshi_2009),
    ("crr", schemes.crr),
    ("tian_1999", schemes.tian_1999),
]


def _compute_results():
    """Compute strike-location diagnostics for each scheme."""
    return {
        key: strike_location_sweep(S, K, T, r, sigma, fn, N_VALUES)
        for key, fn in SCHEMES
    }


def make_figure_paper(results: dict, out_dir: Path):
    """Paper-style monochrome figure: two stacked scatter panels."""
    set_style_paper()

    N_arr = (N_VALUES - N_VALUES % 2) // 2

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top panel: Delta_n
    from gcrr.plotting import SCHEME_LABELS

    for key, _ in SCHEMES:
        out = results[key]
        if key == "joshi_2009":
            ax_top.plot(
                N_VALUES,
                out["Delta"],
                marker="x",
                linestyle="none",
                color="#7F7F7F",
                markersize=2.0,
                markeredgewidth=0.2,
                label=SCHEME_LABELS[key],
            )
        elif key == "tian_1999":
            ax_top.plot(
                N_VALUES,
                out["Delta"],
                marker="v",
                linestyle="none",
                color="#7F7F7F",
                markersize=1.5,
                markeredgewidth=0.2,
                fillstyle="none",
                label=SCHEME_LABELS[key],
            )
        else:
            plot_paper_series(ax_top, N_VALUES, out["Delta"], key)

    ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.6, zorder=0)
    ax_top.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_top.axhline(
        -1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--"
    )
    ax_top.set_ylabel(r"$\Delta_n = 1 - 2\theta_n$")
    ax_top.set_ylim(-1.15, 1.15)

    # Bottom panel: m_n - N
    for key, _ in SCHEMES:
        out = results[key]
        threshold = out["m"] - N_arr
        if key == "joshi_2009":
            ax_bot.plot(
                N_VALUES,
                threshold,
                marker="x",
                linestyle="none",
                color="#7F7F7F",
                markersize=2.0,
                markeredgewidth=0.2,
                label="_nolegend_",
            )
        elif key == "tian_1999":
            ax_bot.plot(
                N_VALUES,
                threshold,
                marker="v",
                linestyle="none",
                color="#7F7F7F",
                markersize=1.5,
                markeredgewidth=0.2,
                fillstyle="none",
                label="_nolegend_",
            )
        else:
            plot_paper_series(ax_bot, N_VALUES, threshold, key, label="_nolegend_")

    ax_bot.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_bot.set_xlabel(r"$n$ (periods)")
    ax_bot.set_ylabel(r"$m_n - N$")

    ax_top.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    # fig.suptitle removed: parameters live in the Word caption
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    out_pdf = out_dir / "delta_and_threshold.pdf"
    out_png = out_dir / "delta_and_threshold.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def make_figure_color(results: dict, out_dir: Path):
    """Legacy color figure: preserved for talks/slides."""
    set_style()

    N_arr = (N_VALUES - N_VALUES % 2) // 2

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for key, _ in SCHEMES:
        out = results[key]
        if key == "joshi_2009":
            ax_top.plot(
                N_VALUES,
                out["Delta"],
                marker="x",
                linestyle="none",
                color="#7F7F7F",
                markersize=2.0,
                markeredgewidth=0.2,
                label=SCHEME_LABELS[key],
            )
        elif key == "tian_1999":
            ax_top.plot(
                N_VALUES,
                out["Delta"],
                marker="v",
                linestyle="none",
                color="#7F7F7F",
                markersize=1.5,
                markeredgewidth=0.2,
                fillstyle="none",
                label=SCHEME_LABELS[key],
            )
        else:
            plot_paper_series(ax_top, N_VALUES, out["Delta"], key)

    ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.5, zorder=0)
    ax_top.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_top.axhline(
        -1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--"
    )
    ax_top.set_ylabel(r"$\Delta_n = 1 - 2\theta_n$")
    ax_top.set_ylim(-1.15, 1.15)

    for key, _ in SCHEMES:
        out = results[key]
        if key == "joshi_2009":
            ax_bot.plot(
                N_VALUES,
                out["m"] - N_arr,
                marker="x",
                linestyle="none",
                color="#7F7F7F",
                markersize=2.0,
                markeredgewidth=0.2,
                label="_nolegend_",
            )
        elif key == "tian_1999":
            ax_bot.plot(
                N_VALUES,
                out["m"] - N_arr,
                marker="v",
                linestyle="none",
                color="#7F7F7F",
                markersize=1.5,
                markeredgewidth=0.2,
                fillstyle="none",
                label="_nolegend_",
            )
        else:
            plot_paper_series(
                ax_bot, N_VALUES, out["m"] - N_arr, key, label="_nolegend_"
            )

    ax_bot.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_bot.set_xlabel(r"$n$ (periods)")
    ax_bot.set_ylabel(r"$m_n - N$")

    ax_top.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle(
        f"Strike-location diagnostics  "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.85, 0.97))

    out_pdf = out_dir / "delta_and_threshold.pdf"
    out_png = out_dir / "delta_and_threshold.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def main():
    project_root = Path(__file__).parent.parent

    # Paper (monochrome) output
    paper_dir = project_root / "figures_paper"
    paper_dir.mkdir(exist_ok=True)

    results = _compute_results()
    make_figure_paper(results, paper_dir)

    # Optional: legacy color output
    if WRITE_COLOR:
        color_dir = project_root / "figures"
        color_dir.mkdir(exist_ok=True)
        make_figure_color(results, color_dir)


if __name__ == "__main__":
    main()
