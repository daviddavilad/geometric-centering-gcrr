"""Matplotlib styling for GCRR-GC figures.

Centralizes color palette, font sizes, and rcParams so that all figures
share a consistent look. Call `set_style()` at the top of any figure
script before doing matplotlib work.

Color palette: six schemes need to be distinguishable in print and on
screen, including for readers with deuteranopia. Uses a manually-curated
subset of ColorBrewer's qualitative palettes ('Set1' / 'Set2'-derived),
with hue and lightness chosen so adjacent series in a legend don't blur.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette for the six schemes.
# Order: CRR, GCRR-fixed, GCRR-XPC, Tian (1999), Joshi (2009), GCRR-GC.
# GCRR-GC is the new scheme -- it gets the visually distinct dark red.
SCHEME_COLORS = {
    "crr":           "#666666",  # neutral gray -- the baseline
    "gcrr_fixed":    "#1f77b4",  # blue -- control
    "gcrr_xpc":      "#2ca02c",  # green -- even-only predecessor
    "tian_1999":     "#ff7f0e",  # orange -- alternative centered
    "joshi_2009":    "#9467bd",  # purple -- alternative centered, odd-only
    "gcrr_gc":       "#d62728",  # red -- the new scheme
}

SCHEME_LABELS = {
    "crr":           "CRR",
    "gcrr_fixed":    "GCRR (fixed)",
    "gcrr_xpc":      "GCRR-XPC",
    "tian_1999":     "Tian (1999)",
    "joshi_2009":    "Joshi (2009)",
    "gcrr_gc":       "GCRR-GC",
}


def set_style():
    """Apply consistent matplotlib styling across all figures."""
    mpl.rcParams.update({
        # Figure
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Fonts
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Axes
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Math text
        "text.usetex": False,
        "mathtext.fontset": "cm",
    })


# --- Paper (monochrome) style ---
#
# Paper figures follow the monochrome style of Chung and Shih (2007):
#   - all-black lines
#   - line style differentiation (solid, dashed, dotted)
#   - marker shape differentiation (circle, square, triangle, cross)
#   - sparse marker frequency (every ~10th point)
#
# GCRR-GC is the paper's contribution and gets slightly heavier/darker
# treatment: linewidth 1.8 vs 1.3 for the others.

SCHEME_STYLES_PAPER = {
    "crr":        {"marker": "o", "linestyle": "none", "color": "#9F9F9F", "linewidth": 0.1, "markersize": 2, "fillstyle": "full", "markeredgewidth": 0.1},
    "gcrr_fixed": {"marker": "^", "linestyle": "none", "color": "#5B5B5B", "linewidth": 0.1, "markersize": 2, "fillstyle": "full", "markeredgewidth": 0.1},
    "gcrr_gc":    {"marker": "s", "linestyle": "none", "color": "#000000", "linewidth": 0.1, "markersize": 2, "fillstyle": "full", "markeredgewidth": 0.2, "markevery": 1},
    "tian_1999":  {"marker": "v", "linestyle": "none", "color": "#5B5B5B", "linewidth": 0.1, "markersize": 2, "fillstyle": "full", "markeredgewidth": 0.1},
    "joshi_2009": {"marker": "x", "linestyle": "none", "color": "#444444", "linewidth": 0.1, "markersize": 2, "fillstyle": "none", "markeredgewidth": 0.5, "markevery": 1},
}

MARKER_EVERY_PAPER = 25


def set_style_paper():
    """Apply paper-style (monochrome, Chung-Shih inspired) styling.

    Call this at the top of a figure script INSTEAD of set_style() when
    generating paper-ready figures. Not compatible with set_style() in
    the same script call.
    """
    mpl.rcParams.update({
        # Figure
        "figure.figsize": (7, 4.5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Fonts (Chung-Shih uses serif, close to Times)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Axes -- Chung-Shih shows all four spines, no grid or very faint
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,

        # Lines and markers -- overridden per-series via SCHEME_STYLES_PAPER
        "lines.linewidth": 1.3,
        "lines.markersize": 5,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,

        # Math text
        "text.usetex": False,
        "mathtext.fontset": "cm",
    })


def plot_paper_series(ax, x, y, scheme_key, label=None, markevery=None):
    """Plot one series in paper style. Returns Line2D."""
    style = SCHEME_STYLES_PAPER[scheme_key]
    label = label if label is not None else SCHEME_LABELS[scheme_key]

    # Per-scheme markevery from the style dict, unless overridden
    effective_markevery = markevery if markevery is not None else style.get("markevery", None)

    return ax.plot(
        x, y,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
        markersize=style["markersize"],
        fillstyle=style["fillstyle"],
        markeredgewidth=style["markeredgewidth"],
        markevery=effective_markevery,
        label=label,
    )