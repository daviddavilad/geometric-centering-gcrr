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
