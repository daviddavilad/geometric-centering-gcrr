"""Matplotlib styling for the GCRR-GC paper figures.

Every paper figure calls set_style_paper() and draws each scheme with
plot_paper_series (or plot_paper_series_by_phase), so all figures share
one style: Okabe-Ito colors, one marker shape per scheme, STIX fonts,
and the journal's rules on line width and transparency.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Color palette for the six schemes.
# Order: CRR, GCRR-fixed, GCRR-XPC, Tian (1999), Joshi (2009), GCRR-GC.
# GCRR-GC is the new scheme -- it gets the visually distinct dark red.

SCHEME_LABELS = {
    "crr": "CRR",
    "gcrr_fixed": "GCRR (fixed)",
    "gcrr_xpc": "GCRR-XPC",
    "tian_1999": "Tian (1999)",
    "joshi_2009": "Joshi (2009)",
    "gcrr_gc": "GCRR-GC",
}


# --- Paper style ---
#
# Color (Okabe-Ito, colorblind-safe) that still reads in grayscale print:
# each scheme has its own marker shape, and markers sit on every
# MARKER_STEP-th point, offset per scheme so they do not overlap. A thin
# line through all points shows the shape. GCRR-GC is drawn heavier and
# on top. Journal rules: lines at least 0.3 pt, no transparency.

MARKER_STEP = 6
MARKER_SIZE = 3  # markers on every MARKER_STEP-th point, with a line
MARKER_SIZE_DENSE = 2  # markers on every point, no line (Figure 3)

SCHEME_STYLES_PAPER = {
    "crr": {
        "color": "#7F7F7F",  # gray for the baseline
        "marker": "o",
        "linewidth": 0.6,
        "marker_offset": 0,
    },
    "gcrr_fixed": {
        "color": "#E69F00",  # orange: fixed-stretch control
        "marker": "D",
        "linewidth": 0.8,
        "marker_offset": 1,
    },
    "tian_1999": {
        "color": "#009E73",  # green
        "marker": "^",
        "linewidth": 0.8,
        "marker_offset": 2,
        "break_at_jumps": True,  # Tian's tilt resets, so its errors jump
    },
    "joshi_2009": {
        "color": "#D55E00",  # vermillion
        "marker": "x",
        "linewidth": 0.8,
        "marker_offset": 3,
    },
    "gcrr_gc": {
        "color": "#0072B2",  # blue: the proposed scheme
        "marker": "s",
        "linewidth": 1.2,
        "marker_offset": 8,
        "zorder": 4,
    },
}


def set_style_paper():
    """Apply the paper style (color, STIX fonts, Chung-Shih-like axes)."""
    
    mpl.rcParams.update(
        {
            # Figure
            "figure.figsize": (7, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Embed fonts as TrueType (journal and arXiv requirement)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Fonts: STIX (Times-style) ships with matplotlib, so figures match on every machine
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
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
            "mathtext.fontset": "stix",
        }
    )


def _break_at_jumps(x, y, factor=20.0):
    """Insert NaN before each jump so a line does not bridge it.

    A jump is a step larger than `factor` times the median step.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    steps = np.abs(np.diff(y))
    if steps.size == 0:
        return x, y
    jumps = np.flatnonzero(steps > factor * np.median(steps)) + 1
    return np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)


def plot_paper_series(ax, x, y, scheme_key, label=None, lines=True):
    """Plot one scheme in paper style and return its Line2D.

    lines=True: a thin line through every point, markers on every
    MARKER_STEP-th point. lines=False: small markers on every point and
    no line, for step-like data (Figure 3).
    """
    style = SCHEME_STYLES_PAPER[scheme_key]
    label = SCHEME_LABELS[scheme_key] if label is None else label

    if lines:
        if style.get("break_at_jumps", False):
            x, y = _break_at_jumps(x, y)
        linestyle = "-"
        markevery = (style["marker_offset"], MARKER_STEP)
        markersize = MARKER_SIZE
    else:
        linestyle = "none"
        markevery = None
        markersize = MARKER_SIZE_DENSE

    (line,) = ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=linestyle,
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=markersize,
        markevery=markevery,
        markeredgewidth=0.6,
        zorder=style.get("zorder", 2),
        label=label,
    )
    return line


def plot_paper_series_by_phase(ax, n, y, scheme_key, label=None):
    """Plot even and odd n as two lines in one style, one legend entry.

    A single line through all n would zigzag between the two phases.
    """
    n = np.asarray(n)
    y = np.asarray(y)
    even = n % 2 == 0
    line = plot_paper_series(ax, n[even], y[even], scheme_key, label=label)
    plot_paper_series(ax, n[~even], y[~even], scheme_key, label="_nolegend_")
    return line
