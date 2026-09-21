"""Shared chart style so every figure in the report looks like it came from one team.

    from src.utils.plotting import set_style, save_fig, BLUE, ORANGE
    set_style()
    ... make a chart with matplotlib / seaborn ...
    save_fig("target_balance")        # -> reports/figures/target_balance.png (report-ready)

Colors are a colorblind-safe pair: blue = "eligible", orange = "not eligible".
Rules of thumb used everywhere: one message per chart, no dual axes, thin marks,
quiet gridlines, and the title states the finding (not just "Chart of X").
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.data.fetch_fema import REPO_ROOT

FIG_DIR = REPO_ROOT / "reports" / "figures"

BLUE = "#2a78d6"      # eligible / primary series
ORANGE = "#eb6834"    # not eligible / second series
GRAY = "#898781"      # neutral / reference lines
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"]   # light -> dark blue


def set_style() -> None:
    """Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "text.color": INK,
        "legend.frameon": False,
    })


def save_fig(name: str) -> None:
    """Save the current figure into reports/figures/ (folder is created if missing)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png")
