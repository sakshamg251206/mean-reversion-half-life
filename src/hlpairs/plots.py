"""Shared figure style so every chart in the paper reads as one system.
Colours: the dataviz skill's validated categorical order (light mode), used in fixed
order; GREY is the neutral for benchmarks / reference series."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8a8a8a"
INK, INK2 = "#0b0b0b", "#52514e"
FIG_DIR = Path("figures")


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#c8c7c2", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False, "figure.constrained_layout.use": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table(df, name: str, floatfmt: str = "%.3f", **kw):
    Path("tables").mkdir(exist_ok=True)
    df.to_csv(f"tables/{name}.csv")
    df.to_latex(f"tables/{name}.tex", float_format=floatfmt, **kw)
