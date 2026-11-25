"""Shared Matplotlib / Seaborn style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config as C  # noqa: E402


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "axes.titleweight": "bold",
                         "axes.titlesize": 12, "savefig.bbox": "tight"})


def save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def cat_colors(categories):
    return [C.PALETTE.get(c, "#999999") for c in categories]
