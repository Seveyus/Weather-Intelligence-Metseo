"""Small multiples of sensor_horizon_study.py results; measured scores only."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "reports" / "sensor_horizons"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Two categorical slots for the methods compared; neutral, dashed references.
STYLE = {
    "Capteurs seuls (arbres)": dict(color="#2a78d6", ls="-", marker="o"),
    "Open-Meteo brut (moy. 6 modèles)": dict(color="#eb6834", ls="-", marker="s"),
    "Persistance": dict(color="#8a8984", ls="--", marker=None),
    "Moyenne saisonnière": dict(color="#8a8984", ls=":", marker=None),
}


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    scores = pd.DataFrame(metrics["scores"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 11})
    variables = list(metrics["variables"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=SURFACE)
    fig.subplots_adjust(top=.83, bottom=.08, left=.06, right=.98, hspace=.42, wspace=.25)
    first = next(iter(metrics["variables"].values()))
    fig.suptitle("Station de Saint-Étienne · jusqu'où les capteurs seuls prévoient-ils ?", x=.06, ha="left",
                 fontsize=15, weight="bold")
    fig.text(.06, .905, f"Test {first['test_start'][:10]} → {first['test_end'][:10]} · mêmes heures pour toutes "
             "les méthodes d'une variable · plus bas = meilleur · échelle des échéances logarithmique", color=INK_2)
    for ax, key in zip(axes.flat, variables):
        info = metrics["variables"][key]
        subset = scores.loc[scores.variable == key]
        metric = "brier" if info["unit"] == "Brier" else "mae"
        for method, style in STYLE.items():
            rows = subset.loc[subset.method == method].sort_values("horizon_hours")
            if rows.empty:
                continue
            ax.plot(rows.horizon_hours, rows[metric], lw=2, ms=7, mec=SURFACE, mew=1.5, label=method, **style)
        ax.set_xscale("log")
        ticks = sorted(subset.horizon_hours.unique())
        ax.set_xticks(ticks, [f"+{h}" for h in ticks])
        ax.minorticks_off()
        ax.set_title(info["label"])
        ax.set_ylabel("Score de Brier" if metric == "brier" else f"MAE ({info['unit']})")
        ax.set_xlabel("Échéance (heures)")
        ax.grid(axis="y", color=GRID, lw=.8)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.06, .885),
               ncol=4, frameon=False, labelcolor=INK_2)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"horizons_capteurs.{suffix}", dpi=150, facecolor=SURFACE)
    print(f"Saved {OUT / 'horizons_capteurs.png'}")


if __name__ == "__main__":
    main()
