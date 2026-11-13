"""Small multiples of the station-network test scores (station_network.py and station_trees.py)."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "reports" / "station_network"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "Réseau MLP (2 graines)": dict(color="#2a78d6", ls="-", marker="o"),
    "Arbres groupés": dict(color="#eb6834", ls="-", marker="s"),
    "Persistance": dict(color="#8a8984", ls="--", marker=None),
    "Cycle diurne (J-1)": dict(color="#8a8984", ls=":", marker=None),
}
TITLES = {"T": "Température (°C)", "U": "Humidité (%)", "PSTAT": "Pression station (hPa)", "FF": "Vent moyen (km/h)",
          "FXI": "Rafale max (km/h)", "u": "Vent est-ouest u (km/h)", "v": "Vent nord-sud v (km/h)",
          "rain": "Pluie dans l'heure (Brier)"}


def table(role):
    network = json.loads((OUT / "metrics.json").read_text())["results"][role]
    trees = json.loads((OUT / "trees_metrics.json").read_text())["results"][role]
    rows = [dict(method=method, **row) for source in (network, trees) for method, scores in source["scores"].items()
            for row in scores]
    frame = pd.DataFrame(rows)
    frame["score"] = frame.brier.fillna(frame.mae) if "brier" in frame else frame.mae
    return frame, network


def main(role="test_stations"):
    frame, info = table(role)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), facecolor=SURFACE)
    fig.subplots_adjust(top=.80, bottom=.08, left=.05, right=.985, hspace=.42, wspace=.25)
    fig.suptitle("Capteurs seuls, appris sur 660 stations Météo-France · erreur sur des stations jamais vues",
                 x=.05, ha="left", fontsize=15, weight="bold")
    fig.text(.05, .895, f"{info['stations']} stations de test · heures d'émission à partir du 2024-01-04 "
             f"({info['issue_times']:,} heures) · mêmes heures pour toutes les méthodes · plus bas = meilleur · "
             "échéances en échelle logarithmique".replace(",", " "), color=INK_2)
    for ax, target in zip(axes.flat, TITLES):
        for method, style in STYLE.items():
            part = frame.loc[(frame.method == method) & (frame.target == target)].sort_values("horizon_hours")
            ax.plot(part.horizon_hours, part.score, lw=2, ms=5, label=method, **style)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 6, 12, 24, 72], ["+1", "+3", "+6", "+12", "+24", "+72 h"])
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.set_facecolor(SURFACE)
        ax.set_ylim(bottom=0)
        ax.set_title(TITLES[target])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.05, .87), ncol=4, frameon=False)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"comparatif_{role}.{suffix}", dpi=130, facecolor=SURFACE)
    print(f"Saved {OUT / f'comparatif_{role}.png'}")


if __name__ == "__main__":
    main()
