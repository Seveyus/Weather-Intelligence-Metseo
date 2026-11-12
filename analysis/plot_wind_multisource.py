"""Render measured results of benchmark_wind_multisource.py; no illustrative scores."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "reports" / "wind_multisource"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots 1-3 (validated all-pairs); raw NWP is a neutral reference, dashed.
SERIES = {"A": ("A · données v1", "#2a78d6"), "B": ("B · + modèles NWP", "#eb6834"),
          "C": ("C · + stations voisines", "#1baf7a")}
RAW = ("Meilleure NWP brute", "#8a8984")
SEASONS = ("Feb–Mar", "Apr–Jun", "Jul–Aug", "Sep–Oct")
SEASON_LABELS = ("févr.–mars", "avr.–juin", "juil.–août", "sept.–oct.")


def line(ax, x, y, color, label, style="-"):
    # B and C nearly coincide, so identity comes from the shared legend, not end labels.
    ax.plot(x, y, style, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=label)


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    scores = pd.DataFrame(metrics["scores"]).set_index(["horizon_hours", "method"])
    horizons = sorted(int(h) for h in metrics["horizons"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 11.5})
    fig = plt.figure(figsize=(14, 10.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, top=.85, bottom=.07, left=.07, right=.97, hspace=.38, wspace=.18)
    first = metrics["horizons"][str(horizons[0])]
    fig.suptitle("Vent à Saint-Étienne · benchmark multi-sources sur un test jamais utilisé", x=.07, ha="left",
                 fontsize=15, weight="bold")
    fig.text(.07, .935, f"{first['test_n']} heures cibles communes, {first['test_start'][:10]} → {first['test_end'][:10]} · "
             "finalistes choisis sur 5 plis de validation avant le test · erreurs en km/h", color=INK_2)

    ax = fig.add_subplot(grid[0, 0])
    for key, (label, color) in SERIES.items():
        y = [scores.loc[(h, metrics["horizons"][str(h)]["finalists"][key]), "mae_kmh"] for h in horizons]
        line(ax, horizons, y, color, label)
    y = [scores.loc[(h, metrics["horizons"][str(h)]["best_raw_nwp_from_validation"]), "mae_kmh"] for h in horizons]
    line(ax, horizons, y, RAW[1], RAW[0], "--")
    ax.set(title="MAE par échéance", xlabel="Échéance (heures)", ylabel="MAE (km/h)", xticks=horizons,
           xlim=(horizons[0] - 4, horizons[-1] + 4))
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.07, .905), ncol=4,
               frameon=False, fontsize=10, labelcolor=INK_2)
    ax.grid(axis="y", color=GRID, lw=.8)

    ax = fig.add_subplot(grid[0, 1])
    horizon = 48
    chosen = metrics["horizons"][str(horizon)]
    x = range(len(SEASONS))
    for key, (label, color) in SERIES.items():
        y = [scores.loc[(horizon, chosen["finalists"][key]), f"mae_{s}"] for s in SEASONS]
        line(ax, x, y, color, label)
    y = [scores.loc[(horizon, chosen["best_raw_nwp_from_validation"]), f"mae_{s}"] for s in SEASONS]
    line(ax, x, y, RAW[1], RAW[0], "--")
    ax.set(title=f"MAE par saison du test · échéance +{horizon} h", ylabel="MAE (km/h)", xticks=list(x),
           xticklabels=SEASON_LABELS, xlim=(-.3, len(SEASONS) - .3))
    ax.grid(axis="y", color=GRID, lw=.8)

    ax = fig.add_subplot(grid[1, :])
    # Fixed in advance: the first seven days of the test at +24 h, whatever they show.
    predictions = pd.read_csv(OUT / "predictions_24h.csv", parse_dates=["valid_time"]).set_index("valid_time")
    start = predictions.index.min().normalize()
    sample = predictions.loc[predictions.index < start + pd.Timedelta(days=7)]
    selected_key = next(k for k, v in metrics["horizons"]["24"]["finalists"].items()
                        if v == metrics["horizons"]["24"]["selected"])
    color = SERIES[selected_key][1]
    ax.fill_between(sample.index, sample.lower80_kmh, sample.upper80_kmh, color=color, alpha=.12, lw=0,
                    label="Intervalle nominal 80 %")
    ax.plot(sample.index, sample.raw_nwp_kmh, "--", color=RAW[1], lw=1.6, label=RAW[0])
    ax.plot(sample.index, sample.selected_kmh, color=color, lw=2,
            label=f"Modèle retenu ({SERIES[selected_key][0]})")
    ax.plot(sample.index, sample.observed_kmh, color=INK, lw=1.6, label="Vent observé")
    ax.set(title="Première semaine du test · échéance +24 h (choisie à l'avance)", ylabel="Vent (km/h)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9.5, labelcolor=INK_2)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"resultats_multisource.{suffix}", dpi=150, facecolor=SURFACE)
    print(f"Saved {OUT / 'resultats_multisource.png'}")


if __name__ == "__main__":
    main()
