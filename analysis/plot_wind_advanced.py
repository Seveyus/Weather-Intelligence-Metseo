"""Render measured results of train_wind_advanced.py; no illustrative scores."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "reports" / "wind_advanced"


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    scores = pd.DataFrame(metrics["scores"])
    horizons = sorted(scores.horizon_hours.unique())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(15, 13), facecolor="#f8fafc")
    grid = fig.add_gridspec(3, 2, top=.88, bottom=.09, left=.075, right=.97, hspace=.56, wspace=.24)
    colors = {"Observations seules": "#94a3b8", "Météo brute retenue sur validation": "#d18a23", "Correction locale sélectionnée": "#007f86"}
    methods = {"Observations seules": "obs HGB15", "Correction locale sélectionnée": "Selected ensemble"}
    series = {label: [] for label in colors}
    for h in horizons:
        group = scores[scores.horizon_hours == h].set_index("method")
        for label, method in methods.items():
            series[label].append(group.loc[method])
        raw = metrics["horizons"][str(h)]["best_raw_nwp_from_validation"]
        series["Météo brute retenue sur validation"].append(group.loc[raw])
    for col, (metric, title) in enumerate((("mae_kmh", "MAE · erreur absolue moyenne"), ("rmse_kmh", "RMSE · sensible aux grandes erreurs"))):
        ax = fig.add_subplot(grid[0, col])
        for label, color in colors.items():
            y = [row[metric] for row in series[label]]
            ax.plot(horizons, y, "o-", color=color, lw=2.5, label=label)
            if label == "Correction locale sélectionnée":
                for h, value in zip(horizons,y):
                    ax.annotate(f"{value:.2f}",(h,value),xytext=(0,10),textcoords="offset points",ha="center",color=color,weight="bold")
        ax.set(title=title, xlabel="Échéance de la prévision (heures)",ylabel="Erreur (km/h)",xticks=horizons)
        ax.grid(axis="y",alpha=.2)
        if col==0:
            ax.legend(frameon=False,fontsize=9)
    ax = fig.add_subplot(grid[1, :])
    # First seven test days fixed before observing whether this looks favourable.
    predictions = pd.read_csv(OUT/"predictions_24h.csv",parse_dates=["valid_time","issue_time"]).set_index("valid_time")
    first = predictions.index.min().normalize()
    sample = predictions.loc[(predictions.index>=first)&(predictions.index<first+pd.Timedelta(days=7))]
    ax.plot(sample.index,sample.observed_kmh,label="Vent observé",color="#172554",lw=1.8)
    ax.plot(sample.index,sample.raw_nwp_kmh,label="Prévision météo brute (+24 h)",color="#d18a23",lw=1.5,alpha=.8)
    ax.plot(sample.index,sample.selected_kmh,label="Correction locale (+24 h)",color="#007f86",lw=1.8)
    ax.fill_between(sample.index,sample.lower80_kmh,sample.upper80_kmh,color="#007f86",alpha=.10,label="Intervalle nominal 80 %")
    ax.set(title="Exemple fixé à l'avance · première semaine du test · échéance +24 h",ylabel="Vent (km/h)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.legend(frameon=False,ncol=2,fontsize=10)
    ax.grid(axis="y",alpha=.2)
    ax = fig.add_subplot(grid[2, 0])
    for label,color in colors.items():
        ax.plot(horizons,[row["high_wind_mae_kmh"] for row in series[label]],"o-",color=color,lw=2.5)
    threshold = metrics["horizons"][str(horizons[0])]["training_p90_wind_kmh"]
    ax.set(title=f"Vents plus élevés (≥ {threshold:.1f} km/h)",xlabel="Échéance (heures)",ylabel="MAE (km/h)",xticks=horizons)
    ax.grid(axis="y",alpha=.2)
    ax = fig.add_subplot(grid[2, 1])
    ax.axis("off")
    lines=["Comparaison PCA : résultats de validation", "Janvier–septembre 2025 · avant le test final", ""]
    selection=json.loads((OUT/"selection.json").read_text())
    for h in horizons:
        detail=selection[str(h)]
        plain=detail["validation_scores"]["NWP residual Ridge"]
        pca=min((detail["validation_scores"][k] for k in ("NWP residual PCA95 + Ridge","NWP residual PCA99 + Ridge")), key=lambda m:m["rmse_kmh"])
        lines += [f"+{h} h : RMSE Ridge {plain['rmse_kmh']:.2f} / PCA {pca['rmse_kmh']:.2f} km/h"]
    lines += ["", "Le modèle final est choisi sur validation.", "Une PCA n'est conservée que si elle aide."]
    ax.text(0,1,"\n".join(lines),va="top",fontsize=11,color="#334155",linespacing=1.6)
    fig.text(.075,.962,"Prévoir le vent à +24, +48 et +72 heures",fontsize=25,weight="bold",color="#0f172a")
    fig.text(.075,.93,"SAINT-ÉTIENNE · test final janvier–début février 2026 · vitesse du vent en km/h",fontsize=13,color="#475569")
    fig.text(.075,.903,"Modèles et assemblage choisis sur 2025 ; aucune sélection sur les erreurs du test final.",fontsize=10,color="#475569")
    fig.text(.075,.042,"Archives météo de 48/72/96 h pour prévoir à 24/48/72 h : marge de 24 h contre les délais de publication.",fontsize=10,color="#475569")
    fig.text(.075,.022,"Test court, hivernal. Chaque point temporel est une prévision distincte ; les intervalles restent à valider sur d'autres saisons.",fontsize=10,color="#475569")
    fig.savefig(OUT/"resultats_24_48_72h.png",dpi=170)
    fig.savefig(OUT/"resultats_24_48_72h.svg")
    plt.close(fig)
    print(OUT/"resultats_24_48_72h.png")


if __name__ == "__main__":
    main()
