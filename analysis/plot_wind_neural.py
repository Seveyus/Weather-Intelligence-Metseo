"""Plot measured neural experiments, including the reused retrospective test.

Run after selection for validation/learning plots and again after evaluation for
the final comparison. Missing phases are reported and skipped; no measurements
are invented. ``--report-dir`` permits use with an isolated experiment directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "wind_neural"
HORIZONS = (24, 48, 72)
CANDIDATES = ("bigru", "bilstm", "bigru_attention", "history_ablation")
LABELS = {
    "bigru": "BiGRU",
    "bilstm": "BiLSTM",
    "bigru_attention": "BiGRU + attention",
    "history_ablation": "Historique séquentiel neutralisé",
}
COLORS = {
    "bigru": "#007f86",
    "bilstm": "#6653aa",
    "bigru_attention": "#c45a22",
    "history_ablation": "#64748b",
}
METHODS = {
    "Tree baseline": ("Baseline arbres", "#6653aa"),
    "Neural ensemble": ("Ensemble neuronal", "#007f86"),
    "Raw NWP": ("Prévisions météo brutes", "#c48719"),
}
INK = "#172554"
MUTED = "#475569"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, directory: Path, stem: str) -> list[Path]:
    paths = [directory / f"{stem}.{suffix}" for suffix in ("png", "svg")]
    fig.savefig(paths[0], dpi=165, facecolor=fig.get_facecolor())
    fig.savefig(paths[1], facecolor=fig.get_facecolor())
    # Matplotlib's SVG output includes whitespace-only lines; keep git diffs clean.
    svg = paths[1].read_text(encoding="utf-8")
    paths[1].write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                        encoding="utf-8")
    plt.close(fig)
    return paths


def style_axes(ax):
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def finite_values(values, description: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite values in {description}")
    return array


def validation_plot(directory: Path) -> list[Path]:
    source = directory / "validation_candidates.json"
    if not source.exists():
        print("Validation : résultats absents, graphique différé.")
        return []
    candidates = read_json(source)
    names = [name for name in CANDIDATES if name in candidates]
    if not names:
        raise ValueError("No recognized candidates in validation_candidates.json")
    selection_path = directory / "selection.json"
    selection = read_json(selection_path) if selection_path.exists() else None
    selected = selection["selected"] if selection else None
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor="#f8fafc")
    fig.subplots_adjust(top=.76, bottom=.24, left=.065, right=.975, wspace=.20)
    for ax, metric, title in zip(axes, ("mae_kmh", "rmse_kmh"),
                               ("MAE · erreur absolue moyenne", "RMSE · grandes erreurs pénalisées")):
        for name in names:
            values = finite_values([candidates[name]["validation_scores"][str(h)][metric]
                                    for h in HORIZONS], f"{name} / {metric}")
            label = LABELS[name] + (" · retenu" if name == selected else "")
            ax.plot(HORIZONS, values, marker="o", color=COLORS[name],
                    linestyle="--" if name == "history_ablation" else "-",
                    linewidth=2.8 if name == selected else 1.8, label=label)
        ax.set(title=title, xlabel="Échéance (heures)", ylabel="Erreur (km/h)", xticks=HORIZONS)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .115),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.065, .94, "Quel réseau apprend la meilleure correction locale ?",
             fontsize=22, weight="bold", color=INK)
    fig.text(.065, .886, "Validation 2025 · mêmes observations et prévisions météo · moyenne de deux graines",
             fontsize=12, color=MUTED)
    if selection:
        status = f"Réseau retenu : {LABELS.get(selected, selected)} · choix effectué avant le calcul des résultats neuronaux de 2026."
    else:
        status = "Comparaison partielle : sélection pas encore figée."
    fig.text(.065, .832, status, fontsize=10.5, color=MUTED)
    fig.text(.065, .075, "Critère de sélection : moyenne des MAE et RMSE normalisées par la climatologie, sur les trois échéances.",
             fontsize=10, color=MUTED)
    fig.text(.065, .04, "Ablation : la séquence est neutralisée ; le contexte conserve des variables historiques. Ce contrôle n’est pas éligible au choix final.",
             fontsize=9.5, color=MUTED)
    return save_figure(fig, directory, "validation_architectures")


def learning_plot(directory: Path) -> list[Path]:
    runs = [read_json(path) for path in sorted(directory.glob("history_*.json"))]
    if not runs:
        print("Apprentissage : historiques absents, graphique différé.")
        return []
    names = [name for name in CANDIDATES if any(run["candidate"] == name for run in runs)]
    if not names:
        raise ValueError("No recognized candidates in history files")
    fig, axes = plt.subplots(len(names), 2, squeeze=False,
                             figsize=(15, 3.25 * len(names) + 2.1), facecolor="#f8fafc")
    fig.subplots_adjust(top=1 - 1.65 / fig.get_figheight(), bottom=.10,
                        left=.07, right=.97, hspace=.60, wspace=.21)
    for row, name in enumerate(names):
        candidate_runs = sorted((run for run in runs if run["candidate"] == name), key=lambda x: x["seed"])
        for index, run in enumerate(candidate_runs):
            history = run["history"]
            if not history:
                raise ValueError(f"Empty training history: {name}, seed {run['seed']}")
            epochs = [item["epoch"] for item in history]
            linestyle = "-" if index % 2 == 0 else "--"
            color = COLORS[name] if index % 2 == 0 else INK
            for col, key in enumerate(("train_loss", "validation_objective")):
                values = finite_values([item[key] for item in history], f"{name}, seed {run['seed']}, {key}")
                axes[row, col].plot(epochs, values, linestyle=linestyle, color=color,
                                     linewidth=1.9, label=f"Graine {run['seed']}")
                best_epoch = run["best_epoch"]
                best = next((item[key] for item in history if item["epoch"] == best_epoch), None)
                if best is None:
                    raise ValueError(f"Best epoch absent from history: {name}, seed {run['seed']}")
                axes[row, col].scatter([best_epoch], [best], s=48, color=color,
                                        marker="D", zorder=4)
            label = LABELS[name]
            for col, subtitle in enumerate(("perte d’entraînement", "critère de validation")):
                axes[row, col].set(title=f"{label} · {subtitle}", xlabel="Époque",
                                    ylabel="Perte sans unité" if col == 0 else "Erreur / climatologie")
                style_axes(axes[row, col])
        axes[row, 0].legend(frameon=False, fontsize=9)
        axes[row, 1].legend(frameon=False, fontsize=9)
    fig.text(.07, 1 - .48 / fig.get_figheight(), "Apprentissage des réseaux · détail de chaque graine",
             fontsize=23, weight="bold", color=INK)
    fig.text(.07, 1 - .91 / fig.get_figheight(),
             "Sélection sur 2025 · les losanges indiquent l’époque retenue pour chaque entraînement",
             fontsize=12, color=MUTED)
    fig.text(.07, .052, "Perte : erreur quadratique + 0,2 × erreur absolue, après division de l’erreur par 5 km/h.",
             fontsize=10, color=MUTED)
    fig.text(.07, .026, "Validation : MAE/RMSE relatives à la climatologie, moyennées sur +24/+48/+72 h. Les courbes utilisent deux échelles différentes.",
             fontsize=10, color=MUTED)
    return save_figure(fig, directory, "apprentissage")


def read_predictions(directory: Path) -> tuple[dict[int, pd.DataFrame], pd.DatetimeIndex]:
    frames = {}
    required = ["observed_kmh", "selected_kmh", "raw_nwp_kmh", "neural_kmh",
                "neural_lower80_kmh", "neural_upper80_kmh"]
    common = None
    for horizon in HORIZONS:
        frame = pd.read_csv(directory / f"predictions_{horizon}h.csv",
                            parse_dates=["valid_time", "issue_time"]).set_index("valid_time").sort_index()
        if frame.empty or frame.index.has_duplicates:
            raise ValueError(f"Empty or duplicate forecast timestamps for +{horizon}h")
        finite_values(frame[required].to_numpy(), f"predictions +{horizon}h")
        leads = frame.index - pd.DatetimeIndex(frame.issue_time)
        if not (leads == pd.Timedelta(hours=horizon)).all():
            raise ValueError(f"Incorrect issue/valid time difference for +{horizon}h")
        if (frame.neural_lower80_kmh > frame.neural_kmh).any() or (frame.neural_upper80_kmh < frame.neural_kmh).any():
            raise ValueError(f"Prediction interval excludes point forecast for +{horizon}h")
        frames[horizon] = frame
        common = frame.index if common is None else common.intersection(frame.index)
    if common.empty:
        raise ValueError("No common timestamps across the three forecast horizons")
    common = common.sort_values()
    reference = frames[HORIZONS[0]].loc[common, "observed_kmh"].to_numpy()
    for horizon in HORIZONS[1:]:
        if not np.allclose(reference, frames[horizon].loc[common, "observed_kmh"], rtol=0, atol=1e-6):
            raise ValueError("Observed winds differ across horizons at common timestamps")
    first_day = common.min().normalize()
    sample = common[(common >= first_day) & (common < first_day + pd.Timedelta(days=7))]
    return frames, sample


def results_plot(directory: Path) -> list[Path]:
    source = directory / "metrics.json"
    if not source.exists():
        print("Évaluation rétrospective : résultats absents, graphique différé.")
        return []
    metrics = read_json(source)
    scores = pd.DataFrame(metrics["scores"]).set_index(["method", "horizon_hours"])
    if scores.index.has_duplicates:
        raise ValueError("Duplicate method/horizon pairs in metrics.json")
    frames, sample = read_predictions(directory)
    fig = plt.figure(figsize=(15, 16.5), facecolor="#f8fafc")
    grid = fig.add_gridspec(4, 2, top=.865, bottom=.125, left=.07, right=.97,
                           hspace=.67, wspace=.23)
    for col, (metric, title) in enumerate((("mae_kmh", "MAE · erreur absolue moyenne"),
                                          ("rmse_kmh", "RMSE · grandes erreurs pénalisées"))):
        ax = fig.add_subplot(grid[0, col])
        for method, (label, color) in METHODS.items():
            values = finite_values([scores.loc[(method, h), metric] for h in HORIZONS], metric)
            ax.plot(HORIZONS, values, "o-", linewidth=2.4, color=color, label=label)
            if method == "Neural ensemble":
                for horizon, value in zip(HORIZONS, values):
                    ax.annotate(f"{value:.2f}", (horizon, value), xytext=(0, 8),
                                textcoords="offset points", ha="center", color=color, weight="bold")
        ax.margins(y=.17)
        ax.set(title=title, xlabel="Échéance (heures)", ylabel="Erreur (km/h)", xticks=HORIZONS)
        style_axes(ax)
        if col == 0:
            ax.legend(frameon=False, fontsize=9)
    # Identical axes keep visual amplitudes comparable between lead times.
    wind_max = max(float(frame.loc[sample, ["observed_kmh", "selected_kmh", "raw_nwp_kmh",
                                           "neural_upper80_kmh"]].max().max())
                   for frame in frames.values())
    for row, horizon in enumerate(HORIZONS, start=1):
        ax = fig.add_subplot(grid[row, :])
        data = frames[horizon].loc[sample]
        ax.fill_between(data.index, data.neural_lower80_kmh, data.neural_upper80_kmh,
                        color=METHODS["Neural ensemble"][1], alpha=.12, label="Intervalle neuronal nominal 80 %")
        ax.plot(data.index, data.observed_kmh, color=INK, linewidth=1.8, label="Vent observé")
        ax.plot(data.index, data.selected_kmh, color=METHODS["Tree baseline"][1],
                linewidth=1.3, alpha=.8, label="Baseline arbres")
        ax.plot(data.index, data.raw_nwp_kmh, color=METHODS["Raw NWP"][1],
                linewidth=1.1, alpha=.8, label="Prévisions météo brutes")
        ax.plot(data.index, data.neural_kmh, color=METHODS["Neural ensemble"][1],
                linewidth=1.8, label="Ensemble neuronal")
        ax.set(title=f"Échéance +{horizon} h · même première semaine commune", ylabel="Vent (km/h)")
        ax.set_ylim(0, wind_max * 1.12)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.set_xlabel("Date observée (UTC)")
        style_axes(ax)
        if row == 1:
            ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left")
    selected = LABELS.get(metrics["selected"], metrics["selected"])
    dates = f"{sample.min():%d/%m/%Y} au {sample.max():%d/%m/%Y}"
    counts = ", ".join(f"+{h} h : {metrics['horizons'][str(h)]['test_n']}" for h in HORIZONS)
    fig.text(.07, .96, "Prévision du vent · réseau neuronal face à la baseline",
             fontsize=23, weight="bold", color=INK)
    fig.text(.07, .934, "SAINT-ÉTIENNE · test 2026 réutilisé · comparaison rétrospective", fontsize=13, color=MUTED)
    fig.text(.07, .910, f"{selected}, ensemble de deux graines · choix du réseau et des époques sur 2025", fontsize=11, color=MUTED)
    fig.text(.07, .888, f"MAE/RMSE sur tous les cas de la baseline ({counts}). Traces : {dates}.", fontsize=10.5, color=MUTED)
    fig.text(.07, .074, "Traces fixées aux sept premiers jours communs. Chaque point résulte d’une nouvelle émission (date observée − échéance).",
             fontsize=10, color=MUTED)
    coverage = " ; ".join(f"+{h} h : {metrics['horizons'][str(h)]['interval_coverage']:.1%}" for h in HORIZONS)
    fig.text(.07, .053, f"Intervalles calibrés fin 2025 ; couverture observée sur le test : {coverage}. Le niveau nominal 80 % n’est pas une garantie.",
             fontsize=9.5, color=MUTED)
    fig.text(.07, .032, "Test court, hivernal et déjà consulté : ces résultats ne démontrent pas une généralisation à de nouvelles saisons.",
             fontsize=10, color=MUTED)
    return save_figure(fig, directory, "resultats_24_48_72h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    if not args.report_dir.is_dir():
        parser.error(f"Experiment directory does not exist: {args.report_dir}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": MUTED,
                         "text.color": INK, "axes.titlesize": 11.5})
    outputs = validation_plot(args.report_dir)
    outputs += learning_plot(args.report_dir)
    outputs += results_plot(args.report_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
