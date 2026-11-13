"""SAINT-ETIENNE: network model (never saw the station) vs single-station trees (sensor_horizons).

Both prediction files are frozen outputs; errors are computed on the verifying hours
present in both, after checking the observations are identical (PROTOCOLE.md, references).

    python analysis/compare_single_station.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
VARIABLES = {"T": "T", "U": "U", "FF": "FF", "FXI": "FXI", "RAIN": "rain"}  # sensor_horizons name -> network name


def main():
    network = pd.read_csv(REPORTS / "station_network/predictions_saint_etienne.csv", parse_dates=["issue_time"])
    rows = []
    for single, name in VARIABLES.items():
        for h in (1, 24, 72):
            other = pd.read_csv(REPORTS / f"sensor_horizons/predictions_{single}_{h}h.csv", parse_dates=["valid_time"])
            ours = network[["issue_time", f"{name}_{h}h_pred", f"{name}_{h}h_obs"]].rename(
                columns={f"{name}_{h}h_pred": "network", f"{name}_{h}h_obs": "network_obs"})
            ours["valid_time"] = ours.issue_time + pd.Timedelta(hours=h)
            both = other.merge(ours, on="valid_time").dropna(
                subset=["observed", "network_obs", "network", "Capteurs seuls (arbres)", "Persistance"])
            if not np.allclose(both.observed, both.network_obs, atol=1e-3):
                raise ValueError(f"{single} +{h} h: observations differ between the two studies")
            score = (lambda e: float(np.mean(e ** 2))) if single == "RAIN" else (lambda e: float(np.mean(np.abs(e))))
            rows.append({"variable": name, "horizon_hours": h, "hours": len(both),
                         "metric": "brier" if single == "RAIN" else "mae",
                         "network_multi_station": score(both.network - both.observed),
                         "trees_single_station": score(both["Capteurs seuls (arbres)"] - both.observed),
                         "persistence": score(both.Persistance - both.observed)})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "station_network/comparaison_mono_station.csv", index=False, float_format="%.4f")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
