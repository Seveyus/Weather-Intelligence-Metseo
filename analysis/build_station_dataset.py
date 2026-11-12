#!/usr/bin/env python3
"""Build a compact multi-station dataset of the planned station's sensors.

Reads the Météo-France hourly department exports downloaded by
fetch_meteofrance.py and keeps, for every station, only the variables the
station will measure: T, U (SHT30), PSTAT (BMP280), FF, DD, FXI (anemometer and
vane), RR1 (tipping-bucket gauge). Each station becomes one float32 array on a
complete hourly grid (missing = NaN, nothing filled), plus a metadata table.

    python analysis/build_station_dataset.py
    python analysis/build_station_dataset.py --input-dir data/meteofrance_france --output-dir data/stations
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("T", "U", "PSTAT", "FF", "DD", "FXI", "RR1")
UNITS = {"T": "°C", "U": "%", "PSTAT": "hPa", "FF": "km/h", "DD": "°", "FXI": "km/h", "RR1": "mm"}
EPOCH = pd.Timestamp("2000-01-01")
# Physical plausibility only; values outside are set missing, never clipped.
# Only stations usable by the station-network protocol keep their series on disk.
REQUIRED = ("T", "U", "FF", "DD", "RR1")
MIN_HOURS = 17_532  # two years
LIMITS = {"T": (-40, 50), "U": (0, 100), "PSTAT": (600, 1100), "FF": (0, 250), "DD": (0, 360),
          "FXI": (0, 350), "RR1": (0, 150)}


def read_department(paths):
    columns = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH", *VARIABLES]
    parts = []
    for path in paths:
        for chunk in pd.read_csv(path, sep=";", usecols=columns, chunksize=500_000,
                                 dtype={"NUM_POSTE": str, "AAAAMMJJHH": str, "NOM_USUEL": str}):
            # Rows with no sensor value at all carry no information.
            parts.append(chunk.loc[chunk[list(VARIABLES)].notna().any(axis=1)])
    return pd.concat(parts, ignore_index=True) if parts else None


def station_array(frame):
    time = pd.to_datetime(frame.AAAAMMJJHH, format="%Y%m%d%H")
    hours = ((time - EPOCH) // pd.Timedelta(hours=1)).to_numpy()
    if len(np.unique(hours)) != len(hours):
        frame = frame.assign(_h=hours).drop_duplicates("_h", keep="last")
        hours = frame._h.to_numpy()
    values = frame[list(VARIABLES)].to_numpy(np.float32)
    for i, name in enumerate(VARIABLES):
        if name in ("FF", "FXI"):
            values[:, i] *= 3.6
        low, high = LIMITS[name]
        values[(values[:, i] < low) | (values[:, i] > high), i] = np.nan
    start, end = hours.min(), hours.max()
    grid = np.full((end - start + 1, len(VARIABLES)), np.nan, np.float32)
    grid[hours - start] = values
    return int(start), grid


def build_department(department, paths, out):
    """Write the department's eligible series; return its metadata rows (cached for resume)."""
    cache = out / "meta" / f"{department}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["sources"] == {p.name: p.stat().st_size for p in paths}:
            return department, cached["rows"]
    raw = read_department(paths)
    rows = []
    for station, frame in ([] if raw is None else raw.groupby("NUM_POSTE")):
        start, grid = station_array(frame.sort_values("AAAAMMJJHH"))
        present = np.isfinite(grid)
        if not present.any():
            continue
        counts = dict(zip(VARIABLES, present.sum(axis=0).tolist()))
        saved = all(counts[name] >= MIN_HOURS for name in REQUIRED)
        if saved:
            np.save(out / "series" / f"{station}.npy", grid)
        last = frame.iloc[-1]
        row = {"station": station, "name": last.NOM_USUEL, "department": department,
               "lat": float(last.LAT), "lon": float(last.LON), "alt": float(last.ALTI),
               "start": str(EPOCH + pd.Timedelta(hours=start)),
               "end": str(EPOCH + pd.Timedelta(hours=start + len(grid) - 1)), "hours": len(grid),
               "series_saved": saved}
        row.update({f"{name}_present": int(present[:, i].sum()) for i, name in enumerate(VARIABLES)})
        rows.append(row)
    cache.write_text(json.dumps({"sources": {p.name: p.stat().st_size for p in paths}, "rows": rows}))
    return department, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/meteofrance_france")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/stations")
    parser.add_argument("--workers", type=int, default=3, help="Departments built in parallel (~2 GB RAM each)")
    args = parser.parse_args()
    out = args.output_dir
    (out / "series").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    by_department = defaultdict(list)
    for path in sorted(args.input_dir.glob("H_*_*.csv.gz")):
        by_department[path.name.split("_")[1]].append(path)
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(build_department, d, paths, out) for d, paths in sorted(by_department.items())]
        for job in as_completed(jobs):
            department, rows = job.result()
            results[department] = rows
            print(f"Department {department}: {len(rows)} stations ({len(results)}/{len(jobs)} departments)", flush=True)
    meta = [row for department in sorted(results) for row in results[department]]
    table = pd.DataFrame(meta)
    table.to_csv(out / "stations.csv", index=False)
    (out / "README.json").write_text(json.dumps({
        "variables": list(VARIABLES), "units": UNITS, "epoch": str(EPOCH),
        "layout": "series/<station>.npy = float32[hours, variables]; row 0 is stations.csv 'start'",
        "conversions": "FF and FXI from m/s to km/h; out-of-range values set to NaN",
        "series_saved_rule": f"{', '.join(REQUIRED)} each present >= {MIN_HOURS} h (other stations listed only)",
        "limits": LIMITS, "missing": "NaN; nothing interpolated or filled",
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for paths in by_department.values() for p in paths},
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"Listed {len(table)} stations; saved {int(table.series_saved.sum())} series to {out}", flush=True)


if __name__ == "__main__":
    main()
