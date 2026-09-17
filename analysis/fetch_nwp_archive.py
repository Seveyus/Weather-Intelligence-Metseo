#!/usr/bin/env python3
"""Download real archived NWP forecasts with explicit daily lead offsets.

Uses Open-Meteo's Previous Runs API, never the historical/reanalysis endpoint.
Only Python's standard library is required. Requests and returned values are
cached locally, with units and provenance, so experiments can be reproduced.

Example:
    python3 analysis/fetch_nwp_archive.py
    python3 analysis/fetch_nwp_archive.py --horizons 96 --output-dir reports/wind_advanced/data/day4

API documentation: https://open-meteo.com/en/docs/previous-runs-api
"""

from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
DOCUMENTATION = "https://open-meteo.com/en/docs/previous-runs-api"
DEFAULT_MODELS = ("ecmwf_ifs025", "icon_global", "icon_eu", "gfs_global")
VARIABLES = ("wind_speed_10m", "wind_direction_10m", "temperature_2m", "surface_pressure")
UNITS = dict(zip(VARIABLES, ("m/s", "°", "°C", "hPa")))
HORIZONS = (24, 48, 72)


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def station_metadata() -> dict:
    """Read the station identity from observations, without inferring coordinates."""
    source = ROOT / "H_42_previous-2020-2024.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row["NOM_USUEL"] == "SAINT-ETIENNE":
                return {
                    "station_id": row["NUM_POSTE"],
                    "name": row["NOM_USUEL"],
                    "latitude": float(row["LAT"]),
                    "longitude": float(row["LON"]),
                    "elevation": float(row["ALTI"]),
                    "source_file": source.name,
                }
    raise ValueError(f"SAINT-ETIENNE not found in {source}")


def month_chunks(start: date, end: date):
    while start <= end:
        last = date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
        last = min(last, end)
        yield start, last
        start = last + timedelta(days=1)


def fetch(url: str, path: Path, retries: int = 3) -> tuple[dict, dict]:
    """GET or read a frozen response. Errors are never converted into weather data."""
    if path.exists():
        raw = gzip.decompress(path.read_bytes())
        cached = True
    else:
        for attempt in range(retries):
            try:
                request = Request(url, headers={"User-Agent": "AI-m-t-o-research/1.0"})
                with urlopen(request, timeout=90) as response:
                    raw = response.read()
                payload = json.loads(raw)
                if payload.get("error"):
                    raise ValueError(payload)
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(gzip.compress(raw, mtime=0))
                temporary.replace(path)
                cached = False
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError):
                    detail = exc.read().decode("utf-8", errors="replace")
                    if exc.code < 500 and exc.code != 429:
                        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
                if attempt == retries - 1:
                    raise
                time.sleep(min(30, 5 * (attempt + 1)))
    payload = json.loads(raw)
    if payload.get("error"):
        raise ValueError(payload)
    return payload, {
        "url": url,
        "cache_file": display_path(path),
        "cache_hit": cached,
        "sha256_uncompressed_response": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
        "frozen_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "response_metadata": {k: v for k, v in payload.items() if k != "hourly"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=range(24, 169, 24), default=list(HORIZONS),
                        help="Archive offsets in hours, not an operational publication timestamp")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/wind_advanced/data")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be on or after --start")
    args.output_dir = args.output_dir.resolve()
    args.horizons = sorted(set(args.horizons))
    station = station_metadata()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "endpoint": ENDPOINT,
        "documentation": DOCUMENTATION,
        "station": station,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "models": args.models,
        "horizon_hours": args.horizons,
        "variable_units": UNITS,
        "time_zone": "UTC",
        "source_type": "archived forecasts; not reanalysis or future observations",
        "lead_time_semantics": (
            "previous_dayN is Open-Meteo's N*24-hour offset before "
            "valid_time. forecast_origin = valid_time - horizon_hours is the nominal "
            "offset time. It is NOT a returned model initialization or publication timestamp. "
            "Operational issue/publication latency is not established by this API. "
            "Use Single Runs API plus publication timing for an operational replay."
        ),
        "publication_latency_audit": {
            "finding": "Archive-offset selection is relative to model initialization, not publication.",
            "evidence": [
                "OmFileSplitter.swift: skip=previousDay*86400/time.dtSeconds; "
                "new forecasts overwrite archived values only when tArray>=skip.",
                "GenericVariableHandle.swift: time range starts at min(run,timeMin). "
                "No publication timestamp gates this selection.",
                "Single Runs documentation distinguishes initialization from publication and "
                "gives typical global-model computation/distribution latency of 4-6 hours.",
            ],
            "sources": [
                "https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Helper/OmFileSplitter.swift#L372-L459",
                "https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Helper/Writer/GenericVariableHandle.swift#L308-L312",
                "https://open-meteo.com/en/docs/single-runs-api",
            ],
            "experiment_policy": (
                "For operational target horizons24/48/72h, use archive offsets48/72/96h, "
                "respectively: a conservative additional24h allowance for publication. "
                "This assumes availability within24h of initialization; it is not a "
                "verified replay of historical provider outages or release timestamps. "
                "The CSV horizon_hours continues to describe archive offset, so remap "
                "explicitly in the trainer; do not relabel archived initialization as issue time."
            ),
        },
        "grid_metadata_note": (
            "Combined-model API responses return a single latitude/longitude; do not "
            "interpret it as all four model grid points. Requested station coordinates "
            "and elevation are explicit in every URL. API defaults select terrain-optimized "
            "land cells; surface pressure and temperature may be elevation-adjusted."
        ),
        "known_coverage_notes": [
            "Documentation says most archives start in January 2024; early January probes "
            "returned null. Missing forecasts remain empty; no interpolation or filling is applied.",
            "ecmwf_ifs04 has missing later coverage; ecmwf_ifs025 is explicitly selected instead.",
            "ARPEGE Europe/World and AROME France returned all-null historical wind probes "
            "on 2024-03-10, 2024-09-10, 2025-01-10, 2025-09-10 (ARPEGE World also "
            "2026-01-10). They are not silently substituted or used in training.",
            "NWP models and archive processing can change over time; these are actual historical "
            "archived predictions, not a homogeneous model-version reforecast dataset.",
        ],
        "requests": [],
        "coverage": {},
        "completed": False,
    }
    csv_path = args.output_dir / "nwp_archive.csv"
    pending = csv_path.with_suffix(".csv.tmp")
    provenance_path = args.output_dir / "provenance.json"
    fieldnames = ["valid_time", "forecast_origin", "model", "horizon_hours", *VARIABLES]
    rows_total = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor, pending.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            jobs = []
            for start, end in month_chunks(args.start, args.end):
                params = {
                    "latitude": station["latitude"],
                    "longitude": station["longitude"],
                    "elevation": station["elevation"],
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "hourly": ",".join(f"{variable}_previous_day{h // 24}" for h in args.horizons for variable in VARIABLES),
                    "models": ",".join(args.models),
                    "wind_speed_unit": "ms",
                    "temperature_unit": "celsius",
                    "timezone": "GMT",
                    "cell_selection": "land",
                }
                url = ENDPOINT + "?" + urlencode(params)
                fingerprint = hashlib.sha256(url.encode()).hexdigest()[:12]
                path = args.output_dir / "raw" / f"{start}_{end}_{fingerprint}.json.gz"
                jobs.append((start, end, executor.submit(fetch, url, path)))
            for start, end, future in jobs:
                payload, request_info = future.result()
                provenance["requests"].append(request_info)
                hourly = payload["hourly"]
                times = hourly["time"]
                expected = ((end - start).days + 1) * 24
                if len(times) != expected or payload.get("utc_offset_seconds") != 0:
                    raise ValueError(f"Unexpected UTC time axis: {start} .. {end}")
                for model in args.models:
                    for horizon in args.horizons:
                        coverage_key = f"{model}/{horizon}"
                        coverage = provenance["coverage"].setdefault(coverage_key, {
                            "rows": 0, "wind_nonmissing": 0, "all_variables_nonmissing": 0,
                            "first_wind_valid_time": None, "last_wind_valid_time": None,
                            "variable_nonmissing": dict.fromkeys(VARIABLES, 0),
                        })
                        values = {}
                        for variable in VARIABLES:
                            key = f"{variable}_previous_day{horizon // 24}"
                            if len(args.models) > 1:
                                key += f"_{model}"
                            if payload["hourly_units"].get(key) != UNITS[variable]:
                                raise ValueError(f"Unexpected unit for {key}: {payload['hourly_units'].get(key)}")
                            values[variable] = hourly[key]
                            if len(values[variable]) != len(times):
                                raise ValueError(f"Mismatched time/variable lengths for {key}")
                        for index, stamp in enumerate(times):
                            valid = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
                            row = {
                                "valid_time": valid.isoformat(),
                                "forecast_origin": (valid - timedelta(hours=horizon)).isoformat(),
                                "model": model,
                                "horizon_hours": horizon,
                                **{variable: values[variable][index] for variable in VARIABLES},
                            }
                            writer.writerow(row)
                            rows_total += 1
                            coverage["rows"] += 1
                            for variable in VARIABLES:
                                coverage["variable_nonmissing"][variable] += row[variable] is not None
                            coverage["all_variables_nonmissing"] += all(row[v] is not None for v in VARIABLES)
                            if row["wind_speed_10m"] is not None:
                                coverage["wind_nonmissing"] += 1
                                if coverage["first_wind_valid_time"] is None:
                                    coverage["first_wind_valid_time"] = row["valid_time"]
                                coverage["last_wind_valid_time"] = row["valid_time"]
                handle.flush()
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                print(f"{start} .. {end}: {len(times)} times, {rows_total} rows written", flush=True)
        pending.replace(csv_path)
        provenance["completed"] = True
        provenance["rows_total"] = rows_total
        provenance["csv_file"] = display_path(csv_path)
        provenance["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Saved {csv_path}", flush=True)
        for key, coverage in provenance["coverage"].items():
            print(key, json.dumps(coverage), flush=True)
    except Exception as exc:
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
