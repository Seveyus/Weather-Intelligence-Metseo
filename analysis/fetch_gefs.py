#!/usr/bin/env python3
"""Extract GEFS v12 control forecasts around Saint-Étienne, 2010 → today.

Two public NOAA archives on AWS, same GEFS v12 model:
  * reforecast, daily 00 UTC runs up to 2019-12-31 (noaa-gefs-retrospective);
  * operational, from 2020-09-23 (noaa-gefs-pds). January–September 2020 has neither.

Only the needed GRIB messages are fetched with HTTP byte ranges (index files),
then cropped to a 33×33 box of 0.25° cells centred on the station and saved as
one compressed .npz per run. Global fields are never written to disk. Reruns
skip finished runs, so the download can be interrupted and resumed.

Example:
    python analysis/fetch_gefs.py                       # 2010-01-01 → yesterday
    python analysis/fetch_gefs.py --start 2019-01-01 --end 2019-01-31 --workers 16
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import eccodes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFORECAST = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OPERATIONAL = "https://noaa-gefs-pds.s3.amazonaws.com"
REFORECAST_END = date(2019, 12, 31)
OPERATIONAL_START = date(2020, 9, 23)
STATION = (45.435333, 4.368167)  # SAINT-ETIENNE 42218011
HALF_WIDTH = 16  # cells each side: 33×33 at 0.25° ≈ ±4°
# Leads cover +24/+48/+72 h targets with a 24 h publication margin from daily runs.
LEADS = tuple(range(48, 121, 6))
# (name, reforecast file stem, index "VAR:level") — identical fields in both archives.
VARIABLES = (
    ("u10", "ugrd_hgt", "UGRD:10 m above ground"),
    ("v10", "vgrd_hgt", "VGRD:10 m above ground"),
    ("gust", "gust_sfc", "GUST:surface"),
    ("mslet", "pres_mslet", "MSLET:mean sea level"),
    ("t2m", "tmp_2m", "TMP:2 m above ground"),
)


def http_get(url: str, byte_range: tuple[int, int | None] | None = None, retries: int = 5) -> bytes:
    headers = {"User-Agent": "AI-m-t-o-research/1.0"}
    if byte_range:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            if attempt == retries - 1:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError("unreachable")


def index_ranges(index_text: str) -> dict[tuple[str, int], tuple[int, int | None]]:
    """Map (VAR:level, lead hours) to the message byte range in the GRIB file."""
    rows = [line.split(":") for line in index_text.strip().splitlines()]
    out = {}
    for i, row in enumerate(rows):
        start = int(row[1])
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
        lead = row[5].split()[0]
        if lead.isdigit():
            out[(f"{row[3]}:{row[4]}", int(lead))] = (start, end)
    return out


def message_jobs(run: date):
    """(variable, lead, url, byte range) for every message of one run."""
    stamp = run.strftime("%Y%m%d") + "00"
    jobs = []
    if run <= REFORECAST_END:
        for name, stem, key in VARIABLES:
            url = f"{REFORECAST}/{run.year}/{stamp}/c00/Days:1-10/{stem}_{stamp}_c00.grib2"
            ranges = index_ranges(http_get(url + ".idx").decode())
            jobs += [(name, lead, url, ranges[(key, lead)]) for lead in LEADS]
    elif run >= OPERATIONAL_START:
        for lead in LEADS:
            url = (f"{OPERATIONAL}/gefs.{run:%Y%m%d}/00/atmos/pgrb2sp25/"
                   f"gec00.t00z.pgrb2s.0p25.f{lead:03d}")
            ranges = index_ranges(http_get(url + ".idx").decode())
            jobs += [(name, lead, url, ranges[(key, lead)]) for name, _, key in VARIABLES]
    else:
        raise FileNotFoundError(f"No GEFS v12 archive for {run}")
    return jobs


def crop(message: bytes):
    handle = eccodes.codes_new_from_message(message)
    try:
        ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
        if (ni, nj) != (1440, 721) or eccodes.codes_get(handle, "iDirectionIncrementInDegrees") != .25:
            raise ValueError(f"Unexpected grid {ni}x{nj}")
        lat0 = eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
        values = eccodes.codes_get_values(handle).reshape(nj, ni)
    finally:
        eccodes.codes_release(handle)
    lats = lat0 - .25 * np.arange(nj)  # GEFS grids run north to south from 90°N
    lons = .25 * np.arange(ni)
    row = int(np.argmin(np.abs(lats - STATION[0])))
    col = int(np.argmin(np.abs(lons - STATION[1] % 360)))
    rows = slice(row - HALF_WIDTH, row + HALF_WIDTH + 1)
    cols = np.arange(col - HALF_WIDTH, col + HALF_WIDTH + 1) % ni  # wraps across 0°E
    return values[rows][:, cols].astype(np.float32), lats[rows], np.where(lons[cols] > 180, lons[cols] - 360, lons[cols])


def fetch_run(run: date, output: Path, workers: int) -> str:
    path = output / f"{run:%Y}" / f"gefs_c00_{run:%Y%m%d}00.npz"
    if path.exists():
        return "cached"
    jobs = message_jobs(run)
    with ThreadPoolExecutor(workers) as pool:
        crops = list(pool.map(lambda job: crop(http_get(job[2], job[3])), jobs))
    names = [v[0] for v in VARIABLES]
    data = np.full((len(names), len(LEADS), 2 * HALF_WIDTH + 1, 2 * HALF_WIDTH + 1), np.nan, np.float32)
    for (name, lead, _, _), (values, lats, lons) in zip(jobs, crops):
        data[names.index(name), LEADS.index(lead)] = values
    if np.isnan(data).any():
        raise ValueError(f"Incomplete run {run}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        np.savez_compressed(handle, data=data, variables=np.array(names), leads_hours=np.array(LEADS),
                            latitudes=lats, longitudes=lons, init_time=np.array(f"{run:%Y-%m-%d}T00:00Z"),
                            source=np.array(jobs[0][2].split("/")[2]))
        temporary = Path(handle.name)
    temporary.replace(path)
    return "saved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date() - timedelta(days=1))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/gefs")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.json").write_text(json.dumps({
        "model": "GEFS v12 control member (c00), 00 UTC runs",
        "archives": {"reforecast": REFORECAST, "operational": OPERATIONAL},
        "variables": {name: key for name, _, key in VARIABLES},
        "units": {"u10": "m/s", "v10": "m/s", "gust": "m/s", "mslet": "Pa", "t2m": "K"},
        "leads_hours": list(LEADS), "grid": "0.25° regular lat/lon, 33×33 box centred on the station cell",
        "station": {"name": "SAINT-ETIENNE", "latitude": STATION[0], "longitude": STATION[1]},
        "gap": "2020-01-01 to 2020-09-22: no GEFS v12 archive",
        "array_layout": "data[variable, lead, latitude (north→south), longitude (west→east)]",
    }, indent=2) + "\n")
    # Newest first: the test period and recent training years are usable earliest.
    runs = [args.end - timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    runs = [r for r in runs if r <= REFORECAST_END or r >= OPERATIONAL_START]
    failures, started = [], time.time()
    for done, run in enumerate(runs, 1):
        try:
            status = fetch_run(run, output, args.workers)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            status = f"missing ({type(exc).__name__}: {exc})"
            failures.append({"run": run.isoformat(), "error": status})
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            status = f"network error, will retry on next launch ({exc})"
            failures.append({"run": run.isoformat(), "error": status})
        elapsed = time.time() - started
        print(f"{done}/{len(runs)} {run} {status} · {elapsed / 3600:.2f} h elapsed", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Finished: {len(runs) - len(failures)} runs available, {len(failures)} failures", flush=True)


if __name__ == "__main__":
    main()
