#!/usr/bin/env python3
"""Download Météo-France hourly station exports (data.gouv.fr) for Saint-Étienne's region.

Files are the public "Données climatologiques de base - horaires" department
exports, kept compressed exactly as published. A provenance file records the URL,
size, SHA-256 and download time of each file, so later runs can verify inputs.

Example:
    python analysis/fetch_meteofrance.py
    python analysis/fetch_meteofrance.py --departments 42 69 --refresh
    python analysis/fetch_meteofrance.py --departments 42 --periods 2000-2009 2010-2019 --output-dir data/meteofrance_history
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR"
DATASET = "https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires/"
# Loire, Rhône, Haute-Loire, Puy-de-Dôme: stations within roughly 120 km.
DEPARTMENTS = ("42", "69", "43", "63")
PERIODS = ("previous-2020-2024", "latest-2025-2026")
# Older decades, used with the GEFS reforecast (2010-2019) for the target station.
OLDER_PERIODS = ("2000-2009", "2010-2019")


def download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "AI-m-t-o-research/1.0"})
    with urlopen(request, timeout=300) as response, tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--departments", nargs="+", default=list(DEPARTMENTS))
    parser.add_argument("--periods", nargs="+", default=list(PERIODS), choices=PERIODS + OLDER_PERIODS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/meteofrance")
    parser.add_argument("--refresh", action="store_true",
                        help="Download again even if a file exists (the latest export grows daily)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance.setdefault("dataset", DATASET)
    files = provenance.setdefault("files", {})
    for department in args.departments:
        for period in args.periods:
            name = f"H_{department}_{period}.csv.gz"
            path = args.output_dir / name
            url = f"{BASE_URL}/{name}"
            if args.refresh or not path.exists():
                print(f"Downloading {url}", flush=True)
                download(url, path)
                files[name] = {"url": url,
                               "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            files.setdefault(name, {"url": url, "downloaded_at_utc": None})
            files[name]["bytes"] = path.stat().st_size
            files[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{name}: {files[name]['bytes']} bytes, sha256 {files[name]['sha256'][:12]}", flush=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
