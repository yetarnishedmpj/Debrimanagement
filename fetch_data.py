"""
Space-Track.org data acquisition and caching helpers for debris-cluster seeding.

The script is intentionally defensive because API access is rate limited and
occasionally unavailable. When fresh cached data exists, it is reused instead of
forcing a network round-trip.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import requests

LOGGER = logging.getLogger(__name__)

SPACE_TRACK_BASE_URL = "https://www.space-track.org"
SPACE_TRACK_LOGIN_URL = f"{SPACE_TRACK_BASE_URL}/ajaxauth/login"
DEFAULT_QUERY_PATH = (
    "/basicspacedata/query/class/gp/OBJECT_TYPE/DEBRIS/"
    "orderby/NORAD_CAT_ID%20asc/format/json"
)
DEFAULT_CACHE_PATH = Path("data/high_risk_tle_cache.json")
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_TTL_HOURS = 12
HIGH_RISK_KEYWORDS = (
    "IRIDIUM 33",
    "COSMOS 2251",
    "FENGYUN 1C",
    "BREEZE-M",
    "SL-16 DEB",
)


def load_credentials() -> tuple[str, str]:
    """Load Space-Track credentials from environment variables."""

    username = os.getenv("SPACETRACK_USERNAME")
    password = os.getenv("SPACETRACK_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing Space-Track credentials. Set SPACETRACK_USERNAME and "
            "SPACETRACK_PASSWORD before running fetch_data.py."
        )
    return username, password


def is_cache_fresh(cache_path: Path, ttl_hours: float) -> bool:
    """Use file age as a simple freshness signal for rate-limit protection."""

    if not cache_path.exists():
        return False
    elapsed_seconds = max(0.0, time.time() - cache_path.stat().st_mtime)
    return elapsed_seconds <= ttl_hours * 3600.0


def load_cached_records(cache_path: Path) -> List[Dict[str, Any]]:
    """Load cached records from JSON or CSV."""

    suffix = cache_path.suffix.lower()
    if suffix == ".json":
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload["records"] if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Unsupported JSON cache layout in {cache_path}.")
        return [dict(record) for record in records]

    if suffix == ".csv":
        with cache_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    raise ValueError(f"Unsupported cache format for {cache_path}. Use .json or .csv.")


def cache_records(records: Sequence[Mapping[str, Any]], cache_path: Path) -> None:
    """Persist filtered TLE records locally for reproducible experiments."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = cache_path.suffix.lower()

    if suffix == ".json":
        payload = {"record_count": len(records), "records": list(records)}
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return

    if suffix == ".csv":
        fieldnames = sorted({key for record in records for key in record.keys()})
        with cache_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
        return

    raise ValueError(f"Unsupported cache format for {cache_path}. Use .json or .csv.")


def login_to_space_track(
    session: requests.Session,
    *,
    username: str,
    password: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Authenticate a session against Space-Track's login endpoint."""

    try:
        response = session.post(
            SPACE_TRACK_LOGIN_URL,
            data={"identity": username, "password": password},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Space-Track login failed: {exc}") from exc


def fetch_tle_records(
    session: requests.Session,
    *,
    query_path: str = DEFAULT_QUERY_PATH,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """Download raw TLE/GP records from Space-Track."""

    url = f"{SPACE_TRACK_BASE_URL}{query_path}"
    try:
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to fetch TLE data from {url}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:200].replace("\n", " ")
        raise RuntimeError(
            "Space-Track returned a non-JSON payload. "
            f"Response preview: {snippet!r}"
        ) from exc

    if not isinstance(payload, list):
        raise RuntimeError(
            f"Expected a JSON list of TLE records, received {type(payload).__name__}."
        )
    return [dict(record) for record in payload]


def filter_high_risk_clusters(
    records: Iterable[Mapping[str, Any]],
    *,
    cluster_keywords: Sequence[str] = HIGH_RISK_KEYWORDS,
) -> List[Dict[str, Any]]:
    """Filter for crowded or collision-derived debris fields."""

    keywords = tuple(keyword.upper() for keyword in cluster_keywords)
    filtered: List[Dict[str, Any]] = []
    for record in records:
        searchable_fields = " ".join(
            str(record.get(field, ""))
            for field in ("OBJECT_NAME", "SATNAME", "OBJECT_ID", "INTLDES")
        ).upper()
        if any(keyword in searchable_fields for keyword in keywords):
            enriched = dict(record)
            enriched.update(parse_tle_fields(record))
            filtered.append(enriched)
    return filtered


def parse_tle_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract a small subset of orbital elements from TLE line 2."""

    line2 = record.get("TLE_LINE2") or record.get("TLE2") or record.get("LINE2") or ""
    if not isinstance(line2, str) or not line2.strip():
        return {}

    try:
        tokens = line2.split()
        if len(tokens) < 8:
            return {}
        eccentricity = float(f"0.{tokens[4].strip()}")
        return {
            "inclination_deg": float(tokens[2]),
            "raan_deg": float(tokens[3]),
            "eccentricity": eccentricity,
            "arg_perigee_deg": float(tokens[5]),
            "mean_anomaly_deg": float(tokens[6]),
            "mean_motion_rev_per_day": float(tokens[7]),
        }
    except (TypeError, ValueError):
        LOGGER.debug("Unable to parse TLE line 2 for record %s", record.get("OBJECT_NAME"))
        return {}


def records_to_relative_positions(
    records: Sequence[Mapping[str, Any]],
    *,
    world_range_km: float,
    max_targets: int,
) -> List[List[float]]:
    """
    Convert cached TLE orbital elements into approximate relative-frame positions.

    This is a lightweight embedding for MARL prototyping rather than a full
    orbital propagator, but it preserves cluster geometry well enough for policy
    initialization and visualization.
    """

    positions: List[List[float]] = []
    for record in records[:max_targets]:
        inclination = np.deg2rad(float(record.get("inclination_deg", 0.0)))
        raan = np.deg2rad(float(record.get("raan_deg", 0.0)))
        mean_anomaly = np.deg2rad(float(record.get("mean_anomaly_deg", 0.0)))
        mean_motion = float(record.get("mean_motion_rev_per_day", 14.5))

        radial_scale = np.clip(mean_motion / 16.5, 0.25, 1.0)
        x = world_range_km * radial_scale * np.cos(mean_anomaly)
        y = world_range_km * radial_scale * np.sin(mean_anomaly)
        z = world_range_km * 0.35 * np.sin(inclination) * np.cos(raan)
        positions.append([float(x), float(y), float(z)])

    return positions


def fetch_and_cache_tles(
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    query_path: str = DEFAULT_QUERY_PATH,
    ttl_hours: float = DEFAULT_CACHE_TTL_HOURS,
    cluster_keywords: Sequence[str] = HIGH_RISK_KEYWORDS,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Acquire high-risk debris TLEs with automatic cache fallback.

    When a live request fails but a cache exists, the cache is returned so the
    MARL pipeline remains usable offline and under API rate limits.
    """

    cache_path = cache_path.resolve()
    if not force_refresh and is_cache_fresh(cache_path, ttl_hours):
        LOGGER.info("Using fresh cached TLE data from %s", cache_path)
        return load_cached_records(cache_path)

    try:
        username, password = load_credentials()
        with requests.Session() as session:
            session.headers.update({"User-Agent": "orbital-debris-marl/1.0"})
            login_to_space_track(session, username=username, password=password)
            records = fetch_tle_records(session, query_path=query_path)
        filtered_records = filter_high_risk_clusters(
            records, cluster_keywords=cluster_keywords
        )
        if not filtered_records:
            raise RuntimeError(
                "Space-Track query succeeded but did not yield any high-risk cluster records."
            )
        cache_records(filtered_records, cache_path)
        LOGGER.info("Cached %d filtered TLE records to %s", len(filtered_records), cache_path)
        return filtered_records
    except Exception as exc:  # noqa: BLE001 - intentional fallback behavior.
        if cache_path.exists():
            LOGGER.warning(
                "Live Space-Track fetch failed (%s). Falling back to cached data at %s.",
                exc,
                cache_path,
            )
            return load_cached_records(cache_path)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and cache Space-Track TLE data.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Local JSON or CSV cache path for filtered TLE records.",
    )
    parser.add_argument(
        "--query-path",
        default=DEFAULT_QUERY_PATH,
        help="Space-Track query path appended to https://www.space-track.org.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=DEFAULT_CACHE_TTL_HOURS,
        help="Reuse the cache when it is newer than this threshold.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the freshness check and force a live API request.",
    )
    parser.add_argument(
        "--cluster-keyword",
        dest="cluster_keywords",
        action="append",
        default=[],
        help="Additional case-insensitive keyword used to filter high-risk clusters.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_argument_parser().parse_args()
    keywords = tuple(HIGH_RISK_KEYWORDS) + tuple(args.cluster_keywords)

    try:
        records = fetch_and_cache_tles(
            cache_path=args.cache_path,
            query_path=args.query_path,
            ttl_hours=args.ttl_hours,
            cluster_keywords=keywords,
            force_refresh=args.force_refresh,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must surface actionable errors.
        LOGGER.error("Unable to acquire TLE data: %s", exc)
        return 1

    example_names = sorted(
        {str(record.get("OBJECT_NAME", "UNKNOWN")) for record in records[:5]}
    )
    LOGGER.info(
        "Prepared %d cached records. Example cluster names: %s",
        len(records),
        ", ".join(example_names) if example_names else "N/A",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
