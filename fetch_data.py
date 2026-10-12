"""
Space-Track.org and CelesTrak GP/OMM data acquisition, SGP4 propagation,
and consistent Hill (LVLH) relative-frame transformations.

Key Astrodynamic Capabilities:
1. Documented orbital data with recorded UTC epochs (CelesTrak GP & Space-Track).
2. Full support for modern CCSDS Orbit Mean-Elements Message (OMM) formats (JSON, CSV)
   accommodating expanded catalog identifiers (>100,000 up to 9 digits, per 2026 CelesTrak standards).
3. SGP4 propagation using established WGS72 gravitational constants.
4. Consistent reference frame transformation: converting inertial TEME (True Equator,
   Mean Equinox) states into the rotating Hill / LVLH (Local-Vertical Local-Horizontal)
   frame at the exact same simulation epoch.
5. Separate relative-motion and maneuver model for the controlled servicing spacecraft
   using Clohessy-Wiltshire (Hill) dynamics and physical Delta-V propellant accounting.
6. Defensive caching and automatic fallback for offline and rate-limited operation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import requests

try:
    from sgp4.api import Satrec, WGS72, jday
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "SGP4 library is required for high-fidelity orbit propagation. "
        "Install via `pip install sgp4`."
    ) from exc

LOGGER = logging.getLogger(__name__)

# Primary CelesTrak GP API endpoints (2026 CCSDS OMM compliant)
CELESTRAK_BASE_URL = "https://celestrak.org"
CELESTRAK_GP_URL = f"{CELESTRAK_BASE_URL}/NORAD/elements/gp.php"

# Space-Track endpoints
SPACE_TRACK_BASE_URL = "https://www.space-track.org"
SPACE_TRACK_LOGIN_URL = f"{SPACE_TRACK_BASE_URL}/ajaxauth/login"
DEFAULT_QUERY_PATH = (
    "/basicspacedata/query/class/gp/OBJECT_TYPE/DEBRIS/"
    "orderby/NORAD_CAT_ID%20asc/format/json"
)

DEFAULT_CACHE_PATH = Path("data/high_risk_tle_cache.json")
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_TTL_HOURS = 12

# Standard WGS72 Earth physical constants (operational standard for SGP4)
EARTH_RADIUS_WGS72_KM = 6378.135
MU_EARTH_WGS72_KM3_S2 = 398600.8

# Unit conversion constants for SGP4 internal mean motion derivatives
NDOT_UNITS = 330023.68999535416  # (720.0 * 1440.0) / pi
NDDOT_UNITS = 475234113.59331    # (720.0 * 1440.0 * 1440.0) / pi
SGP4_MAX_INTERNAL_SATNUM = 339999

HIGH_RISK_KEYWORDS = (
    "IRIDIUM 33",
    "COSMOS 2251",
    "FENGYUN 1C",
    "BREEZE-M",
    "SL-16 DEB",
)


# ==============================================================================
# 1. Epoch and Time Utilities
# ==============================================================================

def parse_epoch(epoch_val: Union[str, datetime]) -> datetime:
    """
    Parse an ISO-8601 UTC epoch timestamp into a timezone-aware datetime object.

    Accommodates variations with/without microsecond resolution and optional 'Z' suffix.
    """
    if isinstance(epoch_val, datetime):
        if epoch_val.tzinfo is None:
            return epoch_val.replace(tzinfo=timezone.utc)
        return epoch_val.astimezone(timezone.utc)

    text = str(epoch_val).strip()
    if text.endswith("Z"):
        text = text[:-1]

    formats = (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    raise ValueError(f"Unable to parse ISO-8601 epoch timestamp: {epoch_val!r}")


def datetime_to_jd(dt: datetime) -> Tuple[float, float]:
    """
    Convert a UTC datetime into Julian Date components (jd, fr) for SGP4 evaluation.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    sec_float = dt.second + dt.microsecond * 1e-6
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, sec_float)


def jd_to_datetime(jd: float, fr: float) -> datetime:
    """
    Convert Julian Date components (jd, fr) back to UTC datetime.
    """
    jd_total = jd + fr
    # Standard Julian Date to Gregorian calendar conversion
    z = int(jd_total + 0.5)
    f = (jd_total + 0.5) - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    day_int = int(day)
    day_frac = day - day_int
    total_seconds = day_frac * 86400.0
    hour = int(total_seconds // 3600)
    minute = int((total_seconds % 3600) // 60)
    second_float = total_seconds % 60
    second = int(second_float)
    microsecond = int(round((second_float - second) * 1e6))
    if microsecond >= 1_000_000:
        microsecond -= 1_000_000
        second += 1

    return datetime(year, month, day_int, hour, minute, second, microsecond, tzinfo=timezone.utc)


# ==============================================================================
# 2. OMM / TLE Parsing and SGP4 Record Initialization
# ==============================================================================

def parse_omm_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize typed fields from a CCSDS OMM or general-perturbations dictionary.

    Supports catalog IDs > 100,000 (up to 9 digits) and standardized OMM keys.
    """
    result: Dict[str, Any] = {}

    # Metadata & Identifiers
    name = record.get("OBJECT_NAME") or record.get("SATNAME") or record.get("name") or "UNKNOWN"
    cat_id_raw = (
        record.get("NORAD_CAT_ID")
        or record.get("CATNR")
        or record.get("OBJECT_NUMBER")
        or record.get("SATNUM")
        or ""
    )
    object_id = record.get("OBJECT_ID") or record.get("INTLDES") or ""

    result["OBJECT_NAME"] = str(name).strip()
    result["NORAD_CAT_ID"] = str(cat_id_raw).strip()
    result["OBJECT_ID"] = str(object_id).strip()

    # Epoch
    epoch = record.get("EPOCH")
    if epoch:
        result["EPOCH"] = str(epoch).strip()
        try:
            result["epoch_dt"] = parse_epoch(epoch)
        except Exception:
            pass

    # Core Keplerian / Brouwer Mean Elements
    key_mapping = {
        "mean_motion_rev_per_day": ("MEAN_MOTION", "mean_motion_rev_per_day", "no"),
        "eccentricity": ("ECCENTRICITY", "eccentricity", "ecco"),
        "inclination_deg": ("INCLINATION", "inclination_deg", "inclo"),
        "raan_deg": ("RA_OF_ASC_NODE", "raan_deg", "nodeo"),
        "arg_perigee_deg": ("ARG_OF_PERICENTER", "arg_perigee_deg", "argpo"),
        "mean_anomaly_deg": ("MEAN_ANOMALY", "mean_anomaly_deg", "mo"),
        "bstar": ("BSTAR", "bstar"),
        "mean_motion_dot": ("MEAN_MOTION_DOT", "ndot"),
        "mean_motion_ddot": ("MEAN_MOTION_DDOT", "nddot"),
        "ephemeris_type": ("EPHEMERIS_TYPE", "ephtype"),
        "classification_type": ("CLASSIFICATION_TYPE", "classification"),
        "element_set_no": ("ELEMENT_SET_NO", "elnum"),
        "rev_at_epoch": ("REV_AT_EPOCH", "revnum"),
    }

    for target_key, candidate_keys in key_mapping.items():
        for cand in candidate_keys:
            if cand in record and record[cand] is not None:
                val = record[cand]
                try:
                    if target_key in ("classification_type",):
                        result[target_key] = str(val).strip()
                    elif target_key in ("ephemeris_type", "element_set_no", "rev_at_epoch"):
                        result[target_key] = int(val)
                    else:
                        result[target_key] = float(val)
                    break
                except (ValueError, TypeError):
                    pass

    return result


def parse_tle_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract a subset of orbital elements from TLE line 2 (backward-compatible helper).
    """
    line2 = record.get("TLE_LINE2") or record.get("TLE2") or record.get("LINE2") or ""
    if not isinstance(line2, str) or not line2.strip():
        # Fall back to OMM fields if present
        parsed = parse_omm_fields(record)
        return {
            k: parsed[k]
            for k in (
                "inclination_deg",
                "raan_deg",
                "eccentricity",
                "arg_perigee_deg",
                "mean_anomaly_deg",
                "mean_motion_rev_per_day",
            )
            if k in parsed
        }

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


def record_to_satrec(
    record: Mapping[str, Any],
    gravconst: Any = WGS72,
) -> Tuple[Optional[Satrec], Optional[str]]:
    """
    Construct and initialize a validated SGP4 Satrec instance from an OMM or TLE record.

    Accommodates 6-digit (>100,000) and 9-digit catalog identifiers without failing
    library-internal bounds.
    """
    # 1. Attempt TLE 2-line initialization if line strings are present
    line1 = record.get("TLE_LINE1") or record.get("LINE1") or record.get("TLE1")
    line2 = record.get("TLE_LINE2") or record.get("LINE2") or record.get("TLE2")
    if line1 and line2 and isinstance(line1, str) and isinstance(line2, str):
        try:
            sat = Satrec.twoline2rv(line1.strip(), line2.strip())
            return sat, None
        except Exception as exc:
            LOGGER.debug("twoline2rv failed (%s), attempting OMM field initialization.", exc)

    # 2. CCSDS OMM field initialization
    fields = parse_omm_fields(record)
    epoch_raw = fields.get("EPOCH") or record.get("EPOCH")
    mean_motion = fields.get("mean_motion_rev_per_day")

    if epoch_raw is None or mean_motion is None:
        return None, "Record lacks required SGP4 initialization fields (EPOCH, MEAN_MOTION)."

    try:
        epoch_dt = parse_epoch(epoch_raw)
        _epoch0 = datetime(1949, 12, 31, tzinfo=timezone.utc)
        epoch_days = (epoch_dt - _epoch0).total_seconds() / 86400.0

        # Handle catalog ID range (SGP4 internal satnum limit is 339999 for legacy Alpha-5)
        norad_id_str = fields.get("NORAD_CAT_ID", "0")
        try:
            norad_id_int = int(norad_id_str)
        except ValueError:
            norad_id_int = 0

        internal_satnum = (
            norad_id_int if norad_id_int <= SGP4_MAX_INTERNAL_SATNUM
            else (norad_id_int % 100000)
        )

        bstar = float(fields.get("bstar", 0.0))
        ndot = float(fields.get("mean_motion_dot", 0.0)) / NDOT_UNITS
        nddot = float(fields.get("mean_motion_ddot", 0.0)) / NDDOT_UNITS
        ecco = float(fields.get("eccentricity", 0.0))
        argpo = math.radians(float(fields.get("arg_perigee_deg", 0.0)))
        inclo = math.radians(float(fields.get("inclination_deg", 0.0)))
        mo = math.radians(float(fields.get("mean_anomaly_deg", 0.0)))
        no_kozai = float(mean_motion) / 720.0 * math.pi
        nodeo = math.radians(float(fields.get("raan_deg", 0.0)))

        sat = Satrec()
        sat.sgp4init(
            gravconst,
            "i",
            internal_satnum,
            epoch_days,
            bstar,
            ndot,
            nddot,
            ecco,
            argpo,
            inclo,
            mo,
            no_kozai,
            nodeo,
        )
        return sat, None
    except Exception as exc:
        return None, f"Failed to initialize SGP4 Satrec from OMM: {exc}"


# ==============================================================================
# 3. SGP4 Propagation & Reference Frame Transformations
# ==============================================================================

def propagate_satrec(
    sat: Satrec,
    when: Union[datetime, Tuple[float, float], float],
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Propagate an initialized Satrec instance to a specified time.

    Args:
        sat: Initialized Satrec object.
        when: Either a UTC datetime, Julian date tuple (jd, fr), or elapsed minutes from epoch.

    Returns:
        (error_code, position_teme_km, velocity_teme_km_s)
        error_code: 0 indicates nominal propagation; non-zero indicates an error (e.g. decayed).
    """
    if isinstance(when, datetime):
        jd, fr = datetime_to_jd(when)
        error_code, r, v = sat.sgp4(jd, fr)
    elif isinstance(when, tuple) and len(when) == 2:
        jd, fr = float(when[0]), float(when[1])
        error_code, r, v = sat.sgp4(jd, fr)
    elif isinstance(when, (int, float)):
        # Treat as tsince in minutes
        error_code, r, v = sat.sgp4_tsince(float(when))
    else:
        raise TypeError(f"Unsupported 'when' time specification: {type(when).__name__}")

    r_vec = np.asarray(r, dtype=np.float64) if r is not None else np.zeros((3,), dtype=np.float64)
    v_vec = np.asarray(v, dtype=np.float64) if v is not None else np.zeros((3,), dtype=np.float64)
    return int(error_code), r_vec, v_vec


def propagate_record(
    record: Mapping[str, Any],
    when: Union[datetime, Tuple[float, float], float],
    gravconst: Any = WGS72,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Directly propagate an orbital record to the specified simulation time.
    """
    sat, err = record_to_satrec(record, gravconst=gravconst)
    if sat is None:
        return 99, np.zeros((3,), dtype=np.float64), np.zeros((3,), dtype=np.float64)
    return propagate_satrec(sat, when)


def teme_to_hill_frame(
    r_target: np.ndarray,
    v_target: np.ndarray,
    r_ref: np.ndarray,
    v_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform target inertial TEME state into the rotating Hill / LVLH reference frame.

    Reference Frame Definition:
    Centered on the reference spacecraft state (r_ref, v_ref):
    - x (Radial): Along r_ref (outward from Earth center).
    - z (Cross-Track): Normal to the orbital plane along angular momentum h = r_ref x v_ref.
    - y (Along-Track / In-Track): Along-track direction completing the right-handed triad: y = z x x.

    Rotational Velocity Kinematics:
    The Hill frame rotates with angular velocity omega = (r_ref x v_ref) / ||r_ref||^2.
    In the rotating frame:
    v_rel = C * (v_target - v_ref) - omega_vec x r_rel
    where omega_vec = [0, 0, ||omega||]^T in the Hill frame basis.

    Returns:
        (r_hill_km, v_hill_km_s) as float64 numpy arrays of shape (3,).
    """
    r_target = np.asarray(r_target, dtype=np.float64).reshape(3)
    v_target = np.asarray(v_target, dtype=np.float64).reshape(3)
    r_ref = np.asarray(r_ref, dtype=np.float64).reshape(3)
    v_ref = np.asarray(v_ref, dtype=np.float64).reshape(3)

    r_norm = float(np.linalg.norm(r_ref))
    if r_norm < 1e-6:
        raise ValueError("Reference position vector magnitude is too small.")

    er = r_ref / r_norm
    h_vec = np.cross(r_ref, v_ref)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm < 1e-6:
        raise ValueError("Reference angular momentum is near zero (degenerate orbit).")

    ec = h_vec / h_norm
    et = np.cross(ec, er)

    # Rotation matrix from TEME to Hill (LVLH)
    c_teme_to_hill = np.vstack([er, et, ec])

    # Relative position in Hill frame
    delta_r = r_target - r_ref
    r_hill = c_teme_to_hill @ delta_r

    # Frame rotation rate
    omega_mag = h_norm / (r_norm * r_norm)
    omega_vec = np.array([0.0, 0.0, omega_mag], dtype=np.float64)

    # Relative velocity accounting for rotating coordinate frame
    delta_v = v_target - v_ref
    v_hill = (c_teme_to_hill @ delta_v) - np.cross(omega_vec, r_hill)

    return r_hill, v_hill


def hill_to_teme_frame(
    r_hill: np.ndarray,
    v_hill: np.ndarray,
    r_ref: np.ndarray,
    v_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform relative state in Hill / LVLH frame back to inertial TEME coordinates.
    """
    r_hill = np.asarray(r_hill, dtype=np.float64).reshape(3)
    v_hill = np.asarray(v_hill, dtype=np.float64).reshape(3)
    r_ref = np.asarray(r_ref, dtype=np.float64).reshape(3)
    v_ref = np.asarray(v_ref, dtype=np.float64).reshape(3)

    r_norm = float(np.linalg.norm(r_ref))
    er = r_ref / r_norm
    h_vec = np.cross(r_ref, v_ref)
    h_norm = float(np.linalg.norm(h_vec))
    ec = h_vec / h_norm
    et = np.cross(ec, er)

    c_teme_to_hill = np.vstack([er, et, ec])
    omega_mag = h_norm / (r_norm * r_norm)
    omega_vec = np.array([0.0, 0.0, omega_mag], dtype=np.float64)

    delta_r = c_teme_to_hill.T @ r_hill
    delta_v = c_teme_to_hill.T @ (v_hill + np.cross(omega_vec, r_hill))

    r_target = r_ref + delta_r
    v_target = v_ref + delta_v
    return r_target, v_target


# ==============================================================================
# 4. Servicing Spacecraft Relative Motion and Maneuver Model
# ==============================================================================

@dataclass
class ServicingSpacecraftModel:
    """
    Dedicated relative-motion and maneuver execution model for the controlled servicing spacecraft.

    Operates in the Hill / LVLH frame centered on the deployment/reference orbit.
    Propagates via the analytical Clohessy-Wiltshire (Hill) state transition matrix,
    applies discrete Delta-V impulses or continuous thrust accelerations, and tracks
    physical propellant consumption.
    """

    position_km: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    velocity_km_s: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    fuel_fraction: float = 1.0
    fuel_capacity_delta_v: float = 300.0  # Normalized scale or m/s
    reference_altitude_km: float = 500.0
    earth_radius_km: float = EARTH_RADIUS_WGS72_KM
    gravitational_mu: float = MU_EARTH_WGS72_KM3_S2

    def __post_init__(self) -> None:
        self.position_km = np.asarray(self.position_km, dtype=np.float32).reshape(3)
        self.velocity_km_s = np.asarray(self.velocity_km_s, dtype=np.float32).reshape(3)
        self.reference_radius_km = float(self.earth_radius_km + self.reference_altitude_km)
        self.mean_motion_n = float(np.sqrt(self.gravitational_mu / (self.reference_radius_km**3)))

    @property
    def state_vector(self) -> np.ndarray:
        """Return 6-DOF relative state vector [x, y, z, vx, vy, vz]^T."""
        return np.concatenate([self.position_km, self.velocity_km_s], axis=0).astype(np.float32)

    def apply_impulse(self, delta_v_km_s: Sequence[float]) -> float:
        """
        Apply a discrete Delta-V impulse maneuver.

        Returns the fuel cost fraction deducted.
        """
        dv = np.asarray(delta_v_km_s, dtype=np.float32).reshape(3)
        cost = float(np.linalg.norm(dv))
        if self.fuel_fraction > 0.0:
            fuel_deduction = cost / max(self.fuel_capacity_delta_v, 1e-6)
            self.fuel_fraction = max(0.0, self.fuel_fraction - fuel_deduction)
            self.velocity_km_s += dv
            return fuel_deduction
        return 0.0

    def propagate_cw(self, delta_t_s: float) -> None:
        """
        Propagate spacecraft relative state forward by delta_t_s seconds using Clohessy-Wiltshire STM.
        """
        n = self.mean_motion_n
        tau = float(delta_t_s)
        ntau = n * tau
        s = float(np.sin(ntau))
        c = float(np.cos(ntau))

        # 6x6 CW STM calculation
        x0, y0, z0 = self.position_km
        vx0, vy0, vz0 = self.velocity_km_s

        # Position updates
        x1 = (4.0 - 3.0 * c) * x0 + (s / n) * vx0 + ((2.0 / n) * (1.0 - c)) * vy0
        y1 = 6.0 * (s - ntau) * x0 + y0 - ((2.0 / n) * (1.0 - c)) * vx0 + ((4.0 * s - 3.0 * ntau) / n) * vy0
        z1 = c * z0 + (s / n) * vz0

        # Velocity updates
        vx1 = 3.0 * n * s * x0 + c * vx0 + (2.0 * s) * vy0
        vy1 = 6.0 * n * (c - 1.0) * x0 - (2.0 * s) * vx0 + (4.0 * c - 3.0) * vy0
        vz1 = -n * s * z0 + c * vz0

        self.position_km = np.array([x1, y1, z1], dtype=np.float32)
        self.velocity_km_s = np.array([vx1, vy1, vz1], dtype=np.float32)

    def step(self, delta_v_burn_km_s: Sequence[float], delta_t_s: float) -> float:
        """Execute burn maneuver then propagate relative motion over delta_t_s seconds."""
        fuel_used = self.apply_impulse(delta_v_burn_km_s)
        self.propagate_cw(delta_t_s)
        return fuel_used


# ==============================================================================
# 5. Batch Propagation to Common Simulation Time & Target Selection
# ==============================================================================

def propagate_records_to_hill_frame(
    records: Sequence[Mapping[str, Any]],
    *,
    simulation_epoch: Optional[Union[datetime, str]] = None,
    reference_record: Optional[Mapping[str, Any]] = None,
    reference_altitude_km: float = 500.0,
    max_targets: Optional[int] = None,
    gravconst: Any = WGS72,
) -> Dict[str, Any]:
    """
    Propagate orbital records to the same common simulation time using SGP4
    and project them into the servicing spacecraft's consistent Hill (LVLH) frame.

    Args:
        records: Collection of OMM or TLE records.
        simulation_epoch: Common UTC simulation time. If None, chosen from reference_record
                          or median epoch of valid records.
        reference_record: Record defining the origin of the Hill frame. If None, the first
                          valid record or a circular reference orbit is used.
        reference_altitude_km: Fallback reference circular altitude if no reference record is provided.
        max_targets: Optional upper bound on targets returned.
        gravconst: SGP4 Earth gravitational constants (default WGS72).

    Returns:
        Dictionary containing simulation metadata and list of propagated target states.
    """
    if not records:
        return {"simulation_epoch": None, "targets": []}

    # 1. Parse and initialize all candidate records with SGP4
    valid_entries: List[Tuple[Mapping[str, Any], Satrec, datetime]] = []
    for record in records:
        sat, err = record_to_satrec(record, gravconst=gravconst)
        if sat is None:
            continue
        try:
            fields = parse_omm_fields(record)
            epoch_dt = fields.get("epoch_dt") or parse_epoch(record["EPOCH"])
            valid_entries.append((record, sat, epoch_dt))
        except Exception:
            continue

    if not valid_entries:
        raise ValueError("No propagatable SGP4 orbital records found in input.")

    # 2. Determine consistent simulation epoch
    if simulation_epoch is not None:
        sim_dt = parse_epoch(simulation_epoch)
    elif reference_record is not None and "EPOCH" in reference_record:
        sim_dt = parse_epoch(reference_record["EPOCH"])
    else:
        # Use median epoch of valid records
        sorted_epochs = sorted(entry[2] for entry in valid_entries)
        sim_dt = sorted_epochs[len(sorted_epochs) // 2]

    jd_sim, fr_sim = datetime_to_jd(sim_dt)

    # 3. Determine reference orbit state at sim_dt
    if reference_record is not None:
        ref_sat, _ = record_to_satrec(reference_record, gravconst=gravconst)
        if ref_sat is not None:
            e0, r0, v0 = ref_sat.sgp4(jd_sim, fr_sim)
            if e0 != 0:
                ref_sat = None
    else:
        ref_sat = None

    if ref_sat is None:
        # Default reference: use first valid entry's orbit at sim_dt
        ref_entry = valid_entries[0]
        e0, r0, v0 = ref_entry[1].sgp4(jd_sim, fr_sim)
        if e0 != 0:
            # Fallback to circular reference orbit in equatorial plane
            r_mag = EARTH_RADIUS_WGS72_KM + reference_altitude_km
            v_mag = math.sqrt(MU_EARTH_WGS72_KM3_S2 / r_mag)
            r0 = np.array([r_mag, 0.0, 0.0], dtype=np.float64)
            v0 = np.array([0.0, v_mag, 0.0], dtype=np.float64)
    r_ref = np.asarray(r0, dtype=np.float64)
    v_ref = np.asarray(v0, dtype=np.float64)

    # 4. Propagate all targets to sim_dt and transform to Hill frame
    targets: List[Dict[str, Any]] = []
    for record, sat, _ in valid_entries:
        e, r_tgt, v_tgt = sat.sgp4(jd_sim, fr_sim)
        if e != 0:
            continue  # Exclude decayed or unpropagatable bodies

        r_hill, v_hill = teme_to_hill_frame(r_tgt, v_tgt, r_ref, v_ref)
        dist_km = float(np.linalg.norm(r_hill))

        omm_data = parse_omm_fields(record)
        targets.append(
            {
                "OBJECT_NAME": omm_data.get("OBJECT_NAME", "UNKNOWN"),
                "NORAD_CAT_ID": omm_data.get("NORAD_CAT_ID", "UNKNOWN"),
                "relative_position_km": r_hill.astype(float).tolist(),
                "relative_velocity_km_s": v_hill.astype(float).tolist(),
                "relative_distance_km": dist_km,
                "teme_position_km": np.asarray(r_tgt).astype(float).tolist(),
                "teme_velocity_km_s": np.asarray(v_tgt).astype(float).tolist(),
                "record": dict(record),
            }
        )

    # Sort by relative distance to reference origin
    targets.sort(key=lambda t: t["relative_distance_km"])
    if max_targets is not None:
        targets = targets[:max_targets]

    return {
        "simulation_epoch": sim_dt.isoformat(),
        "reference_state": {
            "position_teme_km": r_ref.astype(float).tolist(),
            "velocity_teme_km_s": v_ref.astype(float).tolist(),
            "radius_km": float(np.linalg.norm(r_ref)),
        },
        "target_count": len(targets),
        "targets": targets,
    }


def records_to_relative_states(
    records: Sequence[Mapping[str, Any]],
    *,
    world_range_km: float = 250.0,
    max_targets: int = 5,
    simulation_epoch: Optional[Union[datetime, str]] = None,
    reference_record: Optional[Mapping[str, Any]] = None,
    select_nearest: bool = True,
    scale_to_world_range: bool = False,
    clip_to_world_range: bool = True,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Convert orbital records into consistent Hill-frame relative positions and relative velocities.

    Uses full SGP4 propagation to a common simulation time when OMM/TLE records are provided.
    Falls back gracefully to geometric estimation if records contain only bare element values.

    Returns:
        (positions, velocities) where each is a list of 3-element lists [x, y, z].
    """
    has_sgp4_data = any(
        ("EPOCH" in r and ("MEAN_MOTION" in r or "mean_motion_rev_per_day" in r))
        or "TLE_LINE1" in r
        or "LINE1" in r
        for r in records
    )

    if has_sgp4_data:
        try:
            prop_result = propagate_records_to_hill_frame(
                records,
                simulation_epoch=simulation_epoch,
                reference_record=reference_record,
                max_targets=max_targets if not select_nearest else None,
            )
            raw_targets = prop_result.get("targets", [])
            if select_nearest:
                raw_targets.sort(key=lambda t: t["relative_distance_km"])
            selected = raw_targets[:max_targets]

            if selected:
                positions: List[List[float]] = []
                velocities: List[List[float]] = []
                for tgt in selected:
                    pos = np.asarray(tgt["relative_position_km"], dtype=np.float32)
                    vel = np.asarray(tgt["relative_velocity_km_s"], dtype=np.float32)

                    if scale_to_world_range:
                        max_coord = float(np.max(np.abs(pos)))
                        if max_coord > world_range_km > 0.0:
                            pos = (pos / max_coord) * (0.85 * world_range_km)

                    if clip_to_world_range:
                        pos = np.clip(pos, -world_range_km, world_range_km)

                    positions.append(pos.astype(float).tolist())
                    velocities.append(vel.astype(float).tolist())

                return positions, velocities
        except Exception as exc:
            LOGGER.warning("SGP4 Hill-frame propagation failed (%s), using geometric fallback.", exc)

    # Backward-compatible geometric proxy fallback
    positions_fallback: List[List[float]] = []
    velocities_fallback: List[List[float]] = []
    for record in records[:max_targets]:
        inclination = np.deg2rad(float(record.get("inclination_deg", record.get("INCLINATION", 0.0))))
        raan = np.deg2rad(float(record.get("raan_deg", record.get("RA_OF_ASC_NODE", 0.0))))
        mean_anomaly = np.deg2rad(float(record.get("mean_anomaly_deg", record.get("MEAN_ANOMALY", 0.0))))
        mean_motion = float(record.get("mean_motion_rev_per_day", record.get("MEAN_MOTION", 14.5)))

        radial_scale = np.clip(mean_motion / 16.5, 0.25, 1.0)
        x = float(world_range_km * radial_scale * np.cos(mean_anomaly))
        y = float(world_range_km * radial_scale * np.sin(mean_anomaly))
        z = float(world_range_km * 0.35 * np.sin(inclination) * np.cos(raan))

        # Periodic ellipse velocity relation for closed relative motion
        n_proxy = 0.0011  # ~500 km LEO mean motion in rad/s
        vx = 0.0
        vy = float(-2.0 * n_proxy * x)
        vz = 0.0

        positions_fallback.append([x, y, z])
        velocities_fallback.append([vx, vy, vz])

    return positions_fallback, velocities_fallback


def records_to_relative_positions(
    records: Sequence[Mapping[str, Any]],
    *,
    world_range_km: float = 250.0,
    max_targets: int = 5,
    simulation_epoch: Optional[Union[datetime, str]] = None,
    reference_record: Optional[Mapping[str, Any]] = None,
    select_nearest: bool = True,
    scale_to_world_range: bool = False,
    clip_to_world_range: bool = True,
) -> List[List[float]]:
    """
    Convert orbital records into approximate relative-frame Cartesian positions.

    Upgraded to use SGP4 propagation into the consistent Hill / LVLH reference frame
    at the same simulation time, with seamless geometric fallback for legacy element dicts.
    """
    positions, _ = records_to_relative_states(
        records,
        world_range_km=world_range_km,
        max_targets=max_targets,
        simulation_epoch=simulation_epoch,
        reference_record=reference_record,
        select_nearest=select_nearest,
        scale_to_world_range=scale_to_world_range,
        clip_to_world_range=clip_to_world_range,
    )
    return positions


# ==============================================================================
# 6. Data Acquisition: CelesTrak GP API & Space-Track Client
# ==============================================================================

def fetch_celestrak_records(
    *,
    query_param: str = "NAME",
    query_value: str = "COSMOS 2251 DEB",
    format: str = "json",
    session: Optional[requests.Session] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Download documented General Perturbations (GP) orbital records from CelesTrak.

    CelesTrak GP queries support modern CCSDS OMM formats (JSON, CSV, XML, KVN)
    and accommodate catalog identifiers exceeding 5 digits (up to 9 digits).

    Query Parameters:
    - CATNR: Single catalog number (1 to 9 digits)
    - NAME: Satellite or debris name filter
    - GROUP: Group of satellites (e.g. 'active', 'stations', 'visual')
    - INTDES: International designator (YYYY-NNN)
    - SPECIAL: Special sets ('GPZ', 'GPZ-PLUS', 'DECAYING')

    Format Options:
    - 'json' / 'json-pretty': CCSDS OMM JSON array (recommended)
    - 'csv': CCSDS OMM CSV
    - 'tle': Legacy 3LE/2LE
    """
    url = f"{CELESTRAK_GP_URL}?{query_param.upper()}={requests.utils.quote(query_value)}&FORMAT={format.upper()}"
    client = session or requests

    try:
        response = client.get(
            url,
            headers={"User-Agent": "orbital-debris-marl/2.0 (Astrodynamics Research)"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"CelesTrak GP query failed ({url}): {exc}") from exc

    fmt_lower = format.lower()
    if "json" in fmt_lower:
        try:
            data = response.json()
            if isinstance(data, list):
                return [dict(item) for item in data]
            if isinstance(data, dict):
                return [dict(data)]
            raise ValueError(f"Unexpected JSON layout: {type(data).__name__}")
        except Exception as exc:
            snippet = response.text[:200].replace("\n", " ")
            raise RuntimeError(f"Invalid JSON returned from CelesTrak ({snippet!r}): {exc}") from exc

    if fmt_lower == "csv":
        lines = response.text.splitlines()
        reader = csv.DictReader(lines)
        return [dict(row) for row in reader]

    if fmt_lower in ("tle", "2le", "3le"):
        # Parse TLE text block into record dicts
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        records: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
                records.append({"TLE_LINE1": lines[i], "TLE_LINE2": lines[i + 1]})
                i += 2
            elif (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1 ")
                and lines[i + 2].startswith("2 ")
            ):
                records.append({
                    "OBJECT_NAME": lines[i],
                    "TLE_LINE0": lines[i],
                    "TLE_LINE1": lines[i + 1],
                    "TLE_LINE2": lines[i + 2],
                })
                i += 3
            else:
                i += 1
        return records

    raise ValueError(f"Unsupported CelesTrak format requested: {format}")


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


def is_cache_fresh(cache_path: Union[Path, str], ttl_hours: float) -> bool:
    """Use file age as a freshness signal for API rate-limit and offline protection."""
    path = Path(cache_path)
    if not path.exists():
        return False
    elapsed_seconds = max(0.0, time.time() - path.stat().st_mtime)
    return elapsed_seconds <= ttl_hours * 3600.0


def load_cached_records(cache_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """Load cached records from JSON or CSV."""
    path = Path(cache_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload["records"] if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Unsupported JSON cache layout in {path}.")
        return [dict(record) for record in records]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    raise ValueError(f"Unsupported cache format for {path}. Use .json or .csv.")


def cache_records(records: Sequence[Mapping[str, Any]], cache_path: Union[Path, str]) -> None:
    """Persist filtered records locally for reproducible experiments."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

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


def fetch_and_cache_tles(
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    query_path: str = DEFAULT_QUERY_PATH,
    ttl_hours: float = DEFAULT_CACHE_TTL_HOURS,
    cluster_keywords: Sequence[str] = HIGH_RISK_KEYWORDS,
    force_refresh: bool = False,
    source: str = "auto",
    celestrak_query_param: str = "NAME",
    celestrak_query_value: str = "COSMOS 2251 DEB",
) -> List[Dict[str, Any]]:
    """
    Acquire high-risk debris orbital records with automatic cache fallback.

    Supports both CelesTrak GP (OMM format, no API key needed) and Space-Track.
    When a live request fails but a cache exists, the cache is returned so the
    MARL pipeline remains usable offline and under network restrictions.
    """
    cache_path = cache_path.resolve()
    if source == "cache" or (not force_refresh and is_cache_fresh(cache_path, ttl_hours)):
        LOGGER.info("Using cached orbital data from %s", cache_path)
        return load_cached_records(cache_path)

    # 1. Try CelesTrak if requested or in auto mode without Space-Track credentials
    if source in ("celestrak", "auto"):
        try:
            LOGGER.info("Attempting CelesTrak GP query: %s=%s", celestrak_query_param, celestrak_query_value)
            records = fetch_celestrak_records(
                query_param=celestrak_query_param,
                query_value=celestrak_query_value,
                format="json",
            )
            filtered = filter_high_risk_clusters(records, cluster_keywords=cluster_keywords)
            records_to_cache = filtered if filtered else records
            if records_to_cache:
                cache_records(records_to_cache, cache_path)
                LOGGER.info("Successfully acquired %d records from CelesTrak.", len(records_to_cache))
                return records_to_cache
        except Exception as exc:
            LOGGER.warning("CelesTrak fetch attempt failed: %s", exc)
            if source == "celestrak" and cache_path.exists():
                LOGGER.info("Falling back to local cache at %s", cache_path)
                return load_cached_records(cache_path)

    # 2. Try Space-Track if requested or available
    if source in ("space-track", "auto"):
        try:
            username, password = load_credentials()
            with requests.Session() as session:
                session.headers.update({"User-Agent": "orbital-debris-marl/2.0"})
                login_to_space_track(session, username=username, password=password)
                records = fetch_tle_records(session, query_path=query_path)
            filtered_records = filter_high_risk_clusters(
                records, cluster_keywords=cluster_keywords
            )
            if not filtered_records:
                raise RuntimeError("Space-Track query succeeded but yielded no high-risk records.")
            cache_records(filtered_records, cache_path)
            LOGGER.info("Cached %d filtered Space-Track records to %s", len(filtered_records), cache_path)
            return filtered_records
        except Exception as exc:
            LOGGER.warning("Space-Track fetch attempt failed: %s", exc)

    # 3. Defensive fallback to local cache
    if cache_path.exists():
        LOGGER.info("Using cached orbital data from %s following network fallback.", cache_path)
        return load_cached_records(cache_path)

    raise RuntimeError(
        f"Unable to acquire orbital data and no local cache was found at {cache_path}."
    )


# ==============================================================================
# 7. CLI Entry Point
# ==============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch, cache, and propagate CelesTrak/Space-Track orbital data using SGP4."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Local JSON or CSV cache path for orbital records.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "celestrak", "space-track", "cache"],
        default="auto",
        help="Orbital data source (default: auto).",
    )
    parser.add_argument(
        "--celestrak-query",
        default="NAME=COSMOS 2251 DEB",
        help="CelesTrak query formatted as PARAM=VALUE (e.g. NAME=COSMOS 2251 DEB or GROUP=active).",
    )
    parser.add_argument(
        "--query-path",
        default=DEFAULT_QUERY_PATH,
        help="Space-Track query path.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=DEFAULT_CACHE_TTL_HOURS,
        help="Reuse cache when newer than this threshold (hours).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass freshness check and force live network query.",
    )
    parser.add_argument(
        "--cluster-keyword",
        dest="cluster_keywords",
        action="append",
        default=[],
        help="Additional keyword used to filter high-risk clusters.",
    )
    parser.add_argument(
        "--propagate",
        action="store_true",
        help="Run SGP4 propagation on acquired records and display relative Hill-frame coordinates.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=5,
        help="Max targets to display when propagating.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_argument_parser().parse_args()
    keywords = tuple(HIGH_RISK_KEYWORDS) + tuple(args.cluster_keywords)

    ct_param, ct_val = "NAME", "COSMOS 2251 DEB"
    if "=" in args.celestrak_query:
        ct_param, ct_val = args.celestrak_query.split("=", 1)

    try:
        records = fetch_and_cache_tles(
            cache_path=args.cache_path,
            query_path=args.query_path,
            ttl_hours=args.ttl_hours,
            cluster_keywords=keywords,
            force_refresh=args.force_refresh,
            source=args.source,
            celestrak_query_param=ct_param,
            celestrak_query_value=ct_val,
        )
    except Exception as exc:
        LOGGER.error("Unable to acquire orbital data: %s", exc)
        return 1

    LOGGER.info("Loaded %d orbital records from cache/API.", len(records))

    if args.propagate and records:
        try:
            positions, velocities = records_to_relative_states(
                records,
                world_range_km=250.0,
                max_targets=args.max_targets,
            )
            LOGGER.info("SGP4 Relative Propagation Results (Hill / LVLH frame):")
            for idx, (pos, vel) in enumerate(zip(positions, velocities)):
                dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                name = records[idx].get("OBJECT_NAME", f"Target_{idx}")
                cat = records[idx].get("NORAD_CAT_ID", "N/A")
                LOGGER.info(
                    "  [%d] %s (#%s): r = [%.2f, %.2f, %.2f] km (dist: %.2f km), v = [%.4f, %.4f, %.4f] km/s",
                    idx, name, cat, pos[0], pos[1], pos[2], dist, vel[0], vel[1], vel[2],
                )
        except Exception as exc:
            LOGGER.error("Propagation failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
