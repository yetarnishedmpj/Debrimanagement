import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fetch_data import (
    ServicingSpacecraftModel,
    datetime_to_jd,
    fetch_celestrak_records,
    hill_to_teme_frame,
    is_cache_fresh,
    jd_to_datetime,
    parse_epoch,
    parse_omm_fields,
    parse_tle_fields,
    propagate_record,
    propagate_records_to_hill_frame,
    propagate_satrec,
    record_to_satrec,
    records_to_relative_positions,
    records_to_relative_states,
    teme_to_hill_frame,
)


def test_parse_tle_fields_extracts_numeric_elements():
    record = {
        "OBJECT_NAME": "IRIDIUM 33 DEB",
        "TLE_LINE2": "2 24946  86.3974 172.1503 0002267 081.9097 278.2333 14.34218557123456",
    }

    parsed = parse_tle_fields(record)

    assert parsed["inclination_deg"] == 86.3974
    assert parsed["raan_deg"] == 172.1503
    assert parsed["mean_motion_rev_per_day"] == 14.34218557123456


def test_records_to_relative_positions_honors_target_limit():
    records = [
        {
            "inclination_deg": 86.4,
            "raan_deg": 172.1,
            "mean_anomaly_deg": 12.0,
            "mean_motion_rev_per_day": 14.3,
        },
        {
            "inclination_deg": 74.0,
            "raan_deg": 90.0,
            "mean_anomaly_deg": 45.0,
            "mean_motion_rev_per_day": 15.0,
        },
    ]

    positions = records_to_relative_positions(records, world_range_km=200.0, max_targets=1)

    assert len(positions) == 1
    assert len(positions[0]) == 3


def test_parse_omm_fields_supports_expanded_catalog_id():
    # 2026 CelesTrak / USSF SATCAT 6-digit and 9-digit format
    sample_omm = {
        "OBJECT_NAME": "SPACE-FENCE DEB",
        "NORAD_CAT_ID": "100982",
        "OBJECT_ID": "2026-042C",
        "EPOCH": "2026-07-15T14:30:22.500123Z",
        "MEAN_MOTION": "14.85123456",
        "ECCENTRICITY": "0.0012345",
        "INCLINATION": "51.6421",
        "RA_OF_ASC_NODE": "145.8923",
        "ARG_OF_PERICENTER": "78.1234",
        "MEAN_ANOMALY": "281.9012",
        "BSTAR": "0.000125",
        "MEAN_MOTION_DOT": "0.0000015",
        "MEAN_MOTION_DDOT": "0.0",
    }

    fields = parse_omm_fields(sample_omm)

    assert fields["OBJECT_NAME"] == "SPACE-FENCE DEB"
    assert fields["NORAD_CAT_ID"] == "100982"
    assert fields["OBJECT_ID"] == "2026-042C"
    assert fields["mean_motion_rev_per_day"] == 14.85123456
    assert fields["eccentricity"] == 0.0012345
    assert fields["inclination_deg"] == 51.6421
    assert fields["raan_deg"] == 145.8923
    assert fields["bstar"] == 0.000125
    assert fields["epoch_dt"] == datetime(2026, 7, 15, 14, 30, 22, 500123, tzinfo=timezone.utc)


def test_record_to_satrec_and_sgp4_propagation_at_recorded_epoch():
    sample_omm = {
        "OBJECT_NAME": "COSMOS 2251 DEB",
        "NORAD_CAT_ID": "33757",
        "EPOCH": "2026-03-13T03:32:45.154176",
        "MEAN_MOTION": "14.25678910",
        "ECCENTRICITY": "0.0035000",
        "INCLINATION": "74.0321",
        "RA_OF_ASC_NODE": "120.4567",
        "ARG_OF_PERICENTER": "210.1234",
        "MEAN_ANOMALY": "150.9876",
        "BSTAR": "0.000085",
        "MEAN_MOTION_DOT": "0.0000008",
        "MEAN_MOTION_DDOT": "0.0",
    }

    sat, err = record_to_satrec(sample_omm)
    assert sat is not None, f"Expected successful initialization, got error: {err}"

    epoch_dt = parse_epoch(sample_omm["EPOCH"])
    e, r, v = propagate_satrec(sat, epoch_dt)

    assert e == 0, f"SGP4 propagation returned non-zero error code {e}"
    # LEO altitude check (~7000-7200 km radius, ~7.5 km/s orbital speed)
    radius = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v))
    assert 6500.0 < radius < 7500.0
    assert 7.0 < speed < 8.0


def test_omm_9digit_catalog_id_bypasses_legacy_limit():
    # Catalog identifier exceeding 339999 (e.g., Space Fence 9-digit catalog)
    nine_digit_omm = {
        "OBJECT_NAME": "SF-FRAGMENT-9",
        "NORAD_CAT_ID": "900000001",
        "EPOCH": "2026-08-01T12:00:00.000000",
        "MEAN_MOTION": "15.0",
        "ECCENTRICITY": "0.001",
        "INCLINATION": "51.6",
        "RA_OF_ASC_NODE": "45.0",
        "ARG_OF_PERICENTER": "90.0",
        "MEAN_ANOMALY": "0.0",
        "BSTAR": "0.0001",
        "MEAN_MOTION_DOT": "0.0",
        "MEAN_MOTION_DDOT": "0.0",
    }

    sat, err = record_to_satrec(nine_digit_omm)
    assert sat is not None, f"Failed to initialize 9-digit record: {err}"

    e, r, v = propagate_satrec(sat, 0.0)  # at epoch
    assert e == 0
    assert np.linalg.norm(r) > 6500.0


def test_time_conversions_roundtrip():
    original_dt = datetime(2026, 10, 12, 17, 45, 30, 500000, tzinfo=timezone.utc)
    jd, fr = datetime_to_jd(original_dt)
    reconstructed_dt = jd_to_datetime(jd, fr)

    # Within sub-millisecond precision
    diff_seconds = abs((original_dt - reconstructed_dt).total_seconds())
    assert diff_seconds < 1e-3


def test_teme_to_hill_frame_and_inverse_consistency():
    # Servicer / Reference circular orbit in LEO
    r_ref = np.array([6878.135, 0.0, 0.0], dtype=np.float64)  # 500 km altitude
    v_ref = np.array([0.0, 7.6126, 0.0], dtype=np.float64)    # Circular speed

    # Target slightly offset in-track and radial
    r_target = np.array([6888.135, 25.0, -10.0], dtype=np.float64)
    v_target = np.array([-0.05, 7.607, 0.02], dtype=np.float64)

    # Forward transformation to Hill frame
    r_hill, v_hill = teme_to_hill_frame(r_target, v_target, r_ref, v_ref)

    # x is radial: 6888.135 - 6878.135 = 10.0 km
    assert math.isclose(r_hill[0], 10.0, abs_tol=1e-3)
    # y is in-track: 25.0 km
    assert math.isclose(r_hill[1], 25.0, abs_tol=1e-3)
    # z is cross-track: -10.0 km
    assert math.isclose(r_hill[2], -10.0, abs_tol=1e-3)

    # Inverse transformation back to TEME
    r_rec, v_rec = hill_to_teme_frame(r_hill, v_hill, r_ref, v_ref)
    assert np.allclose(r_rec, r_target, atol=1e-9)
    assert np.allclose(v_rec, v_target, atol=1e-9)


def test_co_located_target_has_zero_relative_state():
    r_ref = np.array([7000.0, 1000.0, 500.0], dtype=np.float64)
    v_ref = np.array([-1.5, 7.0, 2.0], dtype=np.float64)

    r_hill, v_hill = teme_to_hill_frame(r_ref, v_ref, r_ref, v_ref)

    assert np.allclose(r_hill, np.zeros(3), atol=1e-9)
    assert np.allclose(v_hill, np.zeros(3), atol=1e-9)


def test_propagate_records_to_hill_frame_common_epoch():
    epoch_str = "2026-03-13T03:32:45.000000"
    records = [
        {
            "OBJECT_NAME": "COSMOS 2251 PARENT",
            "NORAD_CAT_ID": "33757",
            "EPOCH": epoch_str,
            "MEAN_MOTION": "14.25",
            "ECCENTRICITY": "0.003",
            "INCLINATION": "74.0",
            "RA_OF_ASC_NODE": "120.0",
            "ARG_OF_PERICENTER": "210.0",
            "MEAN_ANOMALY": "150.0",
            "BSTAR": "0.00008",
            "MEAN_MOTION_DOT": "0.0",
            "MEAN_MOTION_DDOT": "0.0",
        },
        {
            "OBJECT_NAME": "COSMOS 2251 DEB A",
            "NORAD_CAT_ID": "33758",
            "EPOCH": epoch_str,
            "MEAN_MOTION": "14.26",
            "ECCENTRICITY": "0.0031",
            "INCLINATION": "74.01",
            "RA_OF_ASC_NODE": "120.02",
            "ARG_OF_PERICENTER": "210.05",
            "MEAN_ANOMALY": "150.1",
            "BSTAR": "0.000085",
            "MEAN_MOTION_DOT": "0.0",
            "MEAN_MOTION_DDOT": "0.0",
        },
        {
            "OBJECT_NAME": "COSMOS 2251 DEB B",
            "NORAD_CAT_ID": "33759",
            "EPOCH": epoch_str,
            "MEAN_MOTION": "14.27",
            "ECCENTRICITY": "0.0032",
            "INCLINATION": "74.02",
            "RA_OF_ASC_NODE": "120.04",
            "ARG_OF_PERICENTER": "210.1",
            "MEAN_ANOMALY": "150.2",
            "BSTAR": "0.00009",
            "MEAN_MOTION_DOT": "0.0",
            "MEAN_MOTION_DDOT": "0.0",
        },
    ]

    result = propagate_records_to_hill_frame(
        records,
        simulation_epoch=epoch_str,
        reference_record=records[0],
        max_targets=2,
    )

    assert result["simulation_epoch"].startswith("2026-03-13T03:32:45")
    assert result["target_count"] == 2
    targets = result["targets"]
    # First target is the reference itself (distance 0.0)
    assert math.isclose(targets[0]["relative_distance_km"], 0.0, abs_tol=1e-3)
    # Second target is close member
    assert targets[1]["relative_distance_km"] > 0.0


def test_records_to_relative_states_returns_positions_and_velocities():
    epoch_str = "2026-03-13T00:00:00.000000"
    records = [
        {
            "OBJECT_NAME": "SAT_1",
            "NORAD_CAT_ID": "10001",
            "EPOCH": epoch_str,
            "MEAN_MOTION": "15.0",
            "ECCENTRICITY": "0.001",
            "INCLINATION": "51.6",
            "RA_OF_ASC_NODE": "90.0",
            "ARG_OF_PERICENTER": "0.0",
            "MEAN_ANOMALY": "0.0",
            "BSTAR": "0.0001",
            "MEAN_MOTION_DOT": "0.0",
            "MEAN_MOTION_DDOT": "0.0",
        },
        {
            "OBJECT_NAME": "SAT_2",
            "NORAD_CAT_ID": "10002",
            "EPOCH": epoch_str,
            "MEAN_MOTION": "15.001",
            "ECCENTRICITY": "0.001",
            "INCLINATION": "51.601",
            "RA_OF_ASC_NODE": "90.01",
            "ARG_OF_PERICENTER": "0.0",
            "MEAN_ANOMALY": "0.05",
            "BSTAR": "0.0001",
            "MEAN_MOTION_DOT": "0.0",
            "MEAN_MOTION_DDOT": "0.0",
        },
    ]

    positions, velocities = records_to_relative_states(
        records,
        world_range_km=250.0,
        max_targets=2,
        simulation_epoch=epoch_str,
    )

    assert len(positions) == 2
    assert len(velocities) == 2
    for pos, vel in zip(positions, velocities):
        assert len(pos) == 3
        assert len(vel) == 3
        # Positions are within world range
        assert all(-250.0 <= coord <= 250.0 for coord in pos)


def test_servicing_spacecraft_model_maneuvers_and_cw_propagation():
    servicer = ServicingSpacecraftModel(
        position_km=np.array([5.0, -10.0, 2.0]),
        velocity_km_s=np.array([0.0, -0.01, 0.0]),
        fuel_fraction=1.0,
        fuel_capacity_delta_v=100.0,
        reference_altitude_km=500.0,
    )

    # Initial state check
    assert servicer.fuel_fraction == 1.0
    assert servicer.state_vector.shape == (6,)

    # Apply maneuver impulse (e.g. 0.005 km/s = 5 m/s)
    fuel_used = servicer.apply_impulse([0.0, 0.005, 0.0])
    assert fuel_used > 0.0
    assert servicer.fuel_fraction < 1.0
    assert math.isclose(servicer.velocity_km_s[1], -0.005, abs_tol=1e-5)

    # Propagate relative motion forward by 60 seconds
    init_pos = servicer.position_km.copy()
    servicer.propagate_cw(60.0)

    # Position has evolved under CW dynamics
    assert not np.allclose(servicer.position_km, init_pos)


def test_is_cache_fresh_logic(tmp_path: Path):
    cache_file = tmp_path / "test_cache.json"
    cache_file.write_text("{}", encoding="utf-8")

    assert is_cache_fresh(cache_file, ttl_hours=1.0)
    assert not is_cache_fresh(tmp_path / "non_existent.json", ttl_hours=1.0)


@patch("requests.get")
def test_fetch_celestrak_records_json(mock_get: MagicMock):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": 25544,
            "EPOCH": "2026-06-19T12:16:41.638656",
            "MEAN_MOTION": 15.49315858,
        }
    ]
    mock_get.return_value = mock_response

    records = fetch_celestrak_records(query_param="CATNR", query_value="25544", format="json")

    assert len(records) == 1
    assert records[0]["OBJECT_NAME"] == "ISS (ZARYA)"
    assert records[0]["NORAD_CAT_ID"] == 25544
    mock_get.assert_called_once()
    assert "CATNR=25544" in mock_get.call_args[0][0]
    assert "FORMAT=JSON" in mock_get.call_args[0][0]
