from fetch_data import parse_tle_fields, records_to_relative_positions


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
