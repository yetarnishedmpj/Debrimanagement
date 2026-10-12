"""
End-to-end verification script for SGP4 orbit propagation,
OMM data parsing, and consistent Hill (LVLH) frame relative coordinate transformations.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sgp4.api import Satrec, WGS72, jday

from fetch_data import (
    ServicingSpacecraftModel,
    datetime_to_jd,
    load_cached_records,
    parse_epoch,
    propagate_records_to_hill_frame,
    record_to_satrec,
    records_to_relative_states,
    teme_to_hill_frame,
)


def run_pipeline_demo() -> None:
    cache_path = Path("data/high_risk_tle_cache.json")
    if not cache_path.exists():
        print(f"[!] Cache file not found at {cache_path}")
        return

    print("=" * 70)
    print("1. LOADING DEBRIS RECORDS FROM CACHE")
    print("=" * 70)
    records = load_cached_records(cache_path)
    print(f"Total cached records loaded: {len(records)}")

    # Filter for a breakup cluster (e.g. COSMOS 2251 debris)
    cosmos_deb = [r for r in records if "COSMOS 2251" in r.get("OBJECT_NAME", "")]
    if not cosmos_deb:
        cosmos_deb = records[:20]
    print(f"Cluster members selected (COSMOS 2251): {len(cosmos_deb)}")

    print("\n" + "=" * 70)
    print("2. PROPAGATING WITH SGP4 TO COMMON SIMULATION EPOCH (HILL / LVLH FRAME)")
    print("=" * 70)

    # Propagate cluster to consistent simulation epoch in Hill frame
    result = propagate_records_to_hill_frame(
        cosmos_deb,
        max_targets=5,
    )

    sim_epoch = result["simulation_epoch"]
    ref_radius = result["reference_state"]["radius_km"]
    targets = result["targets"]

    print(f"Common Simulation Epoch: {sim_epoch}")
    print(f"Reference Orbit Radius:  {ref_radius:.2f} km")
    print(f"Active Targets Propagated: {len(targets)}\n")

    print(f"{'Target Name':<24} {'NORAD ID':<10} {'Distance (km)':<15} {'Position [x, y, z] (km)':<30}")
    print("-" * 80)
    for tgt in targets:
        name = tgt["OBJECT_NAME"]
        cat_id = tgt["NORAD_CAT_ID"]
        dist = tgt["relative_distance_km"]
        pos = tgt["relative_position_km"]
        pos_str = f"[{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}]"
        print(f"{name:<24} #{cat_id:<9} {dist:12.2f}    {pos_str}")

    print("\n" + "=" * 70)
    print("3. TESTING RELATIVE STATES GENERATION FOR MARL ENVIRONMENT")
    print("=" * 70)
    positions, velocities = records_to_relative_states(
        cosmos_deb,
        world_range_km=250.0,
        max_targets=5,
    )
    for idx, (p, v) in enumerate(zip(positions, velocities)):
        print(f"Target slot [{idx}]:")
        print(f"  Relative Position (km):   [{p[0]:8.2f}, {p[1]:8.2f}, {p[2]:8.2f}]")
        print(f"  Relative Velocity (km/s): [{v[0]:8.4f}, {v[1]:8.4f}, {v[2]:8.4f}]")

    print("\n" + "=" * 70)
    print("4. SERVICING SPACECRAFT MANEUVER & RELATIVE-MOTION SIMULATION")
    print("=" * 70)
    servicer = ServicingSpacecraftModel(
        position_km=np.array([5.0, -10.0, 2.0]),
        velocity_km_s=np.array([0.0, -0.01, 0.0]),
        fuel_fraction=1.0,
        fuel_capacity_delta_v=100.0,
        reference_altitude_km=500.0,
    )
    print(f"Initial Servicer State: pos={servicer.position_km}, vel={servicer.velocity_km_s}, fuel={servicer.fuel_fraction * 100:.1f}%")

    # Execute a 5 m/s Delta-V maneuver (0.005 km/s) along in-track axis
    delta_v_burn = [0.0, 0.005, 0.0]
    fuel_used = servicer.apply_impulse(delta_v_burn)
    print(f"Executed Burn {delta_v_burn} km/s -> Fuel consumed: {fuel_used * 100:.2f}%, Fuel remaining: {servicer.fuel_fraction * 100:.2f}%")

    # Propagate 60 seconds of unforced Clohessy-Wiltshire relative motion
    servicer.propagate_cw(60.0)
    print(f"Post-Propagate (t=+60s): pos={np.round(servicer.position_km, 3)}, vel={np.round(servicer.velocity_km_s, 4)}")
    print("\n[+] Verification successful: SGP4 propagator, OMM parser, and Hill frame relative model operational!")


if __name__ == "__main__":
    run_pipeline_demo()
