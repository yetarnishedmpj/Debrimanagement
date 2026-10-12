"""
Unit tests for non-learning baseline controllers (Nearest-Target and Fuel-Aware).
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines import (
    BaselineMetrics,
    FuelAwarePolicy,
    NearestTargetPolicy,
    compare_baselines,
    deterministic_los_burn,
    evaluate_policy,
    extract_observation_matrix,
    get_agent_fuel_fraction,
)
from orbital_env import OrbitalDebrisRemovalEnv


# ==============================================================================
# 1. Observation Extraction & Fuel Helpers
# ==============================================================================

def test_extract_observation_matrix_from_array() -> None:
    arr = np.zeros((5, 6), dtype=np.float32)
    extracted = extract_observation_matrix(arr)
    assert extracted.shape == (5, 6)
    assert extracted.dtype == np.float32


def test_extract_observation_matrix_from_dict() -> None:
    arr = np.ones((4, 10), dtype=np.float32)
    extracted = extract_observation_matrix({"local_targets": arr})
    assert extracted.shape == (4, 10)
    np.testing.assert_allclose(extracted, arr)

    extracted_obs = extract_observation_matrix({"obs": arr})
    assert extracted_obs.shape == (4, 10)


def test_get_agent_fuel_fraction_normalization() -> None:
    # Feature 5 is normalized to [-1.0, 1.0] in environment
    # -1.0 maps to 0.0 fuel, +1.0 maps to 1.0 fuel, 0.0 maps to 0.5 fuel
    matrix = np.zeros((3, 6), dtype=np.float32)

    matrix[:, 5] = 1.0  # Full fuel
    assert get_agent_fuel_fraction(matrix) == pytest.approx(1.0, abs=1e-5)

    matrix[:, 5] = -1.0  # Empty fuel
    assert get_agent_fuel_fraction(matrix) == pytest.approx(0.0, abs=1e-5)

    matrix[:, 5] = 0.0  # Half fuel
    assert get_agent_fuel_fraction(matrix) == pytest.approx(0.5, abs=1e-5)

    # Empty / small matrix fallback
    assert get_agent_fuel_fraction(np.zeros((1, 3), dtype=np.float32)) == 1.0


# ==============================================================================
# 2. Guidance Burn Mapping
# ==============================================================================

def test_deterministic_los_burn_directions() -> None:
    # Action mapping: 0 = negative burn (-1), 1 = coast (0), 2 = positive burn (+1)
    # Target is in positive direction (+0.5, +0.2, +0.1)
    rel_pos = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    burn = deterministic_los_burn(rel_pos, deadband=0.02)
    np.testing.assert_array_equal(burn, [2, 2, 2])

    # Target is in negative direction
    rel_neg = np.array([-0.5, -0.2, -0.1], dtype=np.float32)
    burn_neg = deterministic_los_burn(rel_neg, deadband=0.02)
    np.testing.assert_array_equal(burn_neg, [0, 0, 0])


def test_deterministic_los_burn_deadband() -> None:
    # Sub-deadband coordinate along x should coast (1)
    rel = np.array([0.005, 0.3, -0.3], dtype=np.float32)
    burn = deterministic_los_burn(rel, deadband=0.02)
    assert burn[0] == 1  # Coast along x
    assert burn[1] == 2  # Positive burn along y
    assert burn[2] == 0  # Negative burn along z


def test_deterministic_los_burn_damping() -> None:
    # Target is at +0.2, but velocity is closing fast (-0.5)
    # relative_pos * damping_vel = 0.2 * (-0.5) = -0.1 < -0.05
    rel = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    vel = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
    burn = deterministic_los_burn(rel, deadband=0.02, damping_vector=vel)
    # Velocity damping coasts to prevent overshooting
    assert burn[0] == 1


# ==============================================================================
# 3. Nearest-Target Heuristic Policy
# ==============================================================================

def test_nearest_target_selects_closest_active_target() -> None:
    policy = NearestTargetPolicy(deadband=0.02)
    # Observation: 3 targets
    # Feature 0: active mask m_j (1 = active, 0 = inactive)
    # Feature 1-3: relative xyz
    # Feature 4: distance
    # Feature 5: fuel
    obs = np.array([
        [1.0, 0.4, 0.0, 0.0, 0.40, 1.0],  # Target 0: dist 0.40 (active)
        [1.0, 0.1, 0.1, 0.0, 0.14, 1.0],  # Target 1: dist 0.14 (closest active)
        [1.0, 0.9, 0.0, 0.0, 0.90, 1.0],  # Target 2: dist 0.90 (active)
    ], dtype=np.float32)

    action = policy.compute_action(obs, agent_id="agent_0")
    assert action.shape == (4,)
    assert action[0] == 1  # Selected target index 1
    # Burn commands are in {0, 1, 2}
    assert all(a in (0, 1, 2) for a in action[1:])


def test_nearest_target_ignores_inactive_slots() -> None:
    policy = NearestTargetPolicy()
    obs = np.array([
        [0.0, 0.01, 0.0, 0.0, 0.01, 1.0],  # Target 0: inactive (even though closest)
        [1.0, 0.30, 0.0, 0.0, 0.30, 1.0],  # Target 1: active
    ], dtype=np.float32)

    action = policy.compute_action(obs, agent_id="agent_0")
    assert action[0] == 1  # Must skip inactive target 0 and choose active target 1


def test_nearest_target_coasts_when_no_active_targets() -> None:
    policy = NearestTargetPolicy()
    obs = np.array([
        [0.0, 0.1, 0.0, 0.0, 0.1, 1.0],
        [0.0, 0.2, 0.0, 0.0, 0.2, 1.0],
    ], dtype=np.float32)

    action = policy.compute_action(obs)
    np.testing.assert_array_equal(action, [0, 1, 1, 1])


# ==============================================================================
# 4. Fuel-Aware Heuristic Policy
# ==============================================================================

def test_fuel_aware_penalizes_cross_track_under_depleted_fuel() -> None:
    policy = FuelAwarePolicy(
        k1_phasing_weight=0.025,
        k2_plane_change_weight=0.085,
    )
    # Compare two targets:
    # Target 0: Pure in-plane phasing (x=0.3, y=0.0, z=0.0), dist = 0.30
    # Target 1: Pure out-of-plane plane-change (x=0.0, y=0.0, z=0.28), dist = 0.28 (slightly closer in distance)
    #
    # When fuel is depleted (fuel = -0.9 -> ~5% remaining fuel):
    # Plane-change cost is heavily penalized by 1 / max(f, 0.1) = 10.0
    # Fuel-aware policy must choose Target 0 (in-plane) despite Target 1 having slightly shorter Euclidean distance!
    obs_depleted = np.array([
        [1.0, 0.30, 0.0, 0.00, 0.30, -0.9],  # Target 0: in-plane
        [1.0, 0.00, 0.0, 0.28, 0.28, -0.9],  # Target 1: cross-track plane change
    ], dtype=np.float32)

    action = policy.compute_action(obs_depleted)
    assert action[0] == 0, "Fuel-aware policy should prefer in-plane target over plane-change when fuel is low"


def test_fuel_aware_gates_cross_track_burns_when_fuel_is_critical() -> None:
    policy = FuelAwarePolicy(deadband=0.02)
    # Fuel is critical: feature 5 = -0.8 -> fuel_fraction = (-0.8 + 1)/2 = 0.10 (< 0.15)
    # Target has both in-plane and cross-track offset: [0.3, 0.0, 0.4]
    obs_critical = np.array([
        [1.0, 0.30, 0.0, 0.40, 0.50, -0.8],
    ], dtype=np.float32)

    action = policy.compute_action(obs_critical)
    # action is [target_idx, burn_x, burn_y, burn_z]
    assert action[0] == 0
    assert action[1] == 2  # positive burn along x
    assert action[2] == 1  # coast along y
    assert action[3] == 1, "Cross-track z-thrust must be suppressed (coasted) under critical fuel"


def test_choose_actions_fleet_mapping() -> None:
    nearest = NearestTargetPolicy()
    fuel_aware = FuelAwarePolicy()

    fleet_obs = {
        "agent_0": np.array([[1.0, 0.1, 0.0, 0.0, 0.1, 1.0]], dtype=np.float32),
        "agent_1": np.array([[1.0, 0.0, 0.2, 0.0, 0.2, 0.0]], dtype=np.float32),
    }

    n_actions = nearest.choose_actions(fleet_obs)
    f_actions = fuel_aware.choose_actions(fleet_obs)

    assert set(n_actions.keys()) == {"agent_0", "agent_1"}
    assert set(f_actions.keys()) == {"agent_0", "agent_1"}
    for act in n_actions.values():
        assert act.shape == (4,)
        assert act.dtype in (np.int64, int)
    for act in f_actions.values():
        assert act.shape == (4,)
        assert act.dtype in (np.int64, int)


# ==============================================================================
# 5. Integration Evaluation Harness
# ==============================================================================

def test_evaluate_policy_returns_valid_metrics() -> None:
    env = OrbitalDebrisRemovalEnv({
        "num_agents": 2,
        "num_targets": 3,
        "max_targets": 3,
        "max_episode_steps": 20,
        "random_seed": 123,
    })
    policy = NearestTargetPolicy()
    metrics = evaluate_policy(
        policy,
        env,
        policy_name="Test-Nearest",
        num_episodes=2,
        seed_start=123,
    )
    env.close()

    assert isinstance(metrics, BaselineMetrics)
    assert metrics.num_episodes == 2
    assert metrics.policy_name == "Test-Nearest"
    assert 0.0 <= metrics.mean_fuel_used_pct <= 100.0
    assert 0.0 <= metrics.mean_capture_accuracy_pct <= 100.0
    assert 0.0 <= metrics.full_sweep_rate_pct <= 100.0
    assert metrics.mean_steps > 0


def test_compare_baselines_integration() -> None:
    env_config = {
        "num_agents": 2,
        "num_targets": 3,
        "max_targets": 3,
        "max_episode_steps": 15,
        "random_seed": 77,
    }
    results = compare_baselines(env_config, num_episodes=2, seed_start=77)
    assert "Nearest-Target" in results
    assert "Fuel-Aware" in results
    assert results["Nearest-Target"].num_episodes == 2
    assert results["Fuel-Aware"].num_episodes == 2
