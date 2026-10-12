import numpy as np

from orbital_env import (
    OrbitalDebrisRemovalEnv,
    compute_cw_control_matrix,
    compute_cw_stm,
)


def test_reset_observations_are_clamped_and_shaped():
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 2,
            "num_targets": 3,
            "max_targets": 4,
            "random_seed": 123,
        }
    )

    observations, infos = env.reset(seed=123)

    assert set(observations) == {"agent_0", "agent_1"}
    assert set(infos) == {"agent_0", "agent_1"}

    for agent_obs in observations.values():
        assert agent_obs.shape == (4, 6)
        assert np.all(agent_obs <= 1.0)
        assert np.all(agent_obs >= -1.0)
        # Verify explicit availability indicator: slots 0, 1, 2 are active (1.0), slot 3 is inactive (0.0)
        assert np.all(agent_obs[:3, 0] == 1.0)
        assert agent_obs[3, 0] == 0.0

    env.close()


def test_capture_grants_partial_reward_to_teammates():
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 2,
            "num_targets": 1,
            "max_targets": 1,
            "capture_radius_km": 10.0,
            "translation_step_km": 0.0,
            "progress_reward_scale": 0.0,
            "fuel_penalty_scale": 0.0,
            "idle_penalty": 0.0,
            "capture_reward": 12.0,
            "cooperative_reward_share": 3.0,
        }
    )

    env.reset(
        options={
            "agent_positions": [[0.0, 0.0, 0.0], [40.0, 40.0, 0.0]],
            "target_positions": [[1.0, 1.0, 0.0]],
        }
    )
    _, rewards, terminateds, _, infos = env.step(
        {
            "agent_0": np.asarray([0, 1, 1, 1]),
            "agent_1": np.asarray([0, 1, 1, 1]),
        }
    )

    assert rewards["agent_0"] == 12.0
    assert rewards["agent_1"] == 3.0
    assert terminateds["__all__"] is True
    assert infos["__common__"]["captures_completed"] == 1
    env.close()


def test_invalid_actions_are_safely_clipped():
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 1,
            "max_targets": 2,
            "translation_step_km": 0.0,
            "progress_reward_scale": 0.0,
            "fuel_penalty_scale": 0.0,
            "idle_penalty": 0.0,
            "capture_reward": 0.0,
            "capture_radius_km": 1.0,
            "invalid_action_penalty": 0.2,
        }
    )
    env.reset(options={"target_positions": [[20.0, 0.0, 0.0]]})

    _, rewards, _, _, _ = env.step({"agent_0": np.asarray([99, 9, -5, 1])})

    assert rewards["agent_0"] == -0.2
    env.close()


def test_simultaneous_updates_order_invariance_under_contention():
    """
    Verify that when multiple agents contend for the same target,
    reversing agent processing order yields the exact same physical outcome.
    The closer agent wins the capture regardless of processing order.
    """
    config = {
        "num_agents": 2,
        "num_targets": 1,
        "max_targets": 1,
        "capture_radius_km": 10.0,
        "translation_step_km": 0.0,
        "progress_reward_scale": 0.0,
        "fuel_penalty_scale": 0.0,
        "idle_penalty": 0.0,
        "capture_reward": 15.0,
        "cooperative_reward_share": 4.0,
    }
    options = {
        "target_positions": [[0.0, 0.0, 0.0]],
        # agent_0 is 2 km from target; agent_1 is 6 km from target.
        # Both are within capture_radius_km (10 km).
        "agent_positions": [[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
    }
    actions = {
        "agent_0": np.asarray([0, 1, 1, 1]),
        "agent_1": np.asarray([0, 1, 1, 1]),
    }
    actions_reversed_dict = {
        "agent_1": np.asarray([0, 1, 1, 1]),
        "agent_0": np.asarray([0, 1, 1, 1]),
    }

    # Run A: Forward processing order (agent_0 evaluated before agent_1)
    env_fwd = OrbitalDebrisRemovalEnv(config)
    env_fwd.reset(options=options)
    _, rewards_fwd, term_fwd, trunc_fwd, infos_fwd = env_fwd.step(
        actions, agent_order=["agent_0", "agent_1"]
    )
    history_fwd = env_fwd.get_episode_history()
    env_fwd.close()

    # Run B: Reversed processing order (agent_1 evaluated before agent_0)
    env_rev = OrbitalDebrisRemovalEnv(config)
    env_rev.reset(options=options)
    _, rewards_rev, term_rev, trunc_rev, infos_rev = env_rev.step(
        actions_reversed_dict, agent_order=["agent_1", "agent_0"]
    )
    history_rev = env_rev.get_episode_history()
    env_rev.close()

    # Physical outcome must be identical between forward and reversed order
    assert rewards_fwd == rewards_rev
    # agent_0 is closer (2.0 km vs 6.0 km), so agent_0 must win the capture in both runs
    assert rewards_fwd["agent_0"] == 15.0
    assert rewards_fwd["agent_1"] == 4.0
    assert term_fwd == term_rev
    assert trunc_fwd == trunc_rev
    assert infos_fwd == infos_rev

    # State histories must match
    np.testing.assert_array_equal(history_fwd["capture_matrix"], history_rev["capture_matrix"])
    np.testing.assert_allclose(history_fwd["agent_positions"], history_rev["agent_positions"])
    np.testing.assert_allclose(history_fwd["agent_fuel"], history_rev["agent_fuel"])


def test_tie_breaking_rule_under_identical_distances():
    """
    Verify the explicit tie-breaking rule:
    When multiple agents are at the EXACT same distance from the target,
    the deterministic tie-breaker (lower agent index) decides the winner,
    and reversing the processing order does NOT alter the winner.
    """
    config = {
        "num_agents": 2,
        "num_targets": 1,
        "max_targets": 1,
        "capture_radius_km": 10.0,
        "translation_step_km": 0.0,
        "progress_reward_scale": 0.0,
        "fuel_penalty_scale": 0.0,
        "idle_penalty": 0.0,
        "capture_reward": 15.0,
        "cooperative_reward_share": 4.0,
    }
    options = {
        "target_positions": [[0.0, 0.0, 0.0]],
        # Both agents are at exactly 3.0 km distance from target
        "agent_positions": [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
    }
    actions = {
        "agent_0": np.asarray([0, 1, 1, 1]),
        "agent_1": np.asarray([0, 1, 1, 1]),
    }

    env_fwd = OrbitalDebrisRemovalEnv(config)
    env_fwd.reset(options=options)
    _, rewards_fwd, term_fwd, _, _ = env_fwd.step(
        actions, agent_order=["agent_0", "agent_1"]
    )
    env_fwd.close()

    env_rev = OrbitalDebrisRemovalEnv(config)
    env_rev.reset(options=options)
    _, rewards_rev, term_rev, _, _ = env_rev.step(
        {"agent_1": actions["agent_1"], "agent_0": actions["agent_0"]},
        agent_order=["agent_1", "agent_0"],
    )
    env_rev.close()

    # In both runs, lower agent index (agent_0) breaks the tie deterministically
    assert rewards_fwd["agent_0"] == 15.0
    assert rewards_fwd["agent_1"] == 4.0
    assert rewards_rev["agent_0"] == 15.0
    assert rewards_rev["agent_1"] == 4.0
    assert term_fwd["__all__"] is True
    assert term_rev["__all__"] is True


def test_simultaneous_multi_target_capture_no_preemption():
    """
    Verify that when two agents capture distinct targets in the same step,
    neither agent's action preempts the other, and the outcome is identical
    regardless of processing order.
    """
    config = {
        "num_agents": 2,
        "num_targets": 2,
        "max_targets": 2,
        "capture_radius_km": 10.0,
        "translation_step_km": 0.0,
        "progress_reward_scale": 0.0,
        "fuel_penalty_scale": 0.0,
        "idle_penalty": 0.0,
        "capture_reward": 15.0,
        "cooperative_reward_share": 4.0,
    }
    options = {
        "target_positions": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        "agent_positions": [[1.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
    }
    actions = {
        "agent_0": np.asarray([0, 1, 1, 1]),  # target 0
        "agent_1": np.asarray([1, 1, 1, 1]),  # target 1
    }

    env_fwd = OrbitalDebrisRemovalEnv(config)
    env_fwd.reset(options=options)
    _, rewards_fwd, term_fwd, _, infos_fwd = env_fwd.step(
        actions, agent_order=["agent_0", "agent_1"]
    )
    env_fwd.close()

    env_rev = OrbitalDebrisRemovalEnv(config)
    env_rev.reset(options=options)
    _, rewards_rev, term_rev, _, infos_rev = env_rev.step(
        {"agent_1": actions["agent_1"], "agent_0": actions["agent_0"]},
        agent_order=["agent_1", "agent_0"],
    )
    env_rev.close()

    # Both captures succeed in both runs: 15.0 capture + 4.0 cooperative share = 19.0 each
    assert rewards_fwd["agent_0"] == 19.0
    assert rewards_fwd["agent_1"] == 19.0
    assert rewards_rev["agent_0"] == 19.0
    assert rewards_rev["agent_1"] == 19.0
    assert term_fwd["__all__"] is True
    assert term_rev["__all__"] is True
    assert infos_fwd["__common__"]["captures_completed"] == 2
    assert infos_rev["__common__"]["captures_completed"] == 2


def test_target_resolution_no_op_policy_and_action_mask():
    """
    Verify the scientific 'no_op' deliberate target resolution policy:
    1. Selecting an inactive target does NOT silently redirect.
    2. The agent is penalized with invalid_action_penalty.
    3. No target is selected, no progress reward is awarded, and no capture occurs.
    4. Action masks accurately report active/inactive status.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 1,
            "max_targets": 2,
            "target_resolution_policy": "no_op",
            "capture_radius_km": 15.0,
            "translation_step_km": 0.0,
            "progress_reward_scale": 0.0,
            "fuel_penalty_scale": 0.0,
            "idle_penalty": 0.0,
            "capture_reward": 15.0,
            "invalid_action_penalty": 0.2,
        }
    )
    # Target 0 is active at [10, 0, 0], Target 1 is inactive
    obs, infos = env.reset(options={"target_positions": [[10.0, 0.0, 0.0]]})

    # Verify action mask: slot 0 is True, slot 1 is False
    mask = env.get_action_mask("agent_0")
    assert mask[0] is True or mask[0] == 1
    assert mask[1] is False or mask[1] == 0
    assert infos["agent_0"]["action_mask"] == [True, False]

    # Agent requests inactive target 1 (action = [1, 1, 1, 1])
    # The agent is at [0, 0, 0], which is only 10 km from active Target 0 (within capture radius 15 km)
    _, rewards, terminateds, _, step_infos = env.step({"agent_0": np.asarray([1, 1, 1, 1])})

    # Under 'no_op':
    # 1. Agent should NOT be silently redirected to Target 0
    # 2. Agent must NOT capture Target 0
    assert step_infos["agent_0"]["selected_target_index"] is None
    assert step_infos["agent_0"]["invalid_target_selection"] is True
    assert step_infos["agent_0"]["capture_event"] is False
    # 3. Agent is penalized for selecting an unavailable target
    assert rewards["agent_0"] == -0.2
    # 4. Target 0 remains active and uncaptured
    assert step_infos["__common__"]["captures_completed"] == 0
    assert step_infos["__common__"]["active_targets_remaining"] == 1
    assert terminateds["__all__"] is False

    env.close()


def test_target_resolution_nearest_agent_policy():
    """
    Verify the 'nearest_agent' deliberate fallback policy:
    When an inactive target is selected, the fallback redirects to the active
    target closest to the AGENT'S position, NOT the target closest to the origin.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 3,
            "max_targets": 4,
            "target_resolution_policy": "nearest_agent",
            "translation_step_km": 0.0,
            "progress_reward_scale": 0.0,
            "fuel_penalty_scale": 0.0,
            "idle_penalty": 0.0,
            "capture_reward": 15.0,
            "capture_radius_km": 50.0,
        }
    )
    # Targets layout:
    # Slot 0: [100.0, 0.0, 0.0]  (Distance to origin = 100 km)
    # Slot 1: [-100.0, 0.0, 0.0] (Distance to origin = 100 km)
    # Slot 2: [0.0, 30.0, 0.0]   (Distance to origin = 30 km -> closest to origin!)
    # Slot 3: inactive
    # Agent position: [85.0, 0.0, 0.0]
    # -> Distance to Slot 0 is 15 km (closest to agent!)
    # -> Distance to Slot 2 is sqrt(85^2 + 30^2) = 90.1 km
    env.reset(
        options={
            "agent_positions": [[85.0, 0.0, 0.0]],
            "target_positions": [
                [100.0, 0.0, 0.0],
                [-100.0, 0.0, 0.0],
                [0.0, 30.0, 0.0],
            ],
        }
    )

    # Agent requests inactive slot 3
    # Origin-based fallback would select Slot 2 (norm = 30 km).
    # Nearest-agent policy MUST select Slot 0 (distance to agent = 15 km).
    _, rewards, _, _, step_infos = env.step({"agent_0": np.asarray([3, 1, 1, 1])})

    assert step_infos["agent_0"]["selected_target_index"] == 0
    # And since Slot 0 is within capture radius 50 km (distance is 15 km), it captures Slot 0
    assert step_infos["agent_0"]["capture_event"] is True
    assert rewards["agent_0"] == 15.0

    env.close()


def test_seed_reproducibility_across_identical_episodes():
    """
    Stage 1 Completion Criterion:
    Verify that the same initial seed reproduces the exact same episode:
    identical observations, trajectories, fuel depletion, rewards, and info dicts.
    """
    config = {
        "num_agents": 3,
        "num_targets": 5,
        "max_targets": 5,
        "max_episode_steps": 30,
        "random_seed": 777,
    }

    # Deterministic sequence of pseudo-random actions for 10 steps
    rng = np.random.default_rng(999)
    action_sequence = []
    for _ in range(10):
        step_actions = {
            f"agent_{i}": np.asarray(
                [rng.integers(0, 5), rng.integers(0, 3), rng.integers(0, 3), rng.integers(0, 3)],
                dtype=np.int64,
            )
            for i in range(3)
        }
        action_sequence.append(step_actions)

    # Run A
    env_a = OrbitalDebrisRemovalEnv(config)
    obs_a, infos_a = env_a.reset(seed=777)
    rewards_a, terms_a, truncs_a, step_infos_a = [], [], [], []
    for actions in action_sequence:
        o, r, t, tr, info = env_a.step(actions)
        rewards_a.append(r)
        terms_a.append(t)
        truncs_a.append(tr)
        step_infos_a.append(info)
    hist_a = env_a.get_episode_history()
    env_a.close()

    # Run B (independent instance with same initial seed)
    env_b = OrbitalDebrisRemovalEnv(config)
    obs_b, infos_b = env_b.reset(seed=777)
    rewards_b, terms_b, truncs_b, step_infos_b = [], [], [], []
    for actions in action_sequence:
        o, r, t, tr, info = env_b.step(actions)
        rewards_b.append(r)
        terms_b.append(t)
        truncs_b.append(tr)
        step_infos_b.append(info)
    hist_b = env_b.get_episode_history()
    env_b.close()

    # Verify bitwise identity between Run A and Run B
    for agent_id in obs_a:
        np.testing.assert_array_equal(obs_a[agent_id], obs_b[agent_id])
    assert rewards_a == rewards_b
    assert terms_a == terms_b
    assert truncs_a == truncs_b
    assert step_infos_a == step_infos_b

    np.testing.assert_array_equal(hist_a["agent_positions"], hist_b["agent_positions"])
    np.testing.assert_array_equal(hist_a["agent_fuel"], hist_b["agent_fuel"])
    np.testing.assert_array_equal(hist_a["target_positions"], hist_b["target_positions"])
    np.testing.assert_array_equal(hist_a["target_active"], hist_b["target_active"])
    np.testing.assert_array_equal(hist_a["step_rewards"], hist_b["step_rewards"])
    np.testing.assert_array_equal(hist_a["fuel_cost_proxy"], hist_b["fuel_cost_proxy"])
    np.testing.assert_array_equal(hist_a["delta_v"], hist_b["delta_v"])

    # Contrast with different seed (seed=888 must yield different initial conditions)
    env_c = OrbitalDebrisRemovalEnv(config)
    obs_c, _ = env_c.reset(seed=888)
    assert not np.array_equal(obs_a["agent_0"], obs_c["agent_0"])
    env_c.close()


def test_fuel_cost_proxy_semantics_and_plane_change_penalty():
    """
    Stage 1 Completion Criterion:
    Verify fuel-cost proxy semantics:
    - _fuel_cost_proxy produces positive non-zero cost for translations.
    - Out-of-plane plane change is significantly more expensive than in-plane (k2 > k1).
    - Cosine alignment penalty penalizes burns oriented away from target.
    - _lambert_proxy_delta_v alias returns identical output.
    - infos and history expose fuel_cost_proxy.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 2,
            "k1_phasing_cost": 0.025,
            "k2_plane_change_cost": 0.085,
        }
    )
    agent_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

    # Target 1: 50 km purely in-plane [50, 0, 0]
    target_in_plane = np.asarray([50.0, 0.0, 0.0], dtype=np.float32)
    # Target 2: 50 km purely out-of-plane [0, 0, 50]
    target_cross_track = np.asarray([0.0, 0.0, 50.0], dtype=np.float32)

    burn_neutral = np.zeros(3, dtype=np.float32)
    cost_in_plane = env._fuel_cost_proxy(
        agent_position=agent_pos, target_position=target_in_plane, burn_vector=burn_neutral
    )
    cost_cross_track = env._fuel_cost_proxy(
        agent_position=agent_pos, target_position=target_cross_track, burn_vector=burn_neutral
    )

    # k2 (0.085) * 50 = 4.25; k1 (0.025) * 50 = 1.25. Plane change must be 3.4x more expensive
    assert cost_cross_track > cost_in_plane
    assert np.isclose(cost_cross_track / cost_in_plane, 0.085 / 0.025, rtol=1e-3)

    # Alignment penalty test: burning directly toward target vs burning directly away
    burn_toward = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
    burn_away = np.asarray([-10.0, 0.0, 0.0], dtype=np.float32)
    cost_toward = env._fuel_cost_proxy(
        agent_position=agent_pos, target_position=target_in_plane, burn_vector=burn_toward
    )
    cost_away = env._fuel_cost_proxy(
        agent_position=agent_pos, target_position=target_in_plane, burn_vector=burn_away
    )
    # Burning away incurs cosine alignment penalty
    assert cost_away > cost_toward

    # Alias check
    alias_cost = env._lambert_proxy_delta_v(
        agent_position=agent_pos, target_position=target_in_plane, burn_vector=burn_toward
    )
    assert alias_cost == cost_toward

    # Step execution check: infos dictionary must expose both fuel_cost_proxy and delta_v
    env.reset(options={"target_positions": [[20.0, 0.0, 0.0]], "agent_positions": [[0.0, 0.0, 0.0]]})
    _, _, _, _, step_infos = env.step({"agent_0": np.asarray([0, 2, 1, 1])})
    assert "fuel_cost_proxy" in step_infos["agent_0"]
    assert "delta_v" in step_infos["agent_0"]
    assert step_infos["agent_0"]["fuel_cost_proxy"] == step_infos["agent_0"]["delta_v"]
    assert step_infos["agent_0"]["fuel_cost_proxy"] > 0.0

    env.close()


def test_rendezvous_arrival_operational_semantics():
    """
    Stage 1 Completion Criterion:
    Verify operational definition:
    - Entering within capture_radius_km simulates successful rendezvous arrival.
    - Exposes rendezvous_event, rendezvous_completed, and rendezvous_target_index.
    - Target is retired from active set (satisfied rendezvous objective).
    - Agent remains operational with residual fuel to pursue further targets.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 2,
            "max_targets": 2,
            "capture_radius_km": 12.0,
            "translation_step_km": 0.0,
            "initial_fuel": 1.0,
            "capture_reward": 15.0,
        }
    )
    # Agent at [0, 0, 0], Target 0 at [5, 0, 0] (within 12 km), Target 1 at [100, 0, 0]
    env.reset(
        options={
            "agent_positions": [[0.0, 0.0, 0.0]],
            "target_positions": [[5.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        }
    )
    # Target 0 tracked
    _, rewards, terms, truncs, step_infos = env.step({"agent_0": np.asarray([0, 1, 1, 1])})

    assert step_infos["agent_0"]["rendezvous_event"] is True
    assert step_infos["agent_0"]["capture_event"] is True
    assert step_infos["agent_0"]["rendezvous_target_index"] == 0
    assert step_infos["__common__"]["rendezvous_completed"] == 1
    assert step_infos["__common__"]["captures_completed"] == 1
    assert step_infos["__common__"]["active_targets_remaining"] == 1

    # Agent is still alive and has residual fuel for Target 1
    assert terms["__all__"] is False
    assert step_infos["agent_0"]["fuel_remaining"] > 0.0

    env.close()


def test_multistep_episode_order_invariance():
    """
    Stage 1 Completion Criterion:
    Verify that across a multi-step trajectory with multiple agents contending
    for targets, reversing the agent processing order yields 100% identical
    trajectories, rewards, and environment states.
    """
    config = {
        "num_agents": 3,
        "num_targets": 3,
        "max_targets": 3,
        "max_episode_steps": 10,
        "capture_radius_km": 15.0,
        "translation_step_km": 10.0,
        "random_seed": 42,
    }
    actions_step1 = {
        "agent_0": np.asarray([0, 2, 1, 1]),
        "agent_1": np.asarray([0, 1, 2, 1]),
        "agent_2": np.asarray([1, 1, 1, 2]),
    }
    actions_step2 = {
        "agent_0": np.asarray([0, 2, 1, 1]),
        "agent_1": np.asarray([1, 1, 2, 1]),
        "agent_2": np.asarray([2, 1, 1, 2]),
    }
    actions_step3 = {
        "agent_0": np.asarray([2, 2, 1, 1]),
        "agent_1": np.asarray([1, 1, 2, 1]),
        "agent_2": np.asarray([2, 1, 1, 2]),
    }

    # Run Forward: order = agent_0, agent_1, agent_2
    env_fwd = OrbitalDebrisRemovalEnv(config)
    env_fwd.reset(seed=42)
    _, r1_fwd, _, _, _ = env_fwd.step(actions_step1, agent_order=["agent_0", "agent_1", "agent_2"])
    _, r2_fwd, _, _, _ = env_fwd.step(actions_step2, agent_order=["agent_0", "agent_1", "agent_2"])
    _, r3_fwd, _, _, _ = env_fwd.step(actions_step3, agent_order=["agent_0", "agent_1", "agent_2"])
    hist_fwd = env_fwd.get_episode_history()
    env_fwd.close()

    # Run Reversed: order = agent_2, agent_1, agent_0
    env_rev = OrbitalDebrisRemovalEnv(config)
    env_rev.reset(seed=42)
    _, r1_rev, _, _, _ = env_rev.step(actions_step1, agent_order=["agent_2", "agent_1", "agent_0"])
    _, r2_rev, _, _, _ = env_rev.step(actions_step2, agent_order=["agent_2", "agent_1", "agent_0"])
    _, r3_rev, _, _, _ = env_rev.step(actions_step3, agent_order=["agent_2", "agent_1", "agent_0"])
    hist_rev = env_rev.get_episode_history()
    env_rev.close()

    assert r1_fwd == r1_rev
    assert r2_fwd == r2_rev
    assert r3_fwd == r3_rev

    np.testing.assert_array_equal(hist_fwd["agent_positions"], hist_rev["agent_positions"])
    np.testing.assert_array_equal(hist_fwd["agent_fuel"], hist_rev["agent_fuel"])
    np.testing.assert_array_equal(hist_fwd["target_active"], hist_rev["target_active"])
    np.testing.assert_array_equal(hist_fwd["step_rewards"], hist_rev["step_rewards"])


def test_cw_stm_mathematical_properties_and_orbit_closure():
    """
    Stage 2 Physics Milestone:
    Verify fundamental astrodynamic properties of the Clohessy-Wiltshire STM:
    1. Identity at delta_t = 0: Phi(0) == I_6.
    2. Symplectic / Hamiltonian volume preservation: det(Phi(tau)) == 1.0.
    3. Periodic orbit closure: Over one orbital period T = 2*pi/n,
       an initial condition satisfying dy0/dt = -2*n*x0 (zero secular drift)
       closes exactly into a 2:1 relative ellipse: s(T) == s(0).
    """
    # 500 km altitude LEO
    r0 = 6378.137 + 500.0
    mu = 398600.4418
    n = np.sqrt(mu / (r0**3))
    orbital_period = 2.0 * np.pi / n  # ~5676.8 seconds

    # 1. Identity at delta_t = 0
    stm_0 = compute_cw_stm(n, 0.0)
    np.testing.assert_allclose(stm_0, np.eye(6, dtype=np.float32), atol=1e-6)

    # 2. Volume preservation (det = 1.0)
    stm_step = compute_cw_stm(n, 60.0)
    det_step = float(np.linalg.det(stm_step))
    assert np.isclose(det_step, 1.0, atol=1e-4)

    # 3. Periodic relative orbit closure
    stm_period = compute_cw_stm(n, orbital_period)
    # State with initial radial displacement x0 = 5.0 km, periodic along-track velocity vy0 = -2*n*x0
    x0, y0, z0 = 5.0, 10.0, 2.0
    vx0, vy0, vz0 = 0.0, -2.0 * n * x0, 0.0
    s_0 = np.asarray([x0, y0, z0, vx0, vy0, vz0], dtype=np.float32)
    s_T = stm_period @ s_0

    # s(T) must equal s_0 within astrodynamic integration tolerance
    np.testing.assert_allclose(s_T, s_0, atol=1e-3)


def test_dynamical_state_representation():
    """
    Stage 2 Physics Milestone:
    Verify 6-DOF dynamical state representation s_t = [r_x, r_y, r_z, v_x, v_y, v_z]^T:
    1. get_agent_state() returns 6-DOF vector [r_x, r_y, r_z, v_x, v_y, v_z].
    2. get_target_state() returns 6-DOF vector [r_x, r_y, r_z, v_x, v_y, v_z].
    3. get_fleet_states() returns all agents and targets.
    4. Step info dictionaries expose agent_state, agent_velocity, and target relative velocity.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 2,
            "num_targets": 2,
            "max_targets": 2,
            "dynamics_model": "clohessy_wiltshire",
        }
    )
    env.reset(
        options={
            "agent_positions": [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]],
            "target_positions": [[5.0, 20.0, 1.0], [-5.0, -20.0, -1.0]],
        }
    )

    # 1. Agent 6-DOF states
    agent_0_state = env.get_agent_state(0)
    assert agent_0_state.shape == (6,)
    # Position matches initial position
    np.testing.assert_allclose(agent_0_state[:3], [10.0, 0.0, 0.0])
    # Periodic initial along-track velocity vy0 = -2 * n * x0
    expected_vy0 = -2.0 * env.mean_motion_n * 10.0
    assert np.isclose(agent_0_state[4], expected_vy0, atol=1e-5)

    # Can also query by string agent_id
    np.testing.assert_array_equal(env.get_agent_state("agent_0"), agent_0_state)

    # 2. Target 6-DOF states
    target_0_state = env.get_target_state(0)
    assert target_0_state.shape == (6,)
    np.testing.assert_allclose(target_0_state[:3], [5.0, 20.0, 1.0])
    expected_t_vy0 = -2.0 * env.mean_motion_n * 5.0
    assert np.isclose(target_0_state[4], expected_t_vy0, atol=1e-5)

    # 3. Fleet states
    fleet = env.get_fleet_states()
    assert "agent_0" in fleet and "agent_1" in fleet and "targets" in fleet
    assert fleet["targets"].shape == (2, 6)

    # 4. Info dictionary enrichment
    _, _, _, _, step_infos = env.step(
        {
            "agent_0": np.asarray([0, 1, 1, 1]),
            "agent_1": np.asarray([1, 1, 1, 1]),
        }
    )
    assert "agent_state" in step_infos["agent_0"]
    assert len(step_infos["agent_0"]["agent_state"]) == 6
    assert "agent_velocity" in step_infos["agent_0"]
    assert len(step_infos["agent_0"]["agent_velocity"]) == 3
    assert "target_state" in step_infos["agent_0"]
    assert "relative_velocity" in step_infos["agent_0"]
    assert "relative_speed" in step_infos["agent_0"]
    assert step_infos["agent_0"]["relative_speed"] >= 0.0

    env.close()


def test_debris_locations_evolve_dynamically_across_steps():
    """
    Stage 2 Physics Milestone:
    Verify that debris locations do NOT remain fixed throughout the episode,
    but evolve dynamically according to the Clohessy-Wiltshire relative-motion equations.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 2,
            "max_targets": 2,
            "time_step_s": 20.0,
            "dynamics_model": "clohessy_wiltshire",
        }
    )
    env.reset(
        options={
            "agent_positions": [[0.0, 0.0, 0.0]],
            "target_positions": [[15.0, 50.0, 5.0], [-25.0, -30.0, -8.0]],
        }
    )

    initial_target_0 = env.get_target_state(0).copy()
    initial_target_1 = env.get_target_state(1).copy()

    # Step idle (direction [1, 1, 1] means zero thrust)
    for _ in range(5):
        env.step({"agent_0": np.asarray([0, 1, 1, 1])})

    post_target_0 = env.get_target_state(0)
    post_target_1 = env.get_target_state(1)

    # Debris must have moved dynamically: positions and velocities changed
    assert not np.allclose(initial_target_0[:3], post_target_0[:3])
    assert not np.allclose(initial_target_1[:3], post_target_1[:3])

    # Along-track displacement must have changed according to CW equations
    displacement_y_0 = abs(post_target_0[1] - initial_target_0[1])
    assert displacement_y_0 > 0.01  # Moved non-trivially in along-track

    # Episode history also tracks target state evolution
    hist = env.get_episode_history()
    assert "target_states" in hist
    assert "target_velocities" in hist
    assert hist["target_states"].shape[0] == 6  # initial + 5 steps

    env.close()


def test_bounded_impulse_control_and_physical_fuel_accounting():
    """
    Stage 2 Physics Milestone:
    Verify that control inputs represent bounded impulses with physical fuel accounting:
    1. Discrete thrust direction {0, 1, 2} maps to a bounded Delta-V impulse.
    2. Impulse magnitude is bounded by max_impulse_delta_v.
    3. Propellant is deducted physically: Delta_fuel = ||Delta_v|| / fuel_capacity_delta_v.
    4. When fuel is exhausted, delivered impulse is throttled to remaining fuel.
    """
    max_impulse = 0.005  # 5 m/s
    fuel_capacity = 0.05  # 50 m/s total capacity
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 1,
            "max_targets": 1,
            "max_impulse_delta_v": max_impulse,
            "fuel_capacity_delta_v": fuel_capacity,
            "initial_fuel": 1.0,
            "translation_step_km": 1.0,  # Positive value activates impulse mode
            "fuel_penalty_scale": 1.0,
            "progress_reward_scale": 0.0,
            "idle_penalty": 0.0,
        }
    )
    env.reset(
        options={
            "agent_positions": [[0.0, 0.0, 0.0]],
            "target_positions": [[100.0, 0.0, 0.0]],
        }
    )

    # Command burn along +x (action [0, 2, 1, 1] -> direction [1, 0, 0])
    _, rewards, _, _, step_infos = env.step({"agent_0": np.asarray([0, 2, 1, 1])})

    expected_delta_fuel = max_impulse / fuel_capacity  # 0.005 / 0.05 = 0.1
    expected_remaining_fuel = 1.0 - expected_delta_fuel  # 0.9

    actual_remaining_fuel = step_infos["agent_0"]["fuel_remaining"]
    assert np.isclose(actual_remaining_fuel, expected_remaining_fuel, atol=1e-5)
    assert np.isclose(step_infos["agent_0"]["fuel_cost_proxy"], max_impulse, atol=1e-5)
    # Fuel penalty applied to reward: -fuel_penalty_scale * delta_fuel = -0.1
    assert np.isclose(rewards["agent_0"], -expected_delta_fuel, atol=1e-5)

    env.close()


def test_observation_space_includes_relative_velocities_when_configured():
    """
    Stage 2 Physics Milestone:
    Verify observation space when obs_includes_velocity=True:
    1. Observation matrix shape is (max_targets, 10).
    2. Columns 6:9 encode relative velocity Delta_v / velocity_range.
    3. Column 9 encodes relative speed.
    4. Clamped to [-1.0, 1.0].
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 2,
            "max_targets": 3,
            "obs_includes_velocity": True,
            "velocity_range_km_s": 0.1,
        }
    )
    assert env.observation_space.shape == (3, 10)

    obs, _ = env.reset(
        options={
            "agent_positions": [[0.0, 0.0, 0.0]],
            "target_positions": [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
        }
    )
    agent_obs = obs["agent_0"]
    assert agent_obs.shape == (3, 10)

    # Target 0 and 1 active: mask = 1.0; Target 2 inactive: mask = 0.0
    assert agent_obs[0, 0] == 1.0
    assert agent_obs[1, 0] == 1.0
    assert agent_obs[2, 0] == 0.0

    # Relative velocities for active targets are non-zero (due to orbital motion)
    assert not np.all(agent_obs[0, 6:9] == 0.0)
    # Inactive target has 0.0 for relative velocity features
    assert np.all(agent_obs[2, 6:10] == 0.0)

    # All values are bounded within [-1.0, 1.0]
    assert np.all(agent_obs >= -1.0) and np.all(agent_obs <= 1.0)

    env.close()


def test_continuous_acceleration_control_mode():
    """
    Stage 2 Physics Milestone:
    Verify that continuous acceleration mode properly propagates spacecraft state
    using the discrete input response matrix Gamma(tau):
    s(t + tau) = Phi(tau)*s(t) + Gamma(tau)*a.
    """
    env = OrbitalDebrisRemovalEnv(
        {
            "num_agents": 1,
            "num_targets": 1,
            "max_targets": 1,
            "dynamics_model": "clohessy_wiltshire",
            "control_mode": "continuous_acceleration",
            "time_step_s": 10.0,
            "max_impulse_delta_v": 0.01,
            "translation_step_km": 1.0,
        }
    )
    env.reset(
        options={
            "agent_positions": [[0.0, 0.0, 0.0]],
            "target_positions": [[50.0, 0.0, 0.0]],
        }
    )

    # Command continuous acceleration burn along +x: action [0, 2, 1, 1]
    _, rewards, _, _, step_infos = env.step({"agent_0": np.asarray([0, 2, 1, 1])})

    state = env.get_agent_state(0)
    # With positive continuous acceleration in x, radial position rx and velocity vx must be positive
    assert state[0] > 0.0  # rx > 0
    assert state[3] > 0.0  # vx > 0
    assert step_infos["agent_0"]["fuel_cost_proxy"] > 0.0

    env.close()



