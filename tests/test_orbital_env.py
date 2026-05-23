import numpy as np

from orbital_env import OrbitalDebrisRemovalEnv


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
        assert agent_obs.shape == (4, 5)
        assert np.all(agent_obs <= 1.0)
        assert np.all(agent_obs >= -1.0)

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
