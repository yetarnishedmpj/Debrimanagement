"""
Custom multi-agent orbital debris removal environment for Ray RLlib.

The environment models a fleet of autonomous satellites operating in a
relative coordinate frame centered on the deployment orbit. Each agent receives
only its own local target geometry and fuel state, which makes the environment
compatible with decentralized policies such as independent PPO.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - allows local unit tests without Ray.
    class MultiAgentEnv(gym.Env):
        """Fallback shim when Ray is unavailable in the active environment."""


@dataclass(frozen=True)
class OrbitalEnvConfig:
    """Environment configuration tuned for a stable prototype."""

    num_agents: int = 3
    num_targets: int = 5
    max_targets: int = 5
    max_episode_steps: int = 300
    world_range_km: float = 250.0
    capture_radius_km: float = 12.0
    translation_step_km: float = 18.0
    agent_spawn_radius_km: float = 15.0
    target_position_noise_km: float = 4.0
    initial_fuel: float = 1.0
    fuel_capacity_delta_v: float = 300.0
    k1_phasing_cost: float = 0.025
    k2_plane_change_cost: float = 0.085
    capture_reward: float = 15.0
    cooperative_reward_share: float = 4.0
    progress_reward_scale: float = 2.0
    fuel_penalty_scale: float = 0.4
    idle_penalty: float = 0.03
    invalid_action_penalty: float = 0.2
    random_seed: Optional[int] = None
    target_positions: Optional[List[List[float]]] = None


def _coerce_config(env_config: Optional[Mapping[str, Any]]) -> OrbitalEnvConfig:
    """Merge user overrides into the typed config with basic validation."""

    defaults = OrbitalEnvConfig()
    values = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    if env_config:
        values.update(dict(env_config))

    values["num_agents"] = max(1, int(values["num_agents"]))
    values["num_targets"] = max(1, int(values["num_targets"]))
    values["max_targets"] = max(int(values["max_targets"]), int(values["num_targets"]))
    values["max_episode_steps"] = max(1, int(values["max_episode_steps"]))
    values["world_range_km"] = max(1.0, float(values["world_range_km"]))
    values["capture_radius_km"] = max(0.1, float(values["capture_radius_km"]))
    values["translation_step_km"] = max(0.0, float(values["translation_step_km"]))
    values["agent_spawn_radius_km"] = max(0.0, float(values["agent_spawn_radius_km"]))
    values["target_position_noise_km"] = max(0.0, float(values["target_position_noise_km"]))
    values["initial_fuel"] = float(np.clip(values["initial_fuel"], 0.05, 1.0))
    values["fuel_capacity_delta_v"] = max(1.0, float(values["fuel_capacity_delta_v"]))
    values["k1_phasing_cost"] = max(0.0, float(values["k1_phasing_cost"]))
    values["k2_plane_change_cost"] = max(0.0, float(values["k2_plane_change_cost"]))

    target_positions = values.get("target_positions")
    if target_positions is not None:
        sanitized_positions: List[List[float]] = []
        for position in target_positions[: values["max_targets"]]:
            vector = np.asarray(position, dtype=np.float32).reshape(-1)
            if vector.size != 3:
                raise ValueError(
                    "Each configured target position must contain exactly 3 values."
                )
            sanitized_positions.append(vector.astype(float).tolist())
        values["target_positions"] = sanitized_positions

    return OrbitalEnvConfig(**values)


class OrbitalDebrisRemovalEnv(MultiAgentEnv):
    """
    Multi-agent debris-removal environment with decentralized observations.

    Observation design:
    - shape = (max_targets, 5)
      Per target: [fuel, rel_x, rel_y, rel_z, distance]

    Action design:
    - MultiDiscrete([max_targets, 3, 3, 3])
      * target slot to prioritize this control step
      * x/y/z burns encoded as {0,1,2} -> {-1,0,1}
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        self.config = _coerce_config(env_config)
        self.max_targets = self.config.max_targets
        self.possible_agents = [f"agent_{index}" for index in range(self.config.num_agents)]
        self.agents = list(self.possible_agents)
        self._rng = np.random.default_rng(self.config.random_seed)
        self._step_count = 0
        self._capture_count = 0

        self._agent_positions = np.zeros((self.config.num_agents, 3), dtype=np.float32)
        self._agent_fuel = np.full(
            (self.config.num_agents,), self.config.initial_fuel, dtype=np.float32
        )
        self._target_positions = np.zeros((self.max_targets, 3), dtype=np.float32)
        self._target_active = np.zeros((self.max_targets,), dtype=bool)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_targets, 5),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.asarray([self.max_targets, 3, 3, 3], dtype=np.int64)
        )
        self._episode_history: Dict[str, List[np.ndarray]] = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset fleet state, debris field, and episode bookkeeping."""

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.config.random_seed is not None:
            self._rng = np.random.default_rng(self.config.random_seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._capture_count = 0
        self._agent_fuel.fill(self.config.initial_fuel)

        resolved_options: Dict[str, Any] = dict(options or {})
        self._agent_positions = self._resolve_agent_positions(resolved_options).astype(
            np.float32
        )
        self._target_positions, self._target_active = self._resolve_target_positions(
            resolved_options
        )

        self._episode_history = {
            "agent_positions": [self._agent_positions.copy()],
            "agent_fuel": [self._agent_fuel.copy()],
            "target_positions": [self._target_positions.copy()],
            "target_active": [self._target_active.astype(np.int8).copy()],
            "step_rewards": [],
            "delta_v": [],
            "selected_targets": [],
            "capture_matrix": [],
            "action_commands": [],
            "closing_delta": [],
        }

        observations = {
            agent_id: self._build_observation(agent_index)
            for agent_index, agent_id in enumerate(self.agents)
        }
        infos = {
            agent_id: self._build_info(
                agent_index=agent_index,
                selected_target_index=None,
                delta_v=0.0,
                capture_event=False,
            )
            for agent_index, agent_id in enumerate(self.agents)
        }
        return observations, infos

    def step(
        self, action_dict: MutableMapping[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Advance the simulation by one fleet control interval."""

        if not self.agents:
            raise RuntimeError(
                "step() was called after the episode finished. Call reset() first."
            )

        rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        infos: Dict[str, Dict[str, Any]] = {}
        capture_events: List[Tuple[str, int]] = []
        step_delta_v = np.zeros((self.config.num_agents,), dtype=np.float32)
        step_selected_targets = np.full((self.config.num_agents,), -1, dtype=np.int16)
        step_capture_flags = np.zeros((self.config.num_agents,), dtype=np.int8)
        step_action_commands = np.tile(
            np.asarray([0, 1, 1, 1], dtype=np.int64),
            (self.config.num_agents, 1),
        )
        step_closing_delta = np.zeros((self.config.num_agents,), dtype=np.float32)

        active_target_indices = np.where(self._target_active)[0].tolist()
        if not active_target_indices:
            terminateds = {agent_id: True for agent_id in self.possible_agents}
            truncateds = {agent_id: False for agent_id in self.possible_agents}
            terminateds["__all__"] = True
            truncateds["__all__"] = False
            self.agents = []
            return {}, rewards, terminateds, truncateds, infos

        for agent_index, agent_id in enumerate(self.possible_agents):
            selected_target_index: Optional[int] = None
            delta_v = 0.0
            capture_event = False

            if self._agent_fuel[agent_index] <= 0.0:
                infos[agent_id] = self._build_info(
                    agent_index=agent_index,
                    selected_target_index=None,
                    delta_v=0.0,
                    capture_event=False,
                )
                continue

            parsed_action, invalid_penalty = self._parse_action(action_dict.get(agent_id))
            rewards[agent_id] -= invalid_penalty
            step_action_commands[agent_index] = parsed_action
            target_index = self._resolve_target_index(int(parsed_action[0]))
            selected_target_index = target_index
            step_selected_targets[agent_index] = target_index

            current_position = self._agent_positions[agent_index].copy()
            target_position = self._target_positions[target_index].copy()
            previous_distance = float(np.linalg.norm(target_position - current_position))

            burn_direction = parsed_action[1:].astype(np.float32) - 1.0
            burn_norm = float(np.linalg.norm(burn_direction))
            if burn_norm > 0.0:
                burn_vector = (
                    burn_direction / burn_norm * self.config.translation_step_km
                ).astype(np.float32)
            else:
                burn_vector = np.zeros(3, dtype=np.float32)
                rewards[agent_id] -= self.config.idle_penalty

            delta_v = self._lambert_proxy_delta_v(
                agent_position=current_position,
                target_position=target_position,
                burn_vector=burn_vector,
            )

            delta_fuel = delta_v / self.config.fuel_capacity_delta_v
            available_fuel = float(self._agent_fuel[agent_index])
            if delta_fuel > available_fuel and delta_fuel > 0.0:
                scale = available_fuel / delta_fuel
                burn_vector *= scale
                delta_fuel = available_fuel
                delta_v = delta_fuel * self.config.fuel_capacity_delta_v

            next_position = np.clip(
                current_position + burn_vector,
                -self.config.world_range_km,
                self.config.world_range_km,
            )
            self._agent_positions[agent_index] = next_position.astype(np.float32)
            self._agent_fuel[agent_index] = max(0.0, available_fuel - delta_fuel)

            new_distance = float(np.linalg.norm(target_position - next_position))
            step_delta_v[agent_index] = float(delta_v)
            step_closing_delta[agent_index] = float(previous_distance - new_distance)
            rewards[agent_id] += self.config.progress_reward_scale * (
                previous_distance - new_distance
            ) / self.config.world_range_km
            rewards[agent_id] -= self.config.fuel_penalty_scale * delta_fuel

            if self._target_active[target_index] and new_distance <= self.config.capture_radius_km:
                self._target_active[target_index] = False
                self._capture_count += 1
                capture_event = True
                capture_events.append((agent_id, target_index))
                step_capture_flags[agent_index] = 1
                rewards[agent_id] += self.config.capture_reward

            infos[agent_id] = self._build_info(
                agent_index=agent_index,
                selected_target_index=selected_target_index,
                delta_v=delta_v,
                capture_event=capture_event,
            )

        # Global cooperative reward shaping: every teammate gets a partial bonus
        # whenever any agent secures a debris capture.
        for capturing_agent_id, target_index in capture_events:
            for teammate_id in self.possible_agents:
                if teammate_id != capturing_agent_id:
                    rewards[teammate_id] += self.config.cooperative_reward_share
            infos[capturing_agent_id]["captured_target_index"] = target_index

        self._step_count += 1
        self._episode_history["agent_positions"].append(self._agent_positions.copy())
        self._episode_history["agent_fuel"].append(self._agent_fuel.copy())
        self._episode_history["target_positions"].append(self._target_positions.copy())
        self._episode_history["target_active"].append(
            self._target_active.astype(np.int8).copy()
        )
        self._episode_history["step_rewards"].append(
            np.asarray(
                [rewards[agent_id] for agent_id in self.possible_agents],
                dtype=np.float32,
            )
        )
        self._episode_history["delta_v"].append(step_delta_v.copy())
        self._episode_history["selected_targets"].append(step_selected_targets.copy())
        self._episode_history["capture_matrix"].append(step_capture_flags.copy())
        self._episode_history["action_commands"].append(step_action_commands.copy())
        self._episode_history["closing_delta"].append(step_closing_delta.copy())

        all_targets_captured = not bool(np.any(self._target_active))
        all_fuel_depleted = bool(np.all(self._agent_fuel <= 0.0))
        time_limit_reached = self._step_count >= self.config.max_episode_steps

        terminateds: Dict[str, bool] = {}
        truncateds: Dict[str, bool] = {}
        next_observations: Dict[str, np.ndarray] = {}
        surviving_agents: List[str] = []

        for agent_index, agent_id in enumerate(self.possible_agents):
            agent_terminated = all_targets_captured or (
                self._agent_fuel[agent_index] <= 0.0 and not time_limit_reached
            )
            agent_truncated = time_limit_reached and not all_targets_captured
            terminateds[agent_id] = bool(agent_terminated)
            truncateds[agent_id] = bool(agent_truncated)
            if not agent_terminated and not agent_truncated:
                surviving_agents.append(agent_id)
                next_observations[agent_id] = self._build_observation(agent_index)

        terminateds["__all__"] = all_targets_captured or all_fuel_depleted
        truncateds["__all__"] = time_limit_reached and not terminateds["__all__"]
        self.agents = (
            surviving_agents
            if not (terminateds["__all__"] or truncateds["__all__"])
            else []
        )

        # RLlib expects per-agent infos to be present only for agents that also
        # appear in the next observation dict (plus an optional `__common__` key).
        filtered_infos = {
            agent_id: infos[agent_id]
            for agent_id in next_observations
            if agent_id in infos
        }
        filtered_infos["__common__"] = {
            "step_count": self._step_count,
            "captures_completed": self._capture_count,
            "active_targets_remaining": int(np.count_nonzero(self._target_active)),
        }

        return next_observations, rewards, terminateds, truncateds, filtered_infos

    def get_episode_history(self) -> Dict[str, np.ndarray]:
        """Return a copy of the stored episode trajectory."""

        if not self._episode_history:
            return {}
        return {
            key: np.stack(value, axis=0).copy()
            for key, value in self._episode_history.items()
        }

    def _resolve_agent_positions(self, options: Mapping[str, Any]) -> np.ndarray:
        configured_positions = options.get("agent_positions")
        if configured_positions is not None:
            positions = np.asarray(configured_positions, dtype=np.float32)
            if positions.shape != (self.config.num_agents, 3):
                raise ValueError(
                    "agent_positions must have shape "
                    f"({self.config.num_agents}, 3), got {positions.shape}."
                )
            return np.clip(
                positions,
                -self.config.world_range_km,
                self.config.world_range_km,
            )

        positions = self._rng.uniform(
            low=-self.config.agent_spawn_radius_km,
            high=self.config.agent_spawn_radius_km,
            size=(self.config.num_agents, 3),
        )
        positions[:, 2] *= 0.35
        return positions.astype(np.float32)

    def _resolve_target_positions(
        self, options: Mapping[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.zeros((self.max_targets, 3), dtype=np.float32)
        active = np.zeros((self.max_targets,), dtype=bool)

        source_positions = options.get("target_positions")
        if source_positions is None:
            source_positions = self.config.target_positions

        if source_positions is not None:
            raw_positions = np.asarray(source_positions, dtype=np.float32)
            if raw_positions.ndim != 2 or raw_positions.shape[1] != 3:
                raise ValueError(
                    "target_positions must be a sequence of three-element vectors."
                )
            usable = min(raw_positions.shape[0], self.max_targets)
            positions[:usable] = raw_positions[:usable]
            if (
                options.get("target_positions") is None
                and self.config.target_position_noise_km > 0.0
            ):
                noise = self._rng.normal(
                    loc=0.0,
                    scale=self.config.target_position_noise_km,
                    size=(usable, 3),
                ).astype(np.float32)
                noise[:, 2] *= 0.4
                positions[:usable] += noise
            active[:usable] = True
        else:
            positions[: self.config.num_targets] = self._sample_clustered_targets(
                self.config.num_targets
            )
            active[: self.config.num_targets] = True

        positions = np.clip(
            positions,
            -self.config.world_range_km,
            self.config.world_range_km,
        )
        return positions.astype(np.float32), active

    def _sample_clustered_targets(self, count: int) -> np.ndarray:
        """Generate clustered debris fragments reminiscent of breakup clouds."""

        cluster_centers = np.asarray(
            [
                [90.0, -50.0, 18.0],
                [-120.0, 70.0, -25.0],
                [55.0, 135.0, 12.0],
            ],
            dtype=np.float32,
        )
        positions = np.zeros((count, 3), dtype=np.float32)
        for index in range(count):
            center = cluster_centers[index % len(cluster_centers)]
            jitter = self._rng.normal(loc=0.0, scale=18.0, size=3).astype(np.float32)
            jitter[2] *= 0.5
            positions[index] = center + jitter
        return positions

    def _build_observation(self, agent_index: int) -> np.ndarray:
        return self._build_local_target_matrix(agent_index)

    def _build_local_target_matrix(self, agent_index: int) -> np.ndarray:
        matrix = np.zeros((self.max_targets, 5), dtype=np.float32)
        agent_position = self._agent_positions[agent_index]
        fuel_norm = self._normalize_fuel(self._agent_fuel[agent_index])

        for target_index in range(self.max_targets):
            matrix[target_index, 0] = fuel_norm
            if not self._target_active[target_index]:
                continue

            relative_vector = self._target_positions[target_index] - agent_position
            distance = float(np.linalg.norm(relative_vector))
            matrix[target_index, 1:4] = self._clip(
                relative_vector / self.config.world_range_km
            )
            matrix[target_index, 4] = float(
                np.clip(distance / self.config.world_range_km, -1.0, 1.0)
            )
        return self._clip(matrix)

    def _build_info(
        self,
        *,
        agent_index: int,
        selected_target_index: Optional[int],
        delta_v: float,
        capture_event: bool,
    ) -> Dict[str, Any]:
        return {
            "agent_index": agent_index,
            "step_count": self._step_count,
            "selected_target_index": selected_target_index,
            "delta_v": float(delta_v),
            "fuel_remaining": float(self._agent_fuel[agent_index]),
            "active_targets_remaining": int(np.count_nonzero(self._target_active)),
            "capture_event": bool(capture_event),
        }

    def _parse_action(self, raw_action: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        if raw_action is None:
            default_action = np.asarray([0, 1, 1, 1], dtype=np.int64)
            return default_action, self.config.invalid_action_penalty

        try:
            action = np.asarray(raw_action, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError):
            default_action = np.asarray([0, 1, 1, 1], dtype=np.int64)
            return default_action, self.config.invalid_action_penalty

        if action.size != 4:
            default_action = np.asarray([0, 1, 1, 1], dtype=np.int64)
            return default_action, self.config.invalid_action_penalty

        clipped = action.copy()
        clipped[0] = int(np.clip(clipped[0], 0, self.max_targets - 1))
        clipped[1:] = np.clip(clipped[1:], 0, 2)
        penalty = 0.0 if np.array_equal(action, clipped) else self.config.invalid_action_penalty
        return clipped.astype(np.int64), penalty

    def _resolve_target_index(self, requested_index: int) -> int:
        if self._target_active[requested_index]:
            return requested_index

        active_indices = np.where(self._target_active)[0]
        if active_indices.size == 0:
            return int(requested_index)

        cluster_positions = self._target_positions[active_indices]
        cluster_norms = np.linalg.norm(cluster_positions, axis=1)
        return int(active_indices[np.argmin(cluster_norms)])

    def _lambert_proxy_delta_v(
        self,
        *,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        burn_vector: np.ndarray,
    ) -> float:
        """
        Estimate Delta-V using a lightweight Lambert proxy.

        `k1` penalizes in-plane phasing maneuvers and `k2` penalizes out-of-plane
        plane changes. The plane-change term is intentionally more expensive.
        """

        relative_vector = target_position - agent_position
        in_plane_distance = float(np.linalg.norm(relative_vector[:2]))
        cross_track_distance = float(abs(relative_vector[2]))
        burn_magnitude = float(np.linalg.norm(burn_vector))

        alignment_penalty = 0.0
        relative_norm = float(np.linalg.norm(relative_vector))
        if burn_magnitude > 0.0 and relative_norm > 1e-6:
            cosine_alignment = float(
                np.dot(burn_vector, relative_vector) / (burn_magnitude * relative_norm)
            )
            alignment_penalty = 0.5 * (1.0 - float(np.clip(cosine_alignment, -1.0, 1.0)))

        delta_v = (
            self.config.k1_phasing_cost * in_plane_distance
            + self.config.k2_plane_change_cost * cross_track_distance
            + 0.1 * burn_magnitude
            + alignment_penalty
        )
        return max(0.0, float(delta_v))

    @staticmethod
    def _normalize_fuel(fuel_fraction: float) -> float:
        return float(np.clip((2.0 * fuel_fraction) - 1.0, -1.0, 1.0))

    @staticmethod
    def _clip(values: np.ndarray) -> np.ndarray:
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    def render(self) -> Dict[str, Any]:
        """Return a debug-friendly snapshot instead of drawing inline graphics."""

        return {
            "step_count": self._step_count,
            "agent_positions": deepcopy(self._agent_positions.tolist()),
            "target_positions": deepcopy(self._target_positions.tolist()),
            "target_active": deepcopy(self._target_active.tolist()),
        }

    def close(self) -> None:
        self.agents = []
