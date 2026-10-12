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
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

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
    """Environment configuration tuned for Clohessy-Wiltshire orbital dynamics."""

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
    collision_radius_km: float = 0.0
    inter_agent_collision_penalty: float = 0.0
    target_resolution_policy: str = "no_op"
    random_seed: Optional[int] = None
    target_positions: Optional[List[List[float]]] = None

    # Clohessy-Wiltshire (CW / Hill) Astrodynamics Parameters
    dynamics_model: str = "clohessy_wiltshire"
    reference_altitude_km: float = 500.0
    earth_radius_km: float = 6378.137
    gravitational_parameter_mu: float = 398600.4418
    time_step_s: float = 10.0
    max_impulse_delta_v: float = 0.005
    control_mode: str = "impulsive"
    target_drift_mode: str = "periodic_ellipse"
    target_velocity_noise_km_s: float = 0.0
    obs_includes_velocity: bool = False
    velocity_range_km_s: float = 0.1
    target_velocities: Optional[List[List[float]]] = None
    agent_velocities: Optional[List[List[float]]] = None


def compute_cw_stm(mean_motion_n: float, delta_t_s: float) -> np.ndarray:
    """
    Compute the 6x6 Clohessy-Wiltshire (Hill) State Transition Matrix (STM).

    Coordinate frame: Hill (LVLH) frame centered on a circular reference orbit:
    - x (Radial): Points radially outward from Earth center.
    - y (In-Track / Along-Track): Points in the direction of orbital velocity.
    - z (Cross-Track / Out-of-Plane): Normal to the orbital plane along angular momentum.

    Dynamics:
    x''(t) - 2*n*y'(t) - 3*n^2*x(t) = a_x(t)
    y''(t) + 2*n*x'(t) = a_y(t)
    z''(t) + n^2*z(t) = a_z(t)

    State vector: s = [rx, ry, rz, vx, vy, vz]^T.
    Returns: 6x6 numpy array (float32).
    """
    n = float(mean_motion_n)
    tau = float(delta_t_s)
    ntau = n * tau
    s = float(np.sin(ntau))
    c = float(np.cos(ntau))

    stm = np.zeros((6, 6), dtype=np.float32)

    # Position from position (Phi_rr)
    stm[0, 0] = float(4.0 - 3.0 * c)
    stm[1, 0] = float(6.0 * (s - ntau))
    stm[1, 1] = 1.0
    stm[2, 2] = c

    # Position from velocity (Phi_rv)
    stm[0, 3] = float(s / n)
    stm[0, 4] = float((2.0 / n) * (1.0 - c))
    stm[1, 3] = float(-(2.0 / n) * (1.0 - c))
    stm[1, 4] = float((4.0 * s - 3.0 * ntau) / n)
    stm[2, 5] = float(s / n)

    # Velocity from position (Phi_vr)
    stm[3, 0] = float(3.0 * n * s)
    stm[4, 0] = float(6.0 * n * (c - 1.0))
    stm[5, 2] = float(-n * s)

    # Velocity from velocity (Phi_vv)
    stm[3, 3] = c
    stm[3, 4] = float(2.0 * s)
    stm[4, 3] = float(-2.0 * s)
    stm[4, 4] = float(4.0 * c - 3.0)
    stm[5, 5] = c

    return stm


def compute_cw_control_matrix(mean_motion_n: float, delta_t_s: float) -> np.ndarray:
    """
    Compute the 6x3 input matrix Gamma(tau) for continuous constant acceleration.
    Gamma(tau) = integral_0^tau Phi(tau - sigma) * B dsigma, where B = [0; I_3].
    """
    n = float(mean_motion_n)
    tau = float(delta_t_s)
    ntau = n * tau
    s = float(np.sin(ntau))
    c = float(np.cos(ntau))
    n2 = float(n * n)

    gamma = np.zeros((6, 3), dtype=np.float32)

    # Position response to constant acceleration (Gamma_r)
    gamma[0, 0] = float((1.0 - c) / n2)
    gamma[0, 1] = float(2.0 * (ntau - s) / n2)
    gamma[1, 0] = float(-2.0 * (ntau - s) / n2)
    gamma[1, 1] = float((4.0 * (1.0 - c) - 1.5 * ntau * ntau) / n2)
    gamma[2, 2] = float((1.0 - c) / n2)

    # Velocity response to constant acceleration (Gamma_v) = Phi_rv
    gamma[3, 0] = float(s / n)
    gamma[3, 1] = float((2.0 / n) * (1.0 - c))
    gamma[4, 0] = float(-(2.0 / n) * (1.0 - c))
    gamma[4, 1] = float((4.0 * s - 3.0 * ntau) / n)
    gamma[5, 2] = float(s / n)

    return gamma


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
    values["fuel_capacity_delta_v"] = max(1e-4, float(values["fuel_capacity_delta_v"]))
    values["k1_phasing_cost"] = max(0.0, float(values["k1_phasing_cost"]))
    values["k2_plane_change_cost"] = max(0.0, float(values["k2_plane_change_cost"]))
    values["collision_radius_km"] = max(0.0, float(values.get("collision_radius_km", 0.0)))
    values["inter_agent_collision_penalty"] = max(
        0.0, float(values.get("inter_agent_collision_penalty", 0.0))
    )
    target_policy = str(values.get("target_resolution_policy", "no_op")).lower().strip()
    if target_policy not in ("no_op", "nearest_agent"):
        raise ValueError(
            f"Unsupported target_resolution_policy: '{target_policy}'. "
            "Must be 'no_op' or 'nearest_agent'."
        )
    values["target_resolution_policy"] = target_policy

    dyn_model = str(values.get("dynamics_model", "clohessy_wiltshire")).lower().strip()
    if dyn_model in ("clohessy_wiltshire", "cw", "hill"):
        values["dynamics_model"] = "clohessy_wiltshire"
    elif dyn_model in ("kinematic_translation", "translation"):
        values["dynamics_model"] = "kinematic_translation"
    else:
        raise ValueError(
            f"Unsupported dynamics_model: '{dyn_model}'. Must be 'clohessy_wiltshire' or 'kinematic_translation'."
        )

    ctrl_mode = str(values.get("control_mode", "impulsive")).lower().strip()
    if ctrl_mode not in ("impulsive", "continuous_acceleration"):
        raise ValueError(
            f"Unsupported control_mode: '{ctrl_mode}'. Must be 'impulsive' or 'continuous_acceleration'."
        )
    values["control_mode"] = ctrl_mode

    drift_mode = str(values.get("target_drift_mode", "periodic_ellipse")).lower().strip()
    if drift_mode not in ("periodic_ellipse", "stationary_initial", "free_drift"):
        raise ValueError(
            f"Unsupported target_drift_mode: '{drift_mode}'. Must be 'periodic_ellipse', 'stationary_initial', or 'free_drift'."
        )
    values["target_drift_mode"] = drift_mode

    values["reference_altitude_km"] = max(100.0, float(values["reference_altitude_km"]))
    values["earth_radius_km"] = max(1000.0, float(values["earth_radius_km"]))
    values["gravitational_parameter_mu"] = max(1000.0, float(values["gravitational_parameter_mu"]))
    values["time_step_s"] = max(0.1, float(values["time_step_s"]))
    values["max_impulse_delta_v"] = max(0.0, float(values["max_impulse_delta_v"]))
    values["target_velocity_noise_km_s"] = max(0.0, float(values["target_velocity_noise_km_s"]))
    values["velocity_range_km_s"] = max(1e-4, float(values["velocity_range_km_s"]))
    values["obs_includes_velocity"] = bool(values.get("obs_includes_velocity", False))

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

    target_velocities = values.get("target_velocities")
    if target_velocities is not None:
        sanitized_vels: List[List[float]] = []
        for velocity in target_velocities[: values["max_targets"]]:
            vector = np.asarray(velocity, dtype=np.float32).reshape(-1)
            if vector.size != 3:
                raise ValueError("Each configured target velocity must contain exactly 3 values.")
            sanitized_vels.append(vector.astype(float).tolist())
        values["target_velocities"] = sanitized_vels

    agent_velocities = values.get("agent_velocities")
    if agent_velocities is not None:
        sanitized_agent_vels: List[List[float]] = []
        for velocity in agent_velocities[: values["num_agents"]]:
            vector = np.asarray(velocity, dtype=np.float32).reshape(-1)
            if vector.size != 3:
                raise ValueError("Each configured agent velocity must contain exactly 3 values.")
            sanitized_agent_vels.append(vector.astype(float).tolist())
        values["agent_velocities"] = sanitized_agent_vels

    return OrbitalEnvConfig(**values)


class OrbitalDebrisRemovalEnv(MultiAgentEnv):
    """
    Multi-agent orbital debris rendezvous environment with decentralized observations.

    Dynamical Model (Clohessy-Wiltshire / Hill Relative Motion):
    Servicing satellites and debris targets operate in the Local-Vertical/Local-Horizontal (LVLH)
    relative reference frame centered on a circular reference orbit:
    - x (Radial): Points radially outward from Earth center through the reference origin.
    - y (In-Track / Along-Track): Points along the circular reference orbit velocity vector.
    - z (Cross-Track / Out-of-Plane): Normal to the orbital plane along the angular momentum vector.

    6-DOF Dynamical State Vector:
    s_t = [r_x, r_y, r_z, v_x, v_y, v_z]^T
    where [r_x, r_y, r_z] are relative positions (km) and [v_x, v_y, v_z] are relative velocities (km/s).

    Relative Motion Equations (Hill / Clohessy-Wiltshire):
    d^2 x / dt^2 - 2*n*(dy/dt) - 3*n^2*x = u_x
    d^2 y / dt^2 + 2*n*(dx/dt)           = u_y
    d^2 z / dt^2 + n^2*z                 = u_z
    where n = sqrt(mu / R_0^3) is the reference orbit mean motion.

    The state transitions are propagated symplectically using the closed-form 6x6 State Transition
    Matrix Phi(Delta t). Debris targets evolve dynamically under unforced motion throughout each episode.
    Control inputs represent bounded Delta-V impulses (or continuous accelerations) with direct physical
    propellant accounting: Delta_fuel = ||Delta_v|| / v_capacity.

    Operational Scope & Astrodynamic Distinction:
    In space mission design, Active Debris Removal (ADR) spans three distinct phases:
    1. Far-Range Rendezvous: Orbital phasing, plane matching, and relative navigation
       to reach a proximity inspection hold point (e.g., within 12 km).
    2. Close-Proximity Physical Capture: Sub-meter station-keeping, relative velocity
       nulling (Δv_rel -> 0), and mechanical latching/grappling (robotic arm, harpoon, net).
    3. Complete Debris Removal (Disposal): Post-capture retrograde de-orbit burn to
       atmospheric reentry or orbit raising to a graveyard disposal orbit.

    THIS SIMULATION SPECIFICALLY MODELS: PHASE 1 (SUCCESSFUL RENDEZVOUS ARRIVAL).
    - An agent arriving within `capture_radius_km` (default: 12.0 km) completes a
      "rendezvous arrival" (coarse proximity rendezvous / standoff arrival) with the target.
    - Physical mechanical capture (Phase 2) is NOT simulated (no contact or docking mechanics).
    - Complete debris removal (Phase 3) is NOT simulated (no de-orbit or reentry burns).
    - Upon successful rendezvous arrival, the debris target is marked inactive (retiring
      the rendezvous objective for that target), and the agent remains active in orbit with
      residual propellant to pursue subsequent rendezvous targets.

    Observation design:
    - Default shape = (max_targets, 6)
      Per target: [availability_mask, rel_x, rel_y, rel_z, distance, agent_fuel]
      * availability_mask (m_j): 1.0 if target slot is active, 0.0 if inactive
      * rel_x, rel_y, rel_z: relative Cartesian coordinates scaled by world_range_km
      * distance: Euclidean distance scaled by world_range_km
      * agent_fuel: normalized agent propellant fraction in [-1.0, 1.0]
    - When `obs_includes_velocity=True`: shape = (max_targets, 10), adding
      [rel_vx, rel_vy, rel_vz, rel_v_mag] normalized by `velocity_range_km_s`.

    Action design:
    - MultiDiscrete([max_targets, 3, 3, 3])
      * target slot to prioritize this control step
      * x/y/z thrust direction encoded as {0,1,2} -> {-1,0,1}
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

        # Physical orbital reference orbit parameters (Hill frame)
        self.reference_radius_km = float(
            self.config.earth_radius_km + self.config.reference_altitude_km
        )
        self.mean_motion_n = float(
            np.sqrt(self.config.gravitational_parameter_mu / (self.reference_radius_km**3))
        )
        self.orbital_period_s = float(2.0 * np.pi / self.mean_motion_n)
        self._cw_stm = compute_cw_stm(self.mean_motion_n, self.config.time_step_s)
        self._cw_gamma = compute_cw_control_matrix(self.mean_motion_n, self.config.time_step_s)

        # 6-DOF Dynamical state containers: positions [km] and relative velocities [km/s]
        self._agent_positions = np.zeros((self.config.num_agents, 3), dtype=np.float32)
        self._agent_velocities = np.zeros((self.config.num_agents, 3), dtype=np.float32)
        self._agent_fuel = np.full(
            (self.config.num_agents,), self.config.initial_fuel, dtype=np.float32
        )
        self._target_positions = np.zeros((self.max_targets, 3), dtype=np.float32)
        self._target_velocities = np.zeros((self.max_targets, 3), dtype=np.float32)
        self._target_active = np.zeros((self.max_targets,), dtype=bool)

        obs_feature_dim = 10 if self.config.obs_includes_velocity else 6
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_targets, obs_feature_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.asarray([self.max_targets, 3, 3, 3], dtype=np.int64)
        )
        self._episode_history: Dict[str, List[np.ndarray]] = {}

    def get_agent_state(self, agent_index_or_id: Union[int, str]) -> np.ndarray:
        """
        Return the 6-DOF dynamical state of an agent in the Hill / LVLH reference frame.
        s_t = [r_x, r_y, r_z, v_x, v_y, v_z]^T
        Units: positions in km, relative velocities in km/s.
        """
        if isinstance(agent_index_or_id, str):
            agent_index = self.possible_agents.index(agent_index_or_id)
        else:
            agent_index = int(agent_index_or_id)
        return np.concatenate(
            [self._agent_positions[agent_index], self._agent_velocities[agent_index]],
            axis=0,
        ).astype(np.float32)

    def get_target_state(self, target_index: int) -> np.ndarray:
        """
        Return the 6-DOF dynamical state of a debris target in the Hill / LVLH reference frame.
        s_t = [r_x, r_y, r_z, v_x, v_y, v_z]^T
        Units: positions in km, relative velocities in km/s.
        """
        return np.concatenate(
            [self._target_positions[target_index], self._target_velocities[target_index]],
            axis=0,
        ).astype(np.float32)

    def get_fleet_states(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary containing 6-DOF dynamical states for all agents and targets.
        """
        states: Dict[str, np.ndarray] = {
            agent_id: self.get_agent_state(idx)
            for idx, agent_id in enumerate(self.possible_agents)
        }
        states["targets"] = np.stack(
            [self.get_target_state(idx) for idx in range(self.max_targets)],
            axis=0,
        )
        return states

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
        self._target_velocities = self._resolve_target_velocities(
            resolved_options, self._target_positions, self._target_active
        )
        self._agent_velocities = self._resolve_agent_velocities(
            resolved_options, self._agent_positions
        )

        self._episode_history = {
            "agent_positions": [self._agent_positions.copy()],
            "agent_velocities": [self._agent_velocities.copy()],
            "agent_states": [
                np.stack([self.get_agent_state(i) for i in range(self.config.num_agents)], axis=0)
            ],
            "agent_fuel": [self._agent_fuel.copy()],
            "target_positions": [self._target_positions.copy()],
            "target_velocities": [self._target_velocities.copy()],
            "target_states": [
                np.stack([self.get_target_state(i) for i in range(self.max_targets)], axis=0)
            ],
            "target_active": [self._target_active.astype(np.int8).copy()],
            "step_rewards": [],
            "fuel_cost_proxy": [],
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
        self,
        action_dict: MutableMapping[str, np.ndarray],
        *,
        agent_order: Optional[Sequence[str]] = None,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Advance the simulation by one fleet control interval using simultaneous execution.

        Execution sequence:
        1. Snapshot the current state (positions, velocities/motion, fuel, and target status).
        2. Collect all agent actions (all actions evaluate against the identical pre-step state).
        3. Compute all proposed transitions (update motion and fuel without modifying shared state).
        4. Resolve interactions jointly (target conflicts, captures, and collisions).
           Tie-breaking rules for target conflicts:
           - Proximity priority: The candidate agent closest to the target gets the capture.
           - Deterministic tie-break: If distances are tied within 1e-5 km, the lower agent index wins.
        5. Commit the next state and rewards (atomic update of environment state).
        """

        if not self.agents:
            raise RuntimeError(
                "step() was called after the episode finished. Call reset() first."
            )

        rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        infos: Dict[str, Dict[str, Any]] = {}

        active_target_indices = np.where(self._target_active)[0].tolist()
        if not active_target_indices:
            terminateds = {agent_id: True for agent_id in self.possible_agents}
            truncateds = {agent_id: False for agent_id in self.possible_agents}
            terminateds["__all__"] = True
            truncateds["__all__"] = False
            self.agents = []
            return {}, rewards, terminateds, truncateds, infos

        # Determine agent processing order (defaults to canonical possible_agents)
        processing_order = (
            list(agent_order)
            if agent_order is not None
            else list(self.possible_agents)
        )

        # ------------------------------------------------------------------
        # 1. Snapshot the current state
        # ------------------------------------------------------------------
        snapshot_agent_positions = self._agent_positions.copy()
        snapshot_agent_velocities = self._agent_velocities.copy()
        snapshot_agent_fuel = self._agent_fuel.copy()
        snapshot_target_positions = self._target_positions.copy()
        snapshot_target_velocities = self._target_velocities.copy()
        snapshot_target_active = self._target_active.copy()

        # Temporary proposed state containers (uncommitted)
        proposed_positions = snapshot_agent_positions.copy()
        proposed_velocities = snapshot_agent_velocities.copy()
        proposed_fuel = snapshot_agent_fuel.copy()
        proposed_delta_v = np.zeros((self.config.num_agents,), dtype=np.float32)
        proposed_selected_targets = np.full((self.config.num_agents,), -1, dtype=np.int16)
        proposed_closing_delta = np.zeros((self.config.num_agents,), dtype=np.float32)
        proposed_action_commands = np.tile(
            np.asarray([0, 1, 1, 1], dtype=np.int64),
            (self.config.num_agents, 1),
        )
        proposed_distances = np.full((self.config.num_agents,), np.inf, dtype=np.float32)

        # Debris propagation under unforced Clohessy-Wiltshire dynamics
        proposed_target_positions = snapshot_target_positions.copy()
        proposed_target_velocities = snapshot_target_velocities.copy()

        if self.config.dynamics_model == "clohessy_wiltshire":
            for t_idx in range(self.max_targets):
                if snapshot_target_active[t_idx]:
                    s_t = np.concatenate(
                        [snapshot_target_positions[t_idx], snapshot_target_velocities[t_idx]],
                        axis=0,
                    )
                    s_next = self._cw_stm @ s_t
                    proposed_target_positions[t_idx] = np.clip(
                        s_next[:3],
                        -self.config.world_range_km,
                        self.config.world_range_km,
                    ).astype(np.float32)
                    proposed_target_velocities[t_idx] = np.clip(
                        s_next[3:],
                        -self.config.velocity_range_km_s,
                        self.config.velocity_range_km_s,
                    ).astype(np.float32)

        # ------------------------------------------------------------------
        # 2. Collect all agent actions & 3. Compute all proposed transitions
        # ------------------------------------------------------------------
        for agent_id in processing_order:
            agent_index = self.possible_agents.index(agent_id)

            parsed_action, invalid_penalty = self._parse_action(action_dict.get(agent_id))
            rewards[agent_id] -= invalid_penalty
            proposed_action_commands[agent_index] = parsed_action

            current_position = snapshot_agent_positions[agent_index].copy()
            current_velocity = snapshot_agent_velocities[agent_index].copy()
            available_fuel = float(snapshot_agent_fuel[agent_index])

            # Target resolution uses pre-step snapshot and deliberate policy (no_op or nearest_agent)
            target_index = self._resolve_target_index(
                int(parsed_action[0]),
                target_active=snapshot_target_active,
                agent_position=current_position,
            )
            proposed_selected_targets[agent_index] = (
                target_index if target_index is not None else -1
            )

            burn_direction = parsed_action[1:].astype(np.float32) - 1.0
            burn_norm = float(np.linalg.norm(burn_direction))

            if available_fuel <= 0.0:
                # Depleted fuel: ballistic drift without control
                if self.config.dynamics_model == "clohessy_wiltshire":
                    s_agent = np.concatenate([current_position, current_velocity], axis=0)
                    s_next = self._cw_stm @ s_agent
                    next_position = np.clip(
                        s_next[:3], -self.config.world_range_km, self.config.world_range_km
                    ).astype(np.float32)
                    next_velocity = np.clip(
                        s_next[3:],
                        -self.config.velocity_range_km_s,
                        self.config.velocity_range_km_s,
                    ).astype(np.float32)
                else:
                    next_position = current_position
                    next_velocity = current_velocity
                next_fuel = 0.0
                delta_fuel = 0.0
                fuel_cost = 0.0
            else:
                if self.config.dynamics_model == "clohessy_wiltshire":
                    if burn_norm > 0.0 and self.config.translation_step_km > 0.0:
                        unit_dir = (burn_direction / burn_norm).astype(np.float32)
                        cmd_impulse = unit_dir * self.config.max_impulse_delta_v
                    else:
                        cmd_impulse = np.zeros(3, dtype=np.float32)
                        if burn_norm == 0.0:
                            rewards[agent_id] -= self.config.idle_penalty

                    impulse_norm = float(np.linalg.norm(cmd_impulse))
                    delta_fuel = impulse_norm / self.config.fuel_capacity_delta_v
                    if delta_fuel > available_fuel and delta_fuel > 0.0:
                        scale = available_fuel / delta_fuel
                        cmd_impulse *= scale
                        delta_fuel = available_fuel
                        impulse_norm = delta_fuel * self.config.fuel_capacity_delta_v

                    if self.config.control_mode == "continuous_acceleration":
                        accel = cmd_impulse / self.config.time_step_s
                        s_agent = np.concatenate([current_position, current_velocity], axis=0)
                        s_next = (self._cw_stm @ s_agent) + (self._cw_gamma @ accel)
                    else:
                        v_post_impulse = current_velocity + cmd_impulse
                        s_agent = np.concatenate([current_position, v_post_impulse], axis=0)
                        s_next = self._cw_stm @ s_agent

                    next_position = np.clip(
                        s_next[:3],
                        -self.config.world_range_km,
                        self.config.world_range_km,
                    ).astype(np.float32)
                    next_velocity = np.clip(
                        s_next[3:],
                        -self.config.velocity_range_km_s,
                        self.config.velocity_range_km_s,
                    ).astype(np.float32)
                    next_fuel = max(0.0, available_fuel - delta_fuel)
                    fuel_cost = impulse_norm
                else:
                    # Kinematic translation (backward compatibility mode)
                    if burn_norm > 0.0:
                        burn_vector = (
                            burn_direction / burn_norm * self.config.translation_step_km
                        ).astype(np.float32)
                    else:
                        burn_vector = np.zeros(3, dtype=np.float32)
                        rewards[agent_id] -= self.config.idle_penalty

                    if target_index is not None:
                        target_position = snapshot_target_positions[target_index].copy()
                        fuel_cost = self._fuel_cost_proxy(
                            agent_position=current_position,
                            target_position=target_position,
                            burn_vector=burn_vector,
                        )
                    else:
                        fuel_cost = 0.1 * float(np.linalg.norm(burn_vector))
                        if invalid_penalty == 0.0:
                            rewards[agent_id] -= self.config.invalid_action_penalty

                    delta_fuel = fuel_cost / self.config.fuel_capacity_delta_v
                    if delta_fuel > available_fuel and delta_fuel > 0.0:
                        scale = available_fuel / delta_fuel
                        burn_vector *= scale
                        delta_fuel = available_fuel
                        fuel_cost = delta_fuel * self.config.fuel_capacity_delta_v

                    next_position = np.clip(
                        current_position + burn_vector,
                        -self.config.world_range_km,
                        self.config.world_range_km,
                    ).astype(np.float32)
                    next_velocity = current_velocity
                    next_fuel = max(0.0, available_fuel - delta_fuel)

            # Target tracking, closing progress, and distances
            if target_index is not None:
                prev_target_pos = snapshot_target_positions[target_index]
                new_target_pos = proposed_target_positions[target_index]
                previous_distance = float(np.linalg.norm(prev_target_pos - current_position))
                new_distance = float(np.linalg.norm(new_target_pos - next_position))
                closing_delta = float(previous_distance - new_distance)
                rewards[agent_id] += self.config.progress_reward_scale * (
                    closing_delta / self.config.world_range_km
                )
            else:
                new_distance = float(np.inf)
                closing_delta = 0.0
                if invalid_penalty == 0.0 and self.config.dynamics_model == "clohessy_wiltshire":
                    rewards[agent_id] -= self.config.invalid_action_penalty

            rewards[agent_id] -= self.config.fuel_penalty_scale * delta_fuel

            proposed_positions[agent_index] = next_position
            proposed_velocities[agent_index] = next_velocity
            proposed_fuel[agent_index] = next_fuel
            proposed_delta_v[agent_index] = float(fuel_cost)
            proposed_closing_delta[agent_index] = closing_delta
            proposed_distances[agent_index] = float(new_distance)

        # ------------------------------------------------------------------
        # 4. Resolve interactions jointly (target conflicts, captures, collisions)
        # ------------------------------------------------------------------
        capture_events: List[Tuple[str, int]] = []
        step_capture_flags = np.zeros((self.config.num_agents,), dtype=np.int8)

        # Group eligible capturing agents by target index
        for target_idx in np.where(snapshot_target_active)[0]:
            competing_agents: List[int] = []
            for agent_idx in range(self.config.num_agents):
                if snapshot_agent_fuel[agent_idx] <= 0.0:
                    continue
                if proposed_selected_targets[agent_idx] == target_idx:
                    if proposed_distances[agent_idx] <= self.config.capture_radius_km:
                        competing_agents.append(agent_idx)

            if not competing_agents:
                continue

            # Deterministic conflict resolution / tie-breaking:
            # 1. Primary: Minimum post-transition distance to target (closer agent wins).
            # 2. Secondary: Lower agent index (e.g. agent_0 over agent_1) for identical distances within 1e-5 km.
            competing_agents.sort(
                key=lambda a_idx: (
                    round(float(proposed_distances[a_idx]), 5),
                    a_idx,
                )
            )
            winner_agent_idx = competing_agents[0]
            winner_agent_id = self.possible_agents[winner_agent_idx]

            capture_events.append((winner_agent_id, int(target_idx)))
            step_capture_flags[winner_agent_idx] = 1
            rewards[winner_agent_id] += self.config.capture_reward

        # Global cooperative reward sharing: teammates receive bonus
        for capturing_agent_id, target_index in capture_events:
            for teammate_id in self.possible_agents:
                if teammate_id != capturing_agent_id:
                    rewards[teammate_id] += self.config.cooperative_reward_share

        # Inter-agent collision detection (if configured)
        collision_events: List[Tuple[str, str]] = []
        if getattr(self.config, "collision_radius_km", 0.0) > 0.0:
            for idx_a in range(self.config.num_agents):
                for idx_b in range(idx_a + 1, self.config.num_agents):
                    pair_distance = float(
                        np.linalg.norm(proposed_positions[idx_a] - proposed_positions[idx_b])
                    )
                    if pair_distance <= self.config.collision_radius_km:
                        id_a, id_b = self.possible_agents[idx_a], self.possible_agents[idx_b]
                        collision_events.append((id_a, id_b))
                        if getattr(self.config, "inter_agent_collision_penalty", 0.0) > 0.0:
                            rewards[id_a] -= self.config.inter_agent_collision_penalty
                            rewards[id_b] -= self.config.inter_agent_collision_penalty

        # ------------------------------------------------------------------
        # 5. Commit the next state and rewards
        # ------------------------------------------------------------------
        self._agent_positions = proposed_positions.copy()
        self._agent_velocities = proposed_velocities.copy()
        self._agent_fuel = proposed_fuel.copy()
        self._target_positions = proposed_target_positions.copy()
        self._target_velocities = proposed_target_velocities.copy()
        for capturing_agent_id, target_index in capture_events:
            self._target_active[target_index] = False
            self._capture_count += 1

        for agent_idx, agent_id in enumerate(self.possible_agents):
            infos[agent_id] = self._build_info(
                agent_index=agent_idx,
                selected_target_index=(
                    int(proposed_selected_targets[agent_idx])
                    if proposed_selected_targets[agent_idx] >= 0
                    else None
                ),
                fuel_cost_proxy=float(proposed_delta_v[agent_idx]),
                capture_event=bool(step_capture_flags[agent_idx]),
            )
            if proposed_selected_targets[agent_idx] < 0:
                infos[agent_id]["invalid_target_selection"] = True
            if step_capture_flags[agent_idx]:
                captured_slot = int(proposed_selected_targets[agent_idx])
                infos[agent_id]["captured_target_index"] = captured_slot
                infos[agent_id]["rendezvous_target_index"] = captured_slot

        self._step_count += 1
        self._episode_history["agent_positions"].append(self._agent_positions.copy())
        self._episode_history["agent_velocities"].append(self._agent_velocities.copy())
        self._episode_history["agent_states"].append(
            np.stack([self.get_agent_state(i) for i in range(self.config.num_agents)], axis=0)
        )
        self._episode_history["agent_fuel"].append(self._agent_fuel.copy())
        self._episode_history["target_positions"].append(self._target_positions.copy())
        self._episode_history["target_velocities"].append(self._target_velocities.copy())
        self._episode_history["target_states"].append(
            np.stack([self.get_target_state(i) for i in range(self.max_targets)], axis=0)
        )
        self._episode_history["target_active"].append(
            self._target_active.astype(np.int8).copy()
        )
        self._episode_history["step_rewards"].append(
            np.asarray(
                [rewards[agent_id] for agent_id in self.possible_agents],
                dtype=np.float32,
            )
        )
        self._episode_history["fuel_cost_proxy"].append(proposed_delta_v.copy())
        self._episode_history["delta_v"].append(proposed_delta_v.copy())
        self._episode_history["selected_targets"].append(proposed_selected_targets.copy())
        self._episode_history["capture_matrix"].append(step_capture_flags.copy())
        self._episode_history["action_commands"].append(proposed_action_commands.copy())
        self._episode_history["closing_delta"].append(proposed_closing_delta.copy())

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

        filtered_infos = dict(infos)
        filtered_infos["__common__"] = {
            "step_count": self._step_count,
            "captures_completed": self._capture_count,
            "rendezvous_completed": self._capture_count,
            "active_targets_remaining": int(np.count_nonzero(self._target_active)),
        }
        if collision_events:
            filtered_infos["__common__"]["collision_events"] = collision_events

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

    def _resolve_target_velocities(
        self,
        options: Mapping[str, Any],
        positions: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        configured_vels = options.get("target_velocities")
        if configured_vels is None:
            configured_vels = self.config.target_velocities

        velocities = np.zeros((self.max_targets, 3), dtype=np.float32)
        if configured_vels is not None:
            raw_vels = np.asarray(configured_vels, dtype=np.float32)
            usable = min(raw_vels.shape[0], self.max_targets)
            velocities[:usable] = raw_vels[:usable]
            return velocities

        if self.config.dynamics_model == "clohessy_wiltshire":
            # In Hill relative motion, setting vy0 = -2 * n * x0 cancels secular drift,
            # producing a bounded, periodic 2:1 relative ellipse.
            for idx in range(self.max_targets):
                if not active[idx]:
                    continue
                if self.config.target_drift_mode == "periodic_ellipse":
                    x0 = float(positions[idx, 0])
                    vy0 = -2.0 * self.mean_motion_n * x0
                    velocities[idx] = [0.0, vy0, 0.0]
                elif self.config.target_drift_mode == "free_drift":
                    velocities[idx] = self._rng.uniform(
                        -self.config.velocity_range_km_s * 0.1,
                        self.config.velocity_range_km_s * 0.1,
                        size=3,
                    ).astype(np.float32)
                elif self.config.target_drift_mode == "stationary_initial":
                    velocities[idx] = np.zeros(3, dtype=np.float32)

                if self.config.target_velocity_noise_km_s > 0.0:
                    noise = self._rng.normal(
                        0.0, self.config.target_velocity_noise_km_s, size=3
                    ).astype(np.float32)
                    velocities[idx] += noise

        return np.clip(
            velocities,
            -self.config.velocity_range_km_s,
            self.config.velocity_range_km_s,
        ).astype(np.float32)

    def _resolve_agent_velocities(
        self, options: Mapping[str, Any], positions: np.ndarray
    ) -> np.ndarray:
        configured_vels = options.get("agent_velocities")
        if configured_vels is None:
            configured_vels = self.config.agent_velocities

        velocities = np.zeros((self.config.num_agents, 3), dtype=np.float32)
        if configured_vels is not None:
            raw_vels = np.asarray(configured_vels, dtype=np.float32)
            usable = min(raw_vels.shape[0], self.config.num_agents)
            velocities[:usable] = raw_vels[:usable]
            return velocities

        if self.config.dynamics_model == "clohessy_wiltshire":
            for idx in range(self.config.num_agents):
                x0 = float(positions[idx, 0])
                vy0 = -2.0 * self.mean_motion_n * x0
                velocities[idx] = [0.0, vy0, 0.0]

        return np.clip(
            velocities,
            -self.config.velocity_range_km_s,
            self.config.velocity_range_km_s,
        ).astype(np.float32)

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
        obs_cols = 10 if self.config.obs_includes_velocity else 6
        matrix = np.zeros((self.max_targets, obs_cols), dtype=np.float32)
        agent_position = self._agent_positions[agent_index]
        agent_velocity = self._agent_velocities[agent_index]
        fuel_norm = self._normalize_fuel(self._agent_fuel[agent_index])

        for target_index in range(self.max_targets):
            matrix[target_index, 5] = fuel_norm
            if not self._target_active[target_index]:
                matrix[target_index, 0] = 0.0
                continue

            matrix[target_index, 0] = 1.0
            relative_vector = self._target_positions[target_index] - agent_position
            distance = float(np.linalg.norm(relative_vector))
            matrix[target_index, 1:4] = self._clip(
                relative_vector / self.config.world_range_km
            )
            matrix[target_index, 4] = float(
                np.clip(distance / self.config.world_range_km, 0.0, 1.0)
            )

            if self.config.obs_includes_velocity:
                relative_vel = self._target_velocities[target_index] - agent_velocity
                rel_speed = float(np.linalg.norm(relative_vel))
                matrix[target_index, 6:9] = self._clip(
                    relative_vel / self.config.velocity_range_km_s
                )
                matrix[target_index, 9] = float(
                    np.clip(rel_speed / self.config.velocity_range_km_s, 0.0, 1.0)
                )

        return self._clip(matrix)

    def get_action_mask(
        self, agent_id_or_index: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """
        Return the boolean action mask for target selection.

        Shape: (max_targets,), where mask[j] is True if target slot j is currently
        active and available for selection, and False otherwise.
        """
        mask = np.zeros((self.max_targets,), dtype=bool)
        usable = min(len(self._target_active), self.max_targets)
        mask[:usable] = self._target_active[:usable]
        return mask

    def _build_info(
        self,
        *,
        agent_index: int,
        selected_target_index: Optional[int],
        fuel_cost_proxy: Optional[float] = None,
        delta_v: Optional[float] = None,
        capture_event: bool,
    ) -> Dict[str, Any]:
        cost_val = float(
            fuel_cost_proxy
            if fuel_cost_proxy is not None
            else (delta_v if delta_v is not None else 0.0)
        )
        agent_state = self.get_agent_state(agent_index)
        info: Dict[str, Any] = {
            "agent_index": agent_index,
            "step_count": self._step_count,
            "selected_target_index": selected_target_index,
            "fuel_cost_proxy": cost_val,
            "delta_v": cost_val,  # Retained as backwards-compatible alias
            "fuel_remaining": float(self._agent_fuel[agent_index]),
            "active_targets_remaining": int(np.count_nonzero(self._target_active)),
            "capture_event": bool(capture_event),
            "rendezvous_event": bool(capture_event),  # Explicit operational nomenclature
            "action_mask": self.get_action_mask(agent_index).tolist(),
            "agent_state": agent_state.tolist(),
            "agent_velocity": agent_state[3:].tolist(),
        }
        if selected_target_index is not None and 0 <= selected_target_index < self.max_targets:
            target_state = self.get_target_state(selected_target_index)
            rel_vel = target_state[3:] - agent_state[3:]
            info["target_state"] = target_state.tolist()
            info["relative_velocity"] = rel_vel.tolist()
            info["relative_speed"] = float(np.linalg.norm(rel_vel))
        return info

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

    def _resolve_target_index(
        self,
        requested_index: int,
        target_active: Optional[np.ndarray] = None,
        agent_position: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Resolve requested target index against the active targets mask.

        Uses target_active mask if provided (e.g. pre-step snapshot), otherwise
        defaults to current self._target_active.

        Resolution policies (self.config.target_resolution_policy):
        - 'no_op' (default / scientific): Returns None if requested target is inactive.
          The agent is not silently redirected, incurring an invalid action penalty and
          preventing unguided target captures. Exposes action masks to enable valid choice filtering.
        - 'nearest_agent': Redirects to the active target closest to the agent's current
          position (using Euclidean distance from agent_position, not coordinate origin).
        """
        active_mask = target_active if target_active is not None else self._target_active
        if 0 <= requested_index < len(active_mask) and active_mask[requested_index]:
            return requested_index

        active_indices = np.where(active_mask)[0]
        if active_indices.size == 0:
            return None

        if self.config.target_resolution_policy == "nearest_agent":
            if agent_position is not None:
                diffs = self._target_positions[active_indices] - agent_position
                distances = np.linalg.norm(diffs, axis=1)
                return int(active_indices[np.argmin(distances)])
            cluster_positions = self._target_positions[active_indices]
            cluster_norms = np.linalg.norm(cluster_positions, axis=1)
            return int(active_indices[np.argmin(cluster_norms)])

        # 'no_op' policy: do not silently redirect to any target
        return None

    def _fuel_cost_proxy(
        self,
        *,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        burn_vector: np.ndarray,
    ) -> float:
        """
        Compute an empirical fuel-cost proxy for a commanded burn toward a target.

        IMPORTANT / SCIENTIFIC DISCLAIMER:
        This value is a heuristic fuel-cost proxy used for RL reward shaping and
        propellant depletion accounting. It is NOT a physically validated Keplerian
        or Lambert boundary-value Delta-V calculation.
        Until a full orbital dynamics and maneuver solver is integrated, this value
        must be reported as a heuristic fuel-cost proxy.

        The heuristic formula combines:
        - `k1_phasing_cost`: Penalizes in-plane relative displacement.
        - `k2_plane_change_cost`: Penalizes out-of-plane relative displacement
          (intentionally higher to represent the elevated energy cost of plane changes).
        - Direct burn magnitude penalty (0.1 * ||burn_vector||).
        - Alignment penalty: Penalizes burns oriented away from the line-of-sight to target.
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

        fuel_cost = (
            self.config.k1_phasing_cost * in_plane_distance
            + self.config.k2_plane_change_cost * cross_track_distance
            + 0.1 * burn_magnitude
            + alignment_penalty
        )
        return max(0.0, float(fuel_cost))

    # Backwards-compatible alias for existing callers and legacy tests
    _lambert_proxy_delta_v = _fuel_cost_proxy

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
            "agent_velocities": deepcopy(self._agent_velocities.tolist()),
            "target_positions": deepcopy(self._target_positions.tolist()),
            "target_velocities": deepcopy(self._target_velocities.tolist()),
            "target_active": deepcopy(self._target_active.tolist()),
        }

    def close(self) -> None:
        self.agents = []
