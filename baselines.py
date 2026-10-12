"""
Non-learning baseline policies and evaluation harness for Multi-Agent Orbital Debris Removal.

Implements two deterministic baseline controllers:
1. Nearest-Target Heuristic:
   Each satellite selects the closest available target based on normalized Euclidean
   distance and executes a deterministic line-of-sight guidance rule.

2. Fuel-Aware Heuristic:
   Target selection considers both distance and estimated maneuver cost (penalizing
   energy-intensive cross-track plane changes and accounting for remaining propellant budget)
   rather than distance alone, with fuel-conserving guidance.

These baselines provide empirical performance benchmarks to quantify the added value
of learned reinforcement learning policies (e.g. Independent PPO) over non-learning rules.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from orbital_env import OrbitalDebrisRemovalEnv

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# 1. Observation Extraction Helpers
# ==============================================================================

def extract_observation_matrix(observation: Any) -> np.ndarray:
    """
    Extract the 2D local target observation matrix from an observation payload.
    Supports either Box array of shape (max_targets, feature_dim) or dictionary envelopes.
    """
    if isinstance(observation, Mapping):
        if "local_targets" in observation:
            return np.asarray(observation["local_targets"], dtype=np.float32)
        if "obs" in observation:
            return np.asarray(observation["obs"], dtype=np.float32)
    return np.asarray(observation, dtype=np.float32)


def get_agent_fuel_fraction(observation_matrix: np.ndarray) -> float:
    """
    Extract normalized propellant fraction [0.0, 1.0] from the observation matrix.
    In the environment, feature column index 5 contains fuel normalized to [-1.0, 1.0].
    """
    if observation_matrix.ndim >= 2 and observation_matrix.shape[1] >= 6:
        raw_fuel = float(observation_matrix[0, 5])
        return float(np.clip((raw_fuel + 1.0) / 2.0, 0.0, 1.0))
    return 1.0


# ==============================================================================
# 2. Guidance Rules
# ==============================================================================

def deterministic_los_burn(
    relative_vector: np.ndarray,
    *,
    deadband: float = 0.02,
    damping_vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Deterministic Line-of-Sight (LOS) burn rule mapping relative displacement to discrete commands.

    Discrete action mapping per axis:
    - 0: Negative thrust (-1)
    - 1: Zero thrust / Coast (0)
    - 2: Positive thrust (+1)

    Args:
        relative_vector: 3D vector [x, y, z] pointing from agent to target in normalized units.
        deadband: Range deadband below which thrusters coast to avoid chattering.
        damping_vector: Optional normalized relative velocity to damp excessive closing speed.

    Returns:
        Integer array of shape (3,) with values in {0, 1, 2}.
    """
    burn = np.ones((3,), dtype=np.int64)  # Default: coast [1, 1, 1]
    damped_axes = set()

    for axis in range(3):
        pos_val = float(relative_vector[axis])

        # If damping is enabled and velocity is already closing range rapidly, coast along this axis
        if damping_vector is not None:
            vel_val = float(damping_vector[axis])
            if pos_val * vel_val < -0.05:  # closing rate is strong
                damped_axes.add(axis)
                continue

        if pos_val > deadband:
            burn[axis] = 2  # Burn positive along axis to close distance
        elif pos_val < -deadband:
            burn[axis] = 0  # Burn negative along axis to close distance

    # If completely within deadband and not intentionally damped, select dominant axis nudge
    if np.all(burn == 1) and not damped_axes:
        dominant_axis = int(np.argmax(np.abs(relative_vector)))
        val = float(relative_vector[dominant_axis])
        if abs(val) > 1e-4:
            burn[dominant_axis] = 2 if val > 0.0 else 0

    return burn


# ==============================================================================
# 3. Baseline 1: Nearest-Target Heuristic
# ==============================================================================

class NearestTargetPolicy:
    """
    Nearest-target heuristic controller.

    Each satellite evaluates active target slots, selects the one minimizing
    Euclidean distance ||r_rel||, and commands deterministic line-of-sight burns.
    """

    def __init__(self, *, deadband: float = 0.02) -> None:
        self.deadband = deadband

    def compute_action(
        self,
        observation: Any,
        agent_id: str = "",
    ) -> np.ndarray:
        """
        Compute a MultiDiscrete action [target_index, burn_x, burn_y, burn_z] for a single agent.
        """
        obs_matrix = extract_observation_matrix(observation)
        if obs_matrix.ndim < 2 or obs_matrix.shape[0] == 0:
            return np.array([0, 1, 1, 1], dtype=np.int64)

        max_targets = obs_matrix.shape[0]
        best_target_index: Optional[int] = None
        best_distance = float("inf")
        best_rel_vec = np.zeros((3,), dtype=np.float32)

        for target_idx in range(max_targets):
            row = obs_matrix[target_idx]
            # Check availability indicator m_j (feature 0)
            is_active = (row[0] > 0.5) if obs_matrix.shape[1] >= 1 else True
            if not is_active:
                continue

            rel_vec = row[1:4]
            dist = float(row[4]) if obs_matrix.shape[1] >= 5 else float(np.linalg.norm(rel_vec))

            # Skip degenerate zero-distance inactive slots
            if dist < 1e-6 and np.linalg.norm(rel_vec) < 1e-6:
                continue

            if dist < best_distance:
                best_distance = dist
                best_target_index = target_idx
                best_rel_vec = rel_vec.copy()

        # If no active target is available, coast
        if best_target_index is None:
            return np.array([0, 1, 1, 1], dtype=np.int64)

        burn = deterministic_los_burn(best_rel_vec, deadband=self.deadband)
        return np.array([best_target_index, burn[0], burn[1], burn[2]], dtype=np.int64)

    def choose_actions(
        self,
        observations: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute actions for all agents in the fleet."""
        return {
            agent_id: self.compute_action(obs, agent_id=agent_id)
            for agent_id, obs in observations.items()
        }


# ==============================================================================
# 4. Baseline 2: Fuel-Aware Heuristic
# ==============================================================================

class FuelAwarePolicy:
    """
    Fuel-aware heuristic controller.

    Target selection balances Euclidean distance against estimated maneuver cost:
    - Distinguishes lower-energy in-plane phasing (x, y) from high-energy out-of-plane
      plane changes (z).
    - Dynamically scales plane-change penalties by the inverse of remaining propellant fraction.
    - Employs fuel-conserving guidance with wider deadbands and coasting gates.
    """

    def __init__(
        self,
        *,
        k1_phasing_weight: float = 0.025,
        k2_plane_change_weight: float = 0.085,
        deadband: float = 0.035,
        velocity_damping: bool = True,
    ) -> None:
        self.k1 = k1_phasing_weight
        self.k2 = k2_plane_change_weight
        self.deadband = deadband
        self.velocity_damping = velocity_damping

    def compute_action(
        self,
        observation: Any,
        agent_id: str = "",
    ) -> np.ndarray:
        """
        Compute a MultiDiscrete action [target_index, burn_x, burn_y, burn_z] balancing distance and fuel.
        """
        obs_matrix = extract_observation_matrix(observation)
        if obs_matrix.ndim < 2 or obs_matrix.shape[0] == 0:
            return np.array([0, 1, 1, 1], dtype=np.int64)

        max_targets = obs_matrix.shape[0]
        fuel_fraction = get_agent_fuel_fraction(obs_matrix)

        best_target_index: Optional[int] = None
        best_cost = float("inf")
        best_rel_vec = np.zeros((3,), dtype=np.float32)
        best_rel_vel: Optional[np.ndarray] = None

        has_velocity = (obs_matrix.shape[1] >= 10)

        for target_idx in range(max_targets):
            row = obs_matrix[target_idx]
            is_active = (row[0] > 0.5) if obs_matrix.shape[1] >= 1 else True
            if not is_active:
                continue

            rel_vec = row[1:4]
            dist = float(row[4]) if obs_matrix.shape[1] >= 5 else float(np.linalg.norm(rel_vec))

            if dist < 1e-6 and np.linalg.norm(rel_vec) < 1e-6:
                continue

            # Astrodynamic estimated maneuver cost:
            # in-plane phasing vs. cross-track plane change
            in_plane_dist = float(np.linalg.norm(rel_vec[:2]))
            cross_track_dist = float(abs(rel_vec[2]))

            maneuver_cost_proxy = (
                self.k1 * in_plane_dist + self.k2 * cross_track_dist
            )

            # Fuel scarcity multiplier: as fuel drops, plane changes become heavily penalized
            fuel_scarcity_penalty = 1.0 / max(fuel_fraction, 0.1)

            # Composite heuristic score:
            # Distance + propellant-weighted maneuver expense
            score = (
                dist * (1.0 + 0.5 * (1.0 - fuel_fraction))
                + fuel_scarcity_penalty * maneuver_cost_proxy
            )

            if score < best_cost:
                best_cost = score
                best_target_index = target_idx
                best_rel_vec = rel_vec.copy()
                if has_velocity and self.velocity_damping:
                    best_rel_vel = row[6:9].copy()

        if best_target_index is None:
            return np.array([0, 1, 1, 1], dtype=np.int64)

        # Fuel-conserving guidance: wider deadband and coasting
        # If fuel is critically low (< 15%), gate cross-track burns
        rel_vec_guided = best_rel_vec.copy()
        if fuel_fraction < 0.15:
            rel_vec_guided[2] = 0.0  # Conserve fuel by suppressing plane-change burns

        burn = deterministic_los_burn(
            rel_vec_guided,
            deadband=self.deadband,
            damping_vector=best_rel_vel,
        )
        return np.array([best_target_index, burn[0], burn[1], burn[2]], dtype=np.int64)

    def choose_actions(
        self,
        observations: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute actions for all agents in the fleet."""
        return {
            agent_id: self.compute_action(obs, agent_id=agent_id)
            for agent_id, obs in observations.items()
        }


# ==============================================================================
# 5. Benchmarking & Comparative Evaluation Harness
# ==============================================================================

@dataclass
class BaselineMetrics:
    """Statistical summary metrics for a baseline policy evaluation run."""

    policy_name: str
    num_episodes: int
    mean_reward: float
    mean_captures: float
    mean_capture_accuracy_pct: float
    full_sweep_rate_pct: float
    mean_fuel_used_pct: float
    mean_final_fuel_pct: float
    mean_delta_v_total: float
    mean_steps: float
    mean_guidance_accuracy_pct: float


def evaluate_policy(
    policy_fn_or_obj: Union[NearestTargetPolicy, FuelAwarePolicy, Callable[[Mapping[str, Any]], Dict[str, np.ndarray]]],
    env: OrbitalDebrisRemovalEnv,
    *,
    policy_name: str = "Baseline",
    num_episodes: int = 20,
    seed_start: int = 42,
) -> BaselineMetrics:
    """
    Evaluate a policy over multiple deterministic episodes and aggregate performance metrics.
    """
    total_rewards: List[float] = []
    total_captures: List[int] = []
    capture_accuracies: List[float] = []
    full_sweeps: List[bool] = []
    fuels_used_pct: List[float] = []
    final_fuels_pct: List[float] = []
    delta_vs: List[float] = []
    step_counts: List[int] = []
    guidance_accuracies: List[float] = []

    choose_actions = (
        policy_fn_or_obj.choose_actions
        if hasattr(policy_fn_or_obj, "choose_actions")
        else policy_fn_or_obj
    )

    for episode_idx in range(num_episodes):
        ep_seed = seed_start + episode_idx
        obs, _ = env.reset(seed=ep_seed)
        step = 0

        while obs:
            actions = choose_actions(obs)
            obs, rewards, term, trunc, info = env.step(actions)
            step += 1
            if term.get("__all__", False) or trunc.get("__all__", False):
                break

        history = env.get_episode_history()
        target_active = np.asarray(history.get("target_active", []), dtype=bool)
        agent_fuel = np.asarray(history.get("agent_fuel", []), dtype=np.float32)
        step_rewards = np.asarray(history.get("step_rewards", []), dtype=np.float32)
        delta_v = np.asarray(history.get("delta_v", []), dtype=np.float32)
        closing_delta = np.asarray(history.get("closing_delta", []), dtype=np.float32)
        selected_targets = np.asarray(history.get("selected_targets", []), dtype=np.int16)

        num_targets = env.config.num_targets
        captured = int(num_targets - target_active[-1].sum()) if target_active.size else 0
        cap_acc = 100.0 * captured / max(1, num_targets)
        is_full_sweep = (captured >= num_targets and num_targets > 0)

        initial_total_fuel = float(agent_fuel[0].sum()) if agent_fuel.size else 1.0
        final_total_fuel = float(agent_fuel[-1].sum()) if agent_fuel.size else 0.0
        fuel_used_pct = 100.0 * max(0.0, initial_total_fuel - final_total_fuel) / max(initial_total_fuel, 1e-6)
        mean_final_fuel = 100.0 * float(agent_fuel[-1].mean()) if agent_fuel.size else 0.0

        # Guidance accuracy (% engagements that closed distance)
        engagement_mask = selected_targets >= 0
        guidance_success = np.logical_and(engagement_mask, closing_delta > 0.0)
        tot_engagements = int(engagement_mask.sum()) if engagement_mask.size else 0
        succ_engagements = int(guidance_success.sum()) if guidance_success.size else 0
        guidance_acc = 100.0 * succ_engagements / max(1, tot_engagements)

        total_rewards.append(float(step_rewards.sum()) if step_rewards.size else 0.0)
        total_captures.append(captured)
        capture_accuracies.append(cap_acc)
        full_sweeps.append(is_full_sweep)
        fuels_used_pct.append(fuel_used_pct)
        final_fuels_pct.append(mean_final_fuel)
        delta_vs.append(float(delta_v.sum()) if delta_v.size else 0.0)
        step_counts.append(step)
        guidance_accuracies.append(guidance_acc)

    return BaselineMetrics(
        policy_name=policy_name,
        num_episodes=num_episodes,
        mean_reward=float(np.mean(total_rewards)),
        mean_captures=float(np.mean(total_captures)),
        mean_capture_accuracy_pct=float(np.mean(capture_accuracies)),
        full_sweep_rate_pct=100.0 * float(np.mean(full_sweeps)),
        mean_fuel_used_pct=float(np.mean(fuels_used_pct)),
        mean_final_fuel_pct=float(np.mean(final_fuels_pct)),
        mean_delta_v_total=float(np.mean(delta_vs)),
        mean_steps=float(np.mean(step_counts)),
        mean_guidance_accuracy_pct=float(np.mean(guidance_accuracies)),
    )


def compare_baselines(
    env_config: Optional[Mapping[str, Any]] = None,
    *,
    num_episodes: int = 20,
    seed_start: int = 42,
) -> Dict[str, BaselineMetrics]:
    """
    Run side-by-side evaluation of both non-learning baselines over identical random seeds.
    """
    cfg = dict(env_config or {})
    env = OrbitalDebrisRemovalEnv(cfg)

    nearest_policy = NearestTargetPolicy()
    fuel_aware_policy = FuelAwarePolicy()

    results: Dict[str, BaselineMetrics] = {}
    results["Nearest-Target"] = evaluate_policy(
        nearest_policy,
        env,
        policy_name="Nearest-Target Heuristic",
        num_episodes=num_episodes,
        seed_start=seed_start,
    )
    results["Fuel-Aware"] = evaluate_policy(
        fuel_aware_policy,
        env,
        policy_name="Fuel-Aware Heuristic",
        num_episodes=num_episodes,
        seed_start=seed_start,
    )

    env.close()
    return results


def print_comparison_table(results: Dict[str, BaselineMetrics]) -> None:
    """Format and print a markdown comparison table of baseline metrics."""
    headers = [
        "Policy",
        "Rendezvous / Target",
        "Capture Acc (%)",
        "Full Sweep (%)",
        "Fuel Used (%)",
        "Mean Delta-V",
        "Guidance Acc (%)",
        "Mean Steps",
        "Episode Reward",
    ]
    print("\n" + "=" * 96)
    print("NON-LEARNING BASELINES BENCHMARK COMPARISON")
    print("=" * 96)
    fmt_row = "{:<25} {:<18} {:<16} {:<15} {:<14} {:<13} {:<17} {:<12} {:<15}"
    print(fmt_row.format(*headers))
    print("-" * 145)

    for metrics in results.values():
        row = [
            metrics.policy_name,
            f"{metrics.mean_captures:.1f}",
            f"{metrics.mean_capture_accuracy_pct:.1f}%",
            f"{metrics.full_sweep_rate_pct:.1f}%",
            f"{metrics.mean_fuel_used_pct:.1f}%",
            f"{metrics.mean_delta_v_total:.2f}",
            f"{metrics.mean_guidance_accuracy_pct:.1f}%",
            f"{metrics.mean_steps:.1f}",
            f"{metrics.mean_reward:.2f}",
        ]
        print(fmt_row.format(*row))
    print("=" * 96 + "\n")


# ==============================================================================
# 6. CLI Entry Point
# ==============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run non-learning baseline controllers (Nearest-Target & Fuel-Aware) on the ADR environment."
    )
    parser.add_argument(
        "--baseline",
        choices=["all", "nearest", "fuel_aware"],
        default="all",
        help="Which baseline to evaluate (default: all).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per baseline (default: 20).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of chaser satellites (default: 3).",
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=5,
        help="Number of debris targets (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting evaluation random seed (default: 42).",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    env_config = {
        "num_agents": args.agents,
        "num_targets": args.targets,
        "max_targets": args.targets,
        "random_seed": args.seed,
    }

    print(f"Evaluating non-learning baselines over {args.episodes} episodes ({args.agents} agents, {args.targets} targets)...")

    if args.baseline == "all":
        results = compare_baselines(env_config, num_episodes=args.episodes, seed_start=args.seed)
        print_comparison_table(results)
    else:
        env = OrbitalDebrisRemovalEnv(env_config)
        if args.baseline == "nearest":
            policy = NearestTargetPolicy()
            name = "Nearest-Target Heuristic"
        else:
            policy = FuelAwarePolicy()
            name = "Fuel-Aware Heuristic"
        metrics = evaluate_policy(policy, env, policy_name=name, num_episodes=args.episodes, seed_start=args.seed)
        env.close()
        print_comparison_table({args.baseline: metrics})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
