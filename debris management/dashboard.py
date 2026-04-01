"""
Interactive dashboard for exploring the orbital debris MARL environment.

The dashboard is intentionally useful before full RL training is complete:
- Cooperative Autopilot mode lets you stress-test the physics and reward model.
- RL Checkpoint mode lets you compare a trained decentralized policy against the same scenario.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fetch_data import load_cached_records, records_to_relative_positions
from orbital_env import OrbitalDebrisRemovalEnv

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "training_manifest.json"
EXPECTED_TRAINING_TOPOLOGY = "decentralized_ippo"
DEMO_DEFAULTS = {
    "num_agents": 3,
    "num_targets": 5,
    "max_targets": 5,
    "max_episode_steps": 300,
    "random_seed": 42,
    "world_range_km": 250.0,
    "capture_radius_km": 12.0,
    "translation_step_km": 18.0,
    "agent_spawn_radius_km": 15.0,
    "target_position_noise_km": 4.0,
    "initial_fuel": 1.0,
    "fuel_capacity_delta_v": 300.0,
    "k1_phasing_cost": 0.025,
    "k2_plane_change_cost": 0.085,
    "capture_reward": 15.0,
    "cooperative_reward_share": 4.0,
    "progress_reward_scale": 2.0,
    "fuel_penalty_scale": 0.4,
    "idle_penalty": 0.03,
    "invalid_action_penalty": 0.2,
}


@st.cache_data(show_spinner=False)
def read_manifest(manifest_path: str) -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_target_positions_from_cache(
    cache_path: str,
    world_range_km: float,
    max_targets: int,
) -> List[List[float]]:
    records = load_cached_records(Path(cache_path).resolve())
    return records_to_relative_positions(
        records,
        world_range_km=world_range_km,
        max_targets=max_targets,
    )


def normalize_checkpoint_path(raw_checkpoint_path: str) -> str:
    text = str(raw_checkpoint_path).strip()
    if not text:
        return ""
    if text.startswith("Checkpoint(") and "path=" in text:
        extracted = text.split("path=", maxsplit=1)[1].rstrip(")")
        return extracted.strip()
    return text


def find_manifest_for_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    for parent in checkpoint_path.resolve().parents:
        candidate = parent / "training_manifest.json"
        if candidate.exists():
            return candidate
    return None


def extract_local_targets(observation: Any) -> np.ndarray:
    """
    Normalize the observation payload into the local target matrix.

    The helper keeps the dashboard resilient while the project transitions away
    from the old dict-based CTDE observation contract.
    """

    if isinstance(observation, Mapping):
        if "local_targets" in observation:
            return np.asarray(observation["local_targets"], dtype=np.float32)
        return np.asarray(observation, dtype=np.float32)

    return np.asarray(observation, dtype=np.float32)


@st.cache_resource(show_spinner="Loading RL checkpoint...")
def load_algorithm_from_checkpoint(checkpoint_path: str):
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.registry import register_env

    from train_marl import ENV_NAME, TRAINING_TOPOLOGY

    normalized_checkpoint_path = normalize_checkpoint_path(checkpoint_path)
    if not normalized_checkpoint_path:
        raise ValueError("No RL checkpoint path was provided.")

    checkpoint = Path(normalized_checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    manifest_path = find_manifest_for_checkpoint(checkpoint)
    manifest = read_manifest(str(manifest_path)) if manifest_path else {}
    topology = str(manifest.get("training_topology", "")).strip()
    if manifest and topology != TRAINING_TOPOLOGY:
        if not topology:
            raise RuntimeError(
                "This checkpoint predates the decentralized rewrite. Train a new decentralized model before using RL Checkpoint mode."
            )
        raise RuntimeError(
            f"This checkpoint uses '{topology}', but the dashboard expects '{TRAINING_TOPOLOGY}'."
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    register_env(ENV_NAME, lambda config: OrbitalDebrisRemovalEnv(config))
    return Algorithm.from_checkpoint(str(checkpoint))


def apply_dashboard_theme() -> None:
    st.set_page_config(
        page_title="Orbital Debris MARL Dashboard",
        page_icon="O",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2rem;
            max-width: 1560px;
        }
        .hero-card {
            background: linear-gradient(130deg, #071a2b 0%, #0f3b5c 46%, #0f766e 100%);
            color: #f8fafc;
            padding: 1.45rem 1.6rem;
            border-radius: 24px;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.22);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            margin-bottom: 0.35rem;
        }
        .hero-copy {
            font-size: 1rem;
            opacity: 0.93;
            max-width: 62rem;
        }
        .summary-card {
            background: linear-gradient(180deg, rgba(248, 250, 252, 1) 0%, rgba(240, 249, 255, 1) 100%);
            border: 1px solid rgba(14, 116, 144, 0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-top: 0.5rem;
        }
        .summary-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.3rem;
        }
        .summary-copy {
            font-size: 0.95rem;
            color: #334155;
            line-height: 1.45;
        }
        .stMetric {
            background: #f8fafc;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 0.8rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fuel_fraction_from_observation(observation: Any) -> float:
    local_targets = extract_local_targets(observation)
    if local_targets.size == 0:
        return 1.0
    return float(np.clip((local_targets[0, 0] + 1.0) / 2.0, 0.0, 1.0))


def build_burn_command(relative_vector: np.ndarray) -> np.ndarray:
    """Bias burns toward in-plane motion and avoid unnecessary plane changes."""

    relative_vector = np.asarray(relative_vector, dtype=np.float32)
    thresholds = np.asarray([0.025, 0.025, 0.055], dtype=np.float32)
    burn_command = np.ones((3,), dtype=np.int64)

    for axis, value in enumerate(relative_vector):
        if abs(float(value)) >= float(thresholds[axis]):
            burn_command[axis] = 2 if value > 0.0 else 0

    if np.all(burn_command == 1):
        dominant_axis = int(np.argmax(np.abs(relative_vector)))
        if abs(float(relative_vector[dominant_axis])) > 1e-6:
            burn_command[dominant_axis] = 2 if relative_vector[dominant_axis] > 0.0 else 0

    return burn_command


def cooperative_autopilot_actions(
    observations: Mapping[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Coordinate agents so they spread across targets instead of dog-piling.

    The controller prefers low-distance, low-plane-change assignments and
    becomes more conservative as an agent's fuel fraction decreases.
    """

    candidate_rows: List[Tuple[float, str, int, np.ndarray]] = []
    fallback_targets: Dict[str, Optional[Tuple[int, np.ndarray]]] = {}

    for agent_id, observation in observations.items():
        local_targets = extract_local_targets(observation)
        fuel_fraction = fuel_fraction_from_observation(observation)
        best_distance = float("inf")
        best_fallback: Optional[Tuple[int, np.ndarray]] = None

        for target_index in range(local_targets.shape[0]):
            row = local_targets[target_index]
            relative_vector = row[1:4]
            distance = float(row[4])
            if np.linalg.norm(relative_vector) <= 1e-6 and distance <= 1e-6:
                continue

            if distance < best_distance:
                best_distance = distance
                best_fallback = (target_index, relative_vector.copy())

            plane_change_pressure = abs(float(relative_vector[2]))
            score = (
                distance * (1.0 + 0.35 * (1.0 - fuel_fraction))
                + 0.25 * plane_change_pressure
            )
            candidate_rows.append((score, agent_id, target_index, relative_vector.copy()))

        fallback_targets[agent_id] = best_fallback

    candidate_rows.sort(key=lambda item: item[0])
    assignments: Dict[str, Tuple[int, np.ndarray]] = {}
    assigned_agents = set()
    assigned_targets = set()

    for _, agent_id, target_index, relative_vector in candidate_rows:
        if agent_id in assigned_agents or target_index in assigned_targets:
            continue
        assignments[agent_id] = (target_index, relative_vector)
        assigned_agents.add(agent_id)
        assigned_targets.add(target_index)

    for agent_id in observations:
        if agent_id not in assignments and fallback_targets.get(agent_id) is not None:
            assignments[agent_id] = fallback_targets[agent_id]  # type: ignore[assignment]

    action_dict: Dict[str, np.ndarray] = {}
    for agent_id in observations:
        assignment = assignments.get(agent_id)
        if assignment is None:
            action_dict[agent_id] = np.asarray([0, 1, 1, 1], dtype=np.int64)
            continue
        target_index, relative_vector = assignment
        burn_command = build_burn_command(relative_vector)
        action_dict[agent_id] = np.concatenate(
            ([target_index], burn_command.astype(np.int64))
        ).astype(np.int64)

    return action_dict


def choose_actions(
    *,
    controller_mode: str,
    observations: Mapping[str, Any],
    checkpoint_path: str,
) -> Dict[str, np.ndarray]:
    if controller_mode == "Cooperative Autopilot":
        return cooperative_autopilot_actions(observations)

    from train_marl import policy_id_for_agent

    algo = load_algorithm_from_checkpoint(checkpoint_path)
    action_dict: Dict[str, np.ndarray] = {}
    for agent_id, observation in observations.items():
        action = algo.compute_single_action(
            observation,
            policy_id=policy_id_for_agent(agent_id),
            explore=False,
        )
        if isinstance(action, tuple):
            action = action[0]
        action_dict[agent_id] = np.asarray(action, dtype=np.int64)
    return action_dict


def build_env_config(
    *,
    num_agents: int,
    num_targets: int,
    max_targets: int,
    max_episode_steps: int,
    world_range_km: float,
    capture_radius_km: float,
    translation_step_km: float,
    agent_spawn_radius_km: float,
    target_position_noise_km: float,
    initial_fuel: float,
    fuel_capacity_delta_v: float,
    k1_phasing_cost: float,
    k2_plane_change_cost: float,
    capture_reward: float,
    cooperative_reward_share: float,
    progress_reward_scale: float,
    fuel_penalty_scale: float,
    idle_penalty: float,
    invalid_action_penalty: float,
    random_seed: int,
    debris_cache_path: str,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "num_agents": num_agents,
        "num_targets": num_targets,
        "max_targets": max(max_targets, num_targets),
        "max_episode_steps": max_episode_steps,
        "world_range_km": world_range_km,
        "capture_radius_km": capture_radius_km,
        "translation_step_km": translation_step_km,
        "agent_spawn_radius_km": agent_spawn_radius_km,
        "target_position_noise_km": target_position_noise_km,
        "initial_fuel": initial_fuel,
        "fuel_capacity_delta_v": fuel_capacity_delta_v,
        "k1_phasing_cost": k1_phasing_cost,
        "k2_plane_change_cost": k2_plane_change_cost,
        "capture_reward": capture_reward,
        "cooperative_reward_share": cooperative_reward_share,
        "progress_reward_scale": progress_reward_scale,
        "fuel_penalty_scale": fuel_penalty_scale,
        "idle_penalty": idle_penalty,
        "invalid_action_penalty": invalid_action_penalty,
        "random_seed": random_seed,
    }

    if debris_cache_path.strip():
        target_positions = load_target_positions_from_cache(
            debris_cache_path,
            world_range_km=world_range_km,
            max_targets=config["max_targets"],
        )
        if target_positions:
            config["target_positions"] = target_positions
            config["num_targets"] = min(len(target_positions), config["max_targets"])

    return config


def run_simulation(
    *,
    controller_mode: str,
    checkpoint_path: str,
    env_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    env = OrbitalDebrisRemovalEnv(env_config)
    observations, _ = env.reset(seed=seed)
    step_count = 0

    while observations:
        action_dict = choose_actions(
            controller_mode=controller_mode,
            observations=observations,
            checkpoint_path=checkpoint_path,
        )
        observations, _, terminateds, truncateds, _ = env.step(action_dict)
        step_count += 1
        if terminateds.get("__all__") or truncateds.get("__all__"):
            break

    history = env.get_episode_history()
    env.close()

    step_rewards = np.asarray(history.get("step_rewards", []), dtype=np.float32)
    if step_rewards.ndim == 1 and step_rewards.size:
        step_rewards = step_rewards.reshape(1, -1)

    num_agents = int(env_config["num_agents"])
    agent_reward_totals = {
        f"agent_{agent_index}": (
            float(step_rewards[:, agent_index].sum())
            if step_rewards.size and agent_index < step_rewards.shape[1]
            else 0.0
        )
        for agent_index in range(num_agents)
    }

    capture_matrix = np.asarray(history.get("capture_matrix", []), dtype=np.int8)
    final_fuel = np.asarray(history.get("agent_fuel", []), dtype=np.float32)

    return {
        "history": history,
        "total_reward": float(step_rewards.sum()) if step_rewards.size else 0.0,
        "step_count": step_count,
        "capture_events": int(capture_matrix.sum()) if capture_matrix.size else 0,
        "agent_reward_totals": agent_reward_totals,
        "final_fuel": final_fuel[-1] if final_fuel.size else np.zeros((num_agents,), dtype=np.float32),
        "controller_mode": controller_mode,
        "checkpoint_path": normalize_checkpoint_path(checkpoint_path),
        "env_config": dict(env_config),
    }


def compute_capture_steps(target_active: np.ndarray) -> Dict[int, Optional[int]]:
    capture_steps: Dict[int, Optional[int]] = {}
    if target_active.size == 0:
        return capture_steps

    for target_index in range(target_active.shape[1]):
        transition_indices = np.where(
            np.diff(target_active[:, target_index].astype(np.int8)) == -1
        )[0]
        capture_steps[target_index] = (
            int(transition_indices[0] + 1) if transition_indices.size else None
        )
    return capture_steps


def summarize_mission(result: Mapping[str, Any]) -> Dict[str, Any]:
    history = result["history"]
    env_config = result["env_config"]

    agent_fuel = np.asarray(history.get("agent_fuel", []), dtype=np.float32)
    target_active = np.asarray(history.get("target_active", []), dtype=bool)
    delta_v = np.asarray(history.get("delta_v", []), dtype=np.float32)
    selected_targets = np.asarray(history.get("selected_targets", []), dtype=np.int16)
    closing_delta = np.asarray(history.get("closing_delta", []), dtype=np.float32)
    capture_matrix = np.asarray(history.get("capture_matrix", []), dtype=np.int8)

    total_targets = int(env_config.get("num_targets", 0))
    initial_total_fuel = float(agent_fuel[0].sum()) if agent_fuel.size else 0.0
    final_total_fuel = float(agent_fuel[-1].sum()) if agent_fuel.size else 0.0
    fuel_used = max(0.0, initial_total_fuel - final_total_fuel)
    final_active_targets = int(target_active[-1].sum()) if target_active.size else total_targets
    captured_targets = max(0, total_targets - final_active_targets)
    capture_accuracy_pct = 100.0 * captured_targets / max(1, total_targets)
    total_delta_v = float(delta_v.sum()) if delta_v.size else 0.0
    mean_final_fuel_pct = 100.0 * float(agent_fuel[-1].mean()) if agent_fuel.size else 0.0
    fuel_used_pct = 100.0 * fuel_used / max(initial_total_fuel, 1e-6)
    avg_delta_v_per_capture = total_delta_v / max(1, captured_targets)

    engagement_mask = selected_targets >= 0
    guidance_success_mask = np.logical_and(engagement_mask, closing_delta > 0.0)
    total_engagements = int(engagement_mask.sum()) if engagement_mask.size else 0
    successful_guidance_events = int(guidance_success_mask.sum()) if guidance_success_mask.size else 0
    guidance_accuracy_pct = 100.0 * successful_guidance_events / max(1, total_engagements)

    coordination_samples: List[float] = []
    if selected_targets.size:
        for target_row in selected_targets:
            valid_targets = target_row[target_row >= 0]
            if valid_targets.size == 0:
                continue
            coordination_samples.append(len(np.unique(valid_targets)) / len(valid_targets))
    coordination_pct = 100.0 * float(np.mean(coordination_samples)) if coordination_samples else 0.0

    mean_step_delta_v = float(delta_v.sum(axis=1).mean()) if delta_v.size else 0.0
    capture_steps = compute_capture_steps(target_active)

    if captured_targets >= total_targets and total_targets > 0:
        mission_status = "All debris captured"
    elif result["step_count"] >= int(env_config.get("max_episode_steps", 0)):
        mission_status = "Timed out before full sweep"
    elif final_total_fuel <= 1e-6:
        mission_status = "Fuel depleted before sweep completion"
    else:
        mission_status = "Partial completion"

    return {
        "mission_status": mission_status,
        "captured_targets": captured_targets,
        "targets_remaining": final_active_targets,
        "capture_accuracy_pct": capture_accuracy_pct,
        "guidance_accuracy_pct": guidance_accuracy_pct,
        "coordination_pct": coordination_pct,
        "total_delta_v": total_delta_v,
        "mean_step_delta_v": mean_step_delta_v,
        "avg_delta_v_per_capture": avg_delta_v_per_capture,
        "fuel_used": fuel_used,
        "fuel_used_pct": fuel_used_pct,
        "mean_final_fuel_pct": mean_final_fuel_pct,
        "total_engagements": total_engagements,
        "successful_guidance_events": successful_guidance_events,
        "capture_steps": capture_steps,
    }


def build_mission_brief(metrics: Mapping[str, Any]) -> str:
    capture_accuracy = float(metrics["capture_accuracy_pct"])
    guidance_accuracy = float(metrics["guidance_accuracy_pct"])
    coordination_pct = float(metrics["coordination_pct"])

    if capture_accuracy >= 100.0 and guidance_accuracy >= 70.0:
        return (
            "The fleet executed a clean sweep. Guidance was consistently closing range, "
            "and target allocation stayed coordinated throughout the episode."
        )
    if capture_accuracy >= 60.0 and coordination_pct >= 60.0:
        return (
            "The swarm is coordinating well but still leaves debris on orbit. The next "
            "improvement lever is usually more training or a larger fuel budget."
        )
    if guidance_accuracy < 50.0:
        return (
            "The controller is spending too many steps without reducing intercept distance. "
            "This usually means the policy is under-trained or the maneuver costs are too punitive."
        )
    return (
        "The mission is partially effective: some captures are happening, but fleet efficiency "
        "and completion rate still have room to improve."
    )


def create_trajectory_figure(result: Mapping[str, Any], playback_step: int) -> go.Figure:
    history = result["history"]
    agent_positions = np.asarray(history["agent_positions"], dtype=np.float32)
    target_positions = np.asarray(history["target_positions"], dtype=np.float32)
    target_active = np.asarray(history["target_active"], dtype=bool)
    selected_targets = np.asarray(history.get("selected_targets", []), dtype=np.int16)

    colors = ["#0f766e", "#ea580c", "#2563eb", "#be123c", "#7c3aed", "#0891b2", "#ca8a04", "#1d4ed8"]
    figure = go.Figure()

    for agent_index in range(agent_positions.shape[1]):
        trajectory = agent_positions[: playback_step + 1, agent_index, :]
        color = colors[agent_index % len(colors)]
        figure.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode="lines",
                line={"width": 6, "color": color},
                name=f"Agent {agent_index}",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                z=[trajectory[-1, 2]],
                mode="markers",
                marker={"size": 7, "color": color, "symbol": "diamond"},
                name=f"Agent {agent_index} position",
                showlegend=False,
            )
        )

    current_targets = target_positions[playback_step]
    current_active = target_active[playback_step]
    captured_mask = ~current_active

    if np.any(current_active):
        figure.add_trace(
            go.Scatter3d(
                x=current_targets[current_active, 0],
                y=current_targets[current_active, 1],
                z=current_targets[current_active, 2],
                mode="markers",
                marker={"size": 7, "color": "#dc2626", "symbol": "x"},
                name="Active debris",
            )
        )

    if np.any(captured_mask):
        figure.add_trace(
            go.Scatter3d(
                x=current_targets[captured_mask, 0],
                y=current_targets[captured_mask, 1],
                z=current_targets[captured_mask, 2],
                mode="markers",
                marker={"size": 6, "color": "#16a34a", "symbol": "circle"},
                name="Captured debris",
            )
        )

    if selected_targets.size and playback_step > 0 and playback_step - 1 < selected_targets.shape[0]:
        selected_target_row = selected_targets[playback_step - 1]
        current_agent_positions = agent_positions[playback_step]
        for agent_index, target_index in enumerate(selected_target_row):
            if target_index < 0 or target_index >= current_targets.shape[0]:
                continue
            target_position = current_targets[target_index]
            agent_position = current_agent_positions[agent_index]
            figure.add_trace(
                go.Scatter3d(
                    x=[agent_position[0], target_position[0]],
                    y=[agent_position[1], target_position[1]],
                    z=[agent_position[2], target_position[2]],
                    mode="lines",
                    line={"width": 2, "color": "rgba(100,116,139,0.6)", "dash": "dot"},
                    name=f"Agent {agent_index} target link",
                    showlegend=False,
                )
            )

    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={"size": 9, "color": "#111827", "symbol": "diamond"},
            name="Deployment orbit",
        )
    )

    figure.update_layout(
        height=660,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=f"Swarm and Debris Geometry at Step {playback_step}",
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
        scene={
            "xaxis_title": "Relative X (km)",
            "yaxis_title": "Relative Y (km)",
            "zaxis_title": "Relative Z (km)",
            "bgcolor": "rgba(248,250,252,1)",
        },
    )
    return figure


def create_progress_figure(result: Mapping[str, Any]) -> go.Figure:
    history = result["history"]
    target_active = np.asarray(history.get("target_active", []), dtype=np.int8)
    agent_fuel = np.asarray(history.get("agent_fuel", []), dtype=np.float32)
    delta_v = np.asarray(history.get("delta_v", []), dtype=np.float32)
    capture_matrix = np.asarray(history.get("capture_matrix", []), dtype=np.int8)

    progress_steps = list(range(target_active.shape[0])) if target_active.size else [0]
    active_targets = target_active.sum(axis=1) if target_active.size else np.asarray([0])
    mean_fuel = agent_fuel.mean(axis=1) if agent_fuel.size else np.asarray([0.0])
    capture_cumulative = capture_matrix.sum(axis=1).cumsum() if capture_matrix.size else np.asarray([0.0])
    delta_v_per_step = delta_v.sum(axis=1) if delta_v.size else np.asarray([0.0])
    control_steps = list(range(1, len(delta_v_per_step) + 1))

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.18,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("Mission Completion", "Fuel and Maneuver Cost"),
    )

    figure.add_trace(
        go.Scatter(
            x=progress_steps,
            y=active_targets,
            mode="lines+markers",
            name="Active targets",
            line={"width": 3, "color": "#dc2626"},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=progress_steps[: len(capture_cumulative)],
            y=capture_cumulative,
            mode="lines+markers",
            name="Captures completed",
            line={"width": 3, "color": "#16a34a"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=progress_steps[: len(mean_fuel)],
            y=mean_fuel,
            mode="lines+markers",
            name="Mean fuel fraction",
            line={"width": 3, "color": "#0f766e"},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=control_steps,
            y=delta_v_per_step,
            name="Delta-V per step",
            marker={"color": "#2563eb"},
            opacity=0.55,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    figure.update_layout(
        height=500,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        barmode="overlay",
    )
    figure.update_xaxes(title_text="Observation step", row=1, col=1)
    figure.update_xaxes(title_text="Control step", row=2, col=1)
    figure.update_yaxes(title_text="Targets remaining", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Captured targets", row=1, col=1, secondary_y=True)
    figure.update_yaxes(title_text="Mean fuel fraction", row=2, col=1, secondary_y=False, range=[0, 1.05])
    figure.update_yaxes(title_text="Delta-V", row=2, col=1, secondary_y=True)
    return figure


def create_fleet_efficiency_figure(result: Mapping[str, Any]) -> go.Figure:
    history = result["history"]
    agent_fuel = np.asarray(history.get("agent_fuel", []), dtype=np.float32)
    delta_v = np.asarray(history.get("delta_v", []), dtype=np.float32)
    capture_matrix = np.asarray(history.get("capture_matrix", []), dtype=np.int8)
    step_rewards = np.asarray(history.get("step_rewards", []), dtype=np.float32)

    if agent_fuel.ndim < 2:
        agent_fuel = np.zeros((1, int(result["env_config"]["num_agents"])), dtype=np.float32)
    if delta_v.ndim < 2:
        delta_v = np.zeros((0, agent_fuel.shape[1]), dtype=np.float32)
    if capture_matrix.ndim < 2:
        capture_matrix = np.zeros((0, agent_fuel.shape[1]), dtype=np.int8)
    if step_rewards.ndim < 2:
        step_rewards = np.zeros((0, agent_fuel.shape[1]), dtype=np.float32)

    figure = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.18,
        specs=[[{"type": "scatter"}], [{"type": "bar"}]],
        subplot_titles=("Fuel Fraction by Agent", "Per-Agent Efficiency"),
    )

    colors = ["#0f766e", "#ea580c", "#2563eb", "#be123c", "#7c3aed", "#0891b2", "#ca8a04", "#1d4ed8"]
    steps = list(range(agent_fuel.shape[0]))
    for agent_index in range(agent_fuel.shape[1]):
        color = colors[agent_index % len(colors)]
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=agent_fuel[:, agent_index],
                mode="lines+markers",
                name=f"Agent {agent_index} fuel",
                line={"width": 3, "color": color},
            ),
            row=1,
            col=1,
        )

    agent_ids = [f"agent_{idx}" for idx in range(agent_fuel.shape[1])]
    total_delta_v_by_agent = delta_v.sum(axis=0) if delta_v.size else np.zeros((agent_fuel.shape[1],))
    captures_by_agent = capture_matrix.sum(axis=0) if capture_matrix.size else np.zeros((agent_fuel.shape[1],))
    reward_by_agent = step_rewards.sum(axis=0) if step_rewards.size else np.zeros((agent_fuel.shape[1],))

    figure.add_trace(
        go.Bar(x=agent_ids, y=total_delta_v_by_agent, name="Total Delta-V", marker={"color": "#2563eb"}),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=agent_ids, y=reward_by_agent, name="Total reward", marker={"color": "#0f766e"}),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=agent_ids, y=captures_by_agent, name="Captures", marker={"color": "#16a34a"}),
        row=2,
        col=1,
    )

    figure.update_layout(
        height=500,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        barmode="group",
    )
    figure.update_xaxes(title_text="Observation step", row=1, col=1)
    figure.update_xaxes(title_text="Agent", row=2, col=1)
    figure.update_yaxes(title_text="Fuel fraction", row=1, col=1, range=[0, 1.05])
    figure.update_yaxes(title_text="Per-agent totals", row=2, col=1)
    return figure


def build_agent_summary_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    history = result["history"]
    agent_positions = np.asarray(history["agent_positions"], dtype=np.float32)[-1]
    agent_fuel = np.asarray(history.get("agent_fuel", []), dtype=np.float32)
    delta_v = np.asarray(history.get("delta_v", []), dtype=np.float32)
    capture_matrix = np.asarray(history.get("capture_matrix", []), dtype=np.int8)
    closing_delta = np.asarray(history.get("closing_delta", []), dtype=np.float32)
    selected_targets = np.asarray(history.get("selected_targets", []), dtype=np.int16)
    reward_totals = result["agent_reward_totals"]

    if agent_fuel.ndim < 2:
        agent_fuel = np.zeros((1, agent_positions.shape[0]), dtype=np.float32)
    if delta_v.ndim < 2:
        delta_v = np.zeros((0, agent_positions.shape[0]), dtype=np.float32)
    if capture_matrix.ndim < 2:
        capture_matrix = np.zeros((0, agent_positions.shape[0]), dtype=np.int8)
    if closing_delta.ndim < 2:
        closing_delta = np.zeros((0, agent_positions.shape[0]), dtype=np.float32)
    if selected_targets.ndim < 2:
        selected_targets = np.full((0, agent_positions.shape[0]), -1, dtype=np.int16)

    rows: List[Dict[str, Any]] = []
    for agent_index in range(agent_positions.shape[0]):
        agent_id = f"agent_{agent_index}"
        initial_fuel = float(agent_fuel[0, agent_index])
        final_fuel = float(agent_fuel[-1, agent_index])
        fuel_used = max(0.0, initial_fuel - final_fuel)
        engagement_mask = selected_targets[:, agent_index] >= 0 if selected_targets.size else np.asarray([], dtype=bool)
        closing_success = (
            np.logical_and(engagement_mask, closing_delta[:, agent_index] > 0.0)
            if closing_delta.size
            else np.asarray([], dtype=bool)
        )
        guidance_accuracy_pct = (
            100.0 * float(closing_success.sum()) / max(1, int(engagement_mask.sum()))
            if engagement_mask.size
            else 0.0
        )

        rows.append(
            {
                "agent_id": agent_id,
                "fuel_remaining_pct": round(100.0 * final_fuel, 2),
                "fuel_used_pct": round(100.0 * fuel_used / max(initial_fuel, 1e-6), 2),
                "delta_v_total": round(float(delta_v[:, agent_index].sum()) if delta_v.size else 0.0, 3),
                "captures": int(capture_matrix[:, agent_index].sum()) if capture_matrix.size else 0,
                "guidance_accuracy_pct": round(guidance_accuracy_pct, 2),
                "total_reward": round(float(reward_totals.get(agent_id, 0.0)), 3),
                "final_x_km": round(float(agent_positions[agent_index, 0]), 3),
                "final_y_km": round(float(agent_positions[agent_index, 1]), 3),
                "final_z_km": round(float(agent_positions[agent_index, 2]), 3),
            }
        )
    return rows


def build_target_summary_rows(result: Mapping[str, Any], playback_step: int) -> List[Dict[str, Any]]:
    history = result["history"]
    target_positions = np.asarray(history["target_positions"], dtype=np.float32)
    target_active = np.asarray(history["target_active"], dtype=bool)
    selected_targets = np.asarray(history.get("selected_targets", []), dtype=np.int16)
    agent_positions = np.asarray(history["agent_positions"], dtype=np.float32)
    capture_steps = compute_capture_steps(target_active)

    current_targets = target_positions[playback_step]
    current_active = target_active[playback_step]
    current_agent_positions = agent_positions[playback_step]

    rows: List[Dict[str, Any]] = []
    for target_index in range(current_targets.shape[0]):
        assignments = int(np.count_nonzero(selected_targets == target_index)) if selected_targets.size else 0
        distances = np.linalg.norm(current_agent_positions - current_targets[target_index], axis=1)
        nearest_agent_distance = float(distances.min()) if distances.size else 0.0
        rows.append(
            {
                "target_id": f"target_{target_index}",
                "status": "active" if current_active[target_index] else "captured",
                "assignments": assignments,
                "capture_step": capture_steps.get(target_index, None),
                "nearest_agent_distance_km": round(nearest_agent_distance, 3),
                "x_km": round(float(current_targets[target_index, 0]), 3),
                "y_km": round(float(current_targets[target_index, 1]), 3),
                "z_km": round(float(current_targets[target_index, 2]), 3),
            }
        )
    return rows


def render_sidebar() -> Tuple[Dict[str, Any], str, str, bool]:
    manifest = read_manifest(str(DEFAULT_MANIFEST_PATH)) if DEFAULT_MANIFEST_PATH.exists() else {}
    manifest_topology = str(manifest.get("training_topology", "")).strip()
    default_checkpoint = (
        normalize_checkpoint_path(str(manifest.get("best_checkpoint", "")))
        if manifest_topology == EXPECTED_TRAINING_TOPOLOGY
        else ""
    )
    default_debris_cache = str(manifest.get("debris_cache", "")) if manifest.get("debris_cache") else ""

    with st.sidebar:
        st.title("Mission Controls")
        st.caption("Tune the swarm, then watch how the controller adapts.")
        if manifest and manifest_topology != EXPECTED_TRAINING_TOPOLOGY:
            st.info(
                "A legacy training manifest was found in the default outputs folder. "
                "Train a fresh decentralized checkpoint to enable RL Checkpoint mode."
            )

        controller_mode = st.radio(
            "Controller",
            options=["Cooperative Autopilot", "RL Checkpoint"],
            index=0,
            help="Cooperative Autopilot is a built-in coordinated heuristic. RL Checkpoint runs the trained decentralized policy.",
        )
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value=default_checkpoint,
            disabled=controller_mode != "RL Checkpoint",
            help="Path to a saved decentralized RLlib checkpoint. The latest compatible training manifest is used as the default when available.",
        )
        debris_cache_path = st.text_input(
            "Debris cache path",
            value=default_debris_cache,
            help="Optional JSON/CSV created by fetch_data.py. Leave blank to use synthetic clustered debris.",
        )

        with st.expander("Fleet", expanded=True):
            num_agents = st.slider(
                "Agents",
                1,
                8,
                DEMO_DEFAULTS["num_agents"],
                help="Number of removal spacecraft in the swarm.",
            )
            num_targets = st.slider(
                "Targets",
                1,
                12,
                DEMO_DEFAULTS["num_targets"],
                help="Number of debris objects active in the episode.",
            )
            max_targets = st.slider(
                "Max target slots",
                num_targets,
                12,
                max(num_targets, DEMO_DEFAULTS["max_targets"]),
            )
            max_episode_steps = st.slider(
                "Episode steps",
                25,
                500,
                DEMO_DEFAULTS["max_episode_steps"],
                step=5,
            )
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                value=DEMO_DEFAULTS["random_seed"],
                step=1,
            )

        with st.expander("Orbital Geometry", expanded=True):
            world_range_km = st.slider(
                "World range (km)",
                50.0,
                800.0,
                DEMO_DEFAULTS["world_range_km"],
                step=10.0,
            )
            capture_radius_km = st.slider(
                "Capture radius (km)",
                1.0,
                25.0,
                DEMO_DEFAULTS["capture_radius_km"],
                step=0.5,
            )
            translation_step_km = st.slider(
                "Translation step (km)",
                0.0,
                40.0,
                DEMO_DEFAULTS["translation_step_km"],
                step=0.5,
            )
            agent_spawn_radius_km = st.slider(
                "Agent spawn radius (km)",
                0.0,
                60.0,
                DEMO_DEFAULTS["agent_spawn_radius_km"],
                step=1.0,
            )
            target_position_noise_km = st.slider(
                "Target noise (km)",
                0.0,
                20.0,
                DEMO_DEFAULTS["target_position_noise_km"],
                step=0.5,
            )

        with st.expander("Fuel and Reward", expanded=False):
            initial_fuel = st.slider(
                "Initial fuel fraction",
                0.1,
                1.0,
                DEMO_DEFAULTS["initial_fuel"],
                step=0.05,
            )
            fuel_capacity_delta_v = st.slider(
                "Fuel capacity Delta-V",
                10.0,
                400.0,
                DEMO_DEFAULTS["fuel_capacity_delta_v"],
                step=5.0,
            )
            k1_phasing_cost = st.slider(
                "k1 phasing cost",
                0.0,
                0.2,
                DEMO_DEFAULTS["k1_phasing_cost"],
                step=0.005,
            )
            k2_plane_change_cost = st.slider(
                "k2 plane change cost",
                0.0,
                0.3,
                DEMO_DEFAULTS["k2_plane_change_cost"],
                step=0.005,
            )
            capture_reward = st.slider(
                "Capture reward",
                1.0,
                50.0,
                DEMO_DEFAULTS["capture_reward"],
                step=1.0,
            )
            cooperative_reward_share = st.slider(
                "Cooperative reward share",
                0.0,
                20.0,
                DEMO_DEFAULTS["cooperative_reward_share"],
                step=0.5,
            )
            progress_reward_scale = st.slider(
                "Progress reward scale",
                0.0,
                10.0,
                DEMO_DEFAULTS["progress_reward_scale"],
                step=0.25,
            )
            fuel_penalty_scale = st.slider(
                "Fuel penalty scale",
                0.0,
                2.0,
                DEMO_DEFAULTS["fuel_penalty_scale"],
                step=0.05,
            )
            idle_penalty = st.slider(
                "Idle penalty",
                0.0,
                1.0,
                DEMO_DEFAULTS["idle_penalty"],
                step=0.01,
            )
            invalid_action_penalty = st.slider(
                "Invalid action penalty",
                0.0,
                1.0,
                DEMO_DEFAULTS["invalid_action_penalty"],
                step=0.05,
            )

        run_clicked = st.button("Run Mission Analysis", type="primary", use_container_width=True)

    env_config = build_env_config(
        num_agents=int(num_agents),
        num_targets=int(num_targets),
        max_targets=int(max_targets),
        max_episode_steps=int(max_episode_steps),
        world_range_km=float(world_range_km),
        capture_radius_km=float(capture_radius_km),
        translation_step_km=float(translation_step_km),
        agent_spawn_radius_km=float(agent_spawn_radius_km),
        target_position_noise_km=float(target_position_noise_km),
        initial_fuel=float(initial_fuel),
        fuel_capacity_delta_v=float(fuel_capacity_delta_v),
        k1_phasing_cost=float(k1_phasing_cost),
        k2_plane_change_cost=float(k2_plane_change_cost),
        capture_reward=float(capture_reward),
        cooperative_reward_share=float(cooperative_reward_share),
        progress_reward_scale=float(progress_reward_scale),
        fuel_penalty_scale=float(fuel_penalty_scale),
        idle_penalty=float(idle_penalty),
        invalid_action_penalty=float(invalid_action_penalty),
        random_seed=int(random_seed),
        debris_cache_path=debris_cache_path,
    )
    return env_config, controller_mode, checkpoint_path, run_clicked


def render_metric_rows(result: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    metric_row_1 = st.columns(5)
    metric_row_1[0].metric("Controller", result["controller_mode"])
    metric_row_1[1].metric("Mission status", metrics["mission_status"])
    metric_row_1[2].metric("Capture accuracy", f"{metrics['capture_accuracy_pct']:.1f}%")
    metric_row_1[3].metric("Guidance accuracy", f"{metrics['guidance_accuracy_pct']:.1f}%")
    metric_row_1[4].metric("Coordination", f"{metrics['coordination_pct']:.1f}%")

    metric_row_2 = st.columns(5)
    metric_row_2[0].metric("Episode reward", f"{result['total_reward']:.2f}")
    metric_row_2[1].metric("Captures", f"{metrics['captured_targets']} / {result['env_config']['num_targets']}")
    metric_row_2[2].metric("Fuel used", f"{metrics['fuel_used_pct']:.1f}%")
    metric_row_2[3].metric("Mean final fuel", f"{metrics['mean_final_fuel_pct']:.1f}%")
    metric_row_2[4].metric("Total Delta-V", f"{metrics['total_delta_v']:.2f}")


def main() -> None:
    apply_dashboard_theme()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Orbital Debris MARL Mission Dashboard</div>
                <div class="hero-copy">
                    Tune fleet size, Delta-V economics, capture geometry, and reward shaping in one place.
                    Run the built-in Cooperative Autopilot immediately, or switch to a trained RL checkpoint
                    to compare how a decentralized learned swarm behaves under the same orbital conditions.
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    env_config, controller_mode, checkpoint_path, run_clicked = render_sidebar()

    if "dashboard_result" not in st.session_state:
        st.session_state.dashboard_result = None

    if run_clicked:
        if controller_mode == "RL Checkpoint" and not normalize_checkpoint_path(checkpoint_path):
            st.error("RL Checkpoint mode is selected, but no checkpoint path was provided.")
        else:
            try:
                with st.spinner("Simulating orbital mission..."):
                    st.session_state.dashboard_result = run_simulation(
                        controller_mode=controller_mode,
                        checkpoint_path=checkpoint_path,
                        env_config=env_config,
                        seed=int(env_config["random_seed"]),
                    )
            except Exception as exc:  # noqa: BLE001 - surface full dashboard errors.
                st.exception(exc)

    result = st.session_state.dashboard_result
    if result is None:
        st.info(
            "Set your scenario in the sidebar and press 'Run Mission Analysis' to generate the interactive dashboard."
        )
        st.code("streamlit run dashboard.py", language="bash")
        return

    metrics = summarize_mission(result)
    history = result["history"]
    max_playback_step = int(np.asarray(history["agent_positions"]).shape[0] - 1)
    playback_step = st.slider("Playback step", 0, max_playback_step, max_playback_step)

    render_metric_rows(result, metrics)

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">Mission Brief</div>
            <div class="summary-copy">{build_mission_brief(metrics)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, analytics_tab, tables_tab, config_tab = st.tabs(
        ["Mission Overview", "Performance Analytics", "Fleet Tables", "Configuration"]
    )

    with overview_tab:
        left_col, right_col = st.columns([1.75, 1.0])
        with left_col:
            st.plotly_chart(
                create_trajectory_figure(result, playback_step),
                use_container_width=True,
            )
        with right_col:
            st.markdown("#### Mission Scorecard")
            st.dataframe(
                [
                    {"metric": "Targets remaining", "value": metrics["targets_remaining"]},
                    {"metric": "Successful guidance events", "value": metrics["successful_guidance_events"]},
                    {"metric": "Total engagements", "value": metrics["total_engagements"]},
                    {"metric": "Average Delta-V per capture", "value": round(float(metrics["avg_delta_v_per_capture"]), 3)},
                    {"metric": "Mean Delta-V per control step", "value": round(float(metrics["mean_step_delta_v"]), 3)},
                    {"metric": "Checkpoint", "value": result["checkpoint_path"] or "Not applicable"},
                ],
                hide_index=True,
                use_container_width=True,
            )

    with analytics_tab:
        analytics_left, analytics_right = st.columns(2)
        with analytics_left:
            st.plotly_chart(create_progress_figure(result), use_container_width=True)
        with analytics_right:
            st.plotly_chart(create_fleet_efficiency_figure(result), use_container_width=True)

    with tables_tab:
        table_left, table_right = st.columns(2)
        with table_left:
            st.subheader("Agent Summary")
            st.dataframe(build_agent_summary_rows(result), use_container_width=True, hide_index=True)
        with table_right:
            st.subheader("Target Summary")
            st.dataframe(
                build_target_summary_rows(result, playback_step),
                use_container_width=True,
                hide_index=True,
            )

    with config_tab:
        st.subheader("Environment Configuration")
        st.json(result["env_config"])

        st.subheader("Derived Mission Metrics")
        st.json(
            {
                "mission_status": metrics["mission_status"],
                "capture_accuracy_pct": round(float(metrics["capture_accuracy_pct"]), 3),
                "guidance_accuracy_pct": round(float(metrics["guidance_accuracy_pct"]), 3),
                "coordination_pct": round(float(metrics["coordination_pct"]), 3),
                "fuel_used_pct": round(float(metrics["fuel_used_pct"]), 3),
                "mean_final_fuel_pct": round(float(metrics["mean_final_fuel_pct"]), 3),
                "total_delta_v": round(float(metrics["total_delta_v"]), 3),
                "avg_delta_v_per_capture": round(float(metrics["avg_delta_v_per_capture"]), 3),
            }
        )


if __name__ == "__main__":
    main()
