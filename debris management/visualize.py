"""
Post-training visualization for the orbital debris MARL prototype.

The script restores a decentralized RLlib checkpoint, rolls out one evaluation
episode, and plots agent trajectories against the debris coordinates in 3D.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from fetch_data import load_cached_records, records_to_relative_positions
from orbital_env import OrbitalDebrisRemovalEnv
from train_marl import ENV_NAME, TRAINING_TOPOLOGY, policy_id_for_agent

LOGGER = logging.getLogger(__name__)

try:
    import matplotlib
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.registry import register_env
except ImportError as exc:  # pragma: no cover - runtime dependency gate.
    raise RuntimeError(
        "visualize.py requires matplotlib and Ray RLlib. Install requirements.txt first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained decentralized orbital debris policy."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/trajectory_plot.png"))
    parser.add_argument(
        "--debris-cache",
        type=Path,
        default=None,
        help="Optional cache override produced by fetch_data.py.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def find_manifest(checkpoint_path: Path, explicit_manifest: Optional[Path]) -> Optional[Path]:
    if explicit_manifest is not None:
        return explicit_manifest.resolve()

    for parent in checkpoint_path.resolve().parents:
        candidate = parent / "training_manifest.json"
        if candidate.exists():
            return candidate
    return None


def load_manifest(manifest_path: Optional[Path]) -> Dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ensure_manifest_is_decentralized(manifest: Mapping[str, Any]) -> None:
    if not manifest:
        return

    topology = str(manifest.get("training_topology", "")).strip()
    if topology == TRAINING_TOPOLOGY:
        return

    if not topology:
        raise RuntimeError(
            "The checkpoint manifest predates the decentralized rewrite. "
            "Retrain with the current train_marl.py before visualizing."
        )

    raise RuntimeError(
        f"Checkpoint topology '{topology}' is not supported by the decentralized visualizer."
    )


def resolve_env_config(args: argparse.Namespace, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    env_config = dict(manifest.get("env_config", {}))
    if not env_config:
        env_config = {
            "num_agents": 3,
            "num_targets": 5,
            "max_targets": 5,
            "max_episode_steps": 300,
            "world_range_km": 250.0,
            "capture_radius_km": 12.0,
            "translation_step_km": 18.0,
            "fuel_capacity_delta_v": 300.0,
            "random_seed": args.seed,
        }

    cache_candidate = args.debris_cache
    if cache_candidate is None and manifest.get("debris_cache"):
        cache_candidate = Path(str(manifest["debris_cache"]))

    if cache_candidate:
        try:
            records = load_cached_records(cache_candidate.resolve())
            target_positions = records_to_relative_positions(
                records,
                world_range_km=float(env_config.get("world_range_km", 250.0)),
                max_targets=int(env_config.get("max_targets", env_config.get("num_targets", 5))),
            )
            if target_positions:
                env_config["target_positions"] = target_positions
                env_config["num_targets"] = min(
                    len(target_positions),
                    int(env_config.get("max_targets", len(target_positions))),
                )
        except Exception as exc:  # noqa: BLE001 - visualization should still proceed.
            LOGGER.warning("Unable to use debris cache %s: %s", cache_candidate, exc)

    env_config["random_seed"] = args.seed
    return env_config


def restore_algorithm(checkpoint_path: Path) -> Algorithm:
    register_env(ENV_NAME, lambda config: OrbitalDebrisRemovalEnv(config))
    try:
        return Algorithm.from_checkpoint(str(checkpoint_path.resolve()))
    except Exception as exc:  # noqa: BLE001 - wrap with checkpoint-specific guidance.
        raise RuntimeError(
            "Unable to restore the checkpoint. If it was produced before the decentralized "
            "rewrite, retrain the policy with the current train_marl.py."
        ) from exc


def rollout_episode(
    *,
    algo: Algorithm,
    env_config: Mapping[str, Any],
    seed: int,
) -> Tuple[Dict[str, Any], float, int]:
    env = OrbitalDebrisRemovalEnv(env_config)
    observations, _ = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0

    while observations:
        action_dict = {}
        for agent_id, observation in observations.items():
            action = algo.compute_single_action(
                observation,
                policy_id=policy_id_for_agent(agent_id),
                explore=False,
            )
            if isinstance(action, tuple):
                action = action[0]
            action_dict[agent_id] = action

        observations, rewards, terminateds, truncateds, _ = env.step(action_dict)
        total_reward += sum(float(reward) for reward in rewards.values())
        step_count += 1
        if terminateds.get("__all__") or truncateds.get("__all__"):
            break

    history = env.get_episode_history()
    env.close()
    return history, total_reward, step_count


def plot_trajectories(
    *,
    history: Mapping[str, Any],
    output_path: Path,
    show_plot: bool,
) -> None:
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        raise RuntimeError("No trajectory history was recorded for visualization.")

    agent_positions = history["agent_positions"]
    target_positions = history["target_positions"][0]
    final_target_active = history["target_active"][-1].astype(bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")

    for agent_index in range(agent_positions.shape[1]):
        trajectory = agent_positions[:, agent_index, :]
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            linewidth=2.0,
            label=f"Agent {agent_index}",
        )
        axis.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            trajectory[0, 2],
            marker="o",
            s=45,
            alpha=0.9,
        )
        axis.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            trajectory[-1, 2],
            marker="^",
            s=55,
            alpha=0.9,
        )

    captured_mask = ~final_target_active
    if captured_mask.any():
        axis.scatter(
            target_positions[captured_mask, 0],
            target_positions[captured_mask, 1],
            target_positions[captured_mask, 2],
            c="tab:green",
            marker="o",
            s=70,
            label="Captured Debris",
        )

    if final_target_active.any():
        axis.scatter(
            target_positions[final_target_active, 0],
            target_positions[final_target_active, 1],
            target_positions[final_target_active, 2],
            c="tab:red",
            marker="x",
            s=80,
            label="Uncaptured Debris",
        )

    axis.scatter([0.0], [0.0], [0.0], c="black", marker="*", s=120, label="Deployment Orbit")
    axis.set_title("Decentralized Orbital Debris Removal Trajectories")
    axis.set_xlabel("Relative X (km)")
    axis.set_ylabel("Relative Y (km)")
    axis.set_zlabel("Relative Z (km)")
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()
    plt.close(figure)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        LOGGER.error("Checkpoint does not exist: %s", checkpoint_path)
        return 1

    manifest_path = find_manifest(checkpoint_path, args.manifest)
    manifest = load_manifest(manifest_path)
    ensure_manifest_is_decentralized(manifest)
    env_config = resolve_env_config(args, manifest)

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    algo = restore_algorithm(checkpoint_path)

    try:
        history, total_reward, step_count = rollout_episode(
            algo=algo,
            env_config=env_config,
            seed=args.seed,
        )
        plot_trajectories(
            history=history,
            output_path=args.output.resolve(),
            show_plot=args.show,
        )
        LOGGER.info(
            "Saved trajectory plot to %s after %d steps (total reward %.4f).",
            args.output.resolve(),
            step_count,
            total_reward,
        )
    finally:
        algo.stop()
        ray.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
