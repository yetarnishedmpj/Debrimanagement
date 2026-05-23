"""
RLlib training entry point for the orbital debris MARL prototype.

This version uses a fully decentralized multi-agent setup:
- each satellite receives only its own local observation matrix
- each satellite trains its own PPO policy (independent PPO / IPPO)
- no centralized critic or fleet-level global state is exposed

That makes the resulting checkpoint consistent with decentralized execution and
also keeps the architecture easier to reason about during experimentation.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from fetch_data import load_cached_records, records_to_relative_positions
from orbital_env import OrbitalDebrisRemovalEnv

LOGGER = logging.getLogger(__name__)
ENV_NAME = "orbital_debris_marl_env"
TRAINING_TOPOLOGY = "decentralized_ippo"
POLICY_ID_PREFIX = "policy_agent"
DEFAULT_CAPTURE_RADIUS_KM = 12.0
DEFAULT_TRANSLATION_STEP_KM = 18.0
DEFAULT_FUEL_CAPACITY_DELTA_V = 300.0
DEFAULT_AGENT_SPAWN_RADIUS_KM = 15.0
DEFAULT_TARGET_POSITION_NOISE_KM = 4.0

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    from ray.tune.registry import register_env
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency gate.
    raise RuntimeError(
        "train_marl.py requires Ray RLlib and PyTorch. Install dependencies from "
        "requirements.txt before running training."
    ) from exc


def agent_index_from_id(agent_id: str | int) -> int:
    """
    Normalize `agent_3` or `3` into a numeric agent index.

    RLlib passes string agent IDs through the policy mapping function, while the
    dashboard and visualization code sometimes need the same mapping logic.
    """

    if isinstance(agent_id, int):
        return max(0, int(agent_id))

    text = str(agent_id).strip()
    if text.startswith("agent_"):
        text = text.split("_", maxsplit=1)[1]

    try:
        return max(0, int(text))
    except ValueError as exc:
        raise ValueError(f"Unable to extract an agent index from '{agent_id}'.") from exc


def policy_id_for_agent(agent_id: str | int) -> str:
    """Return the decentralized PPO policy ID assigned to one agent."""

    return f"{POLICY_ID_PREFIX}_{agent_index_from_id(agent_id)}"


def build_policy_specs(
    *,
    num_agents: int,
    observation_space,
    action_space,
) -> Dict[str, PolicySpec]:
    """Create one PPO policy per satellite for independent decentralized learning."""

    return {
        policy_id_for_agent(agent_index): PolicySpec(
            observation_space=observation_space,
            action_space=action_space,
            config={},
        )
        for agent_index in range(num_agents)
    }


def policy_mapping_fn(agent_id: str, *args, **kwargs) -> str:
    """Map each satellite to its own independent decentralized policy."""

    return policy_id_for_agent(agent_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a decentralized orbital debris IPPO policy."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stop-iters", type=int, default=20)
    parser.add_argument("--stop-timesteps", type=int, default=150000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-targets", type=int, default=5)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--world-range-km", type=float, default=250.0)
    parser.add_argument("--capture-radius-km", type=float, default=DEFAULT_CAPTURE_RADIUS_KM)
    parser.add_argument(
        "--translation-step-km",
        type=float,
        default=DEFAULT_TRANSLATION_STEP_KM,
    )
    parser.add_argument(
        "--fuel-capacity-delta-v",
        type=float,
        default=DEFAULT_FUEL_CAPACITY_DELTA_V,
    )
    parser.add_argument(
        "--agent-spawn-radius-km",
        type=float,
        default=DEFAULT_AGENT_SPAWN_RADIUS_KM,
    )
    parser.add_argument(
        "--target-position-noise-km",
        type=float,
        default=DEFAULT_TARGET_POSITION_NOISE_KM,
    )
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument(
        "--minibatch-size",
        "--sgd-minibatch-size",
        dest="minibatch_size",
        type=int,
        default=512,
        help="Learner minibatch size. `--sgd-minibatch-size` is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--num-epochs",
        "--num-sgd-iter",
        dest="num_epochs",
        type=int,
        default=10,
        help="Number of PPO epochs per training batch. `--num-sgd-iter` is kept as a compatibility alias.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional decentralized RLlib checkpoint to restore before continuing training.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore any compatible checkpoint discovered in the output directory and train from scratch.",
    )
    parser.add_argument(
        "--debris-cache",
        type=Path,
        default=None,
        help="Optional cached JSON/CSV generated by fetch_data.py.",
    )
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> Dict[str, Any]:
    env_config: Dict[str, Any] = {
        "num_agents": args.num_agents,
        "num_targets": args.num_targets,
        "max_targets": max(args.max_targets, args.num_targets),
        "max_episode_steps": args.max_episode_steps,
        "world_range_km": args.world_range_km,
        "capture_radius_km": args.capture_radius_km,
        "translation_step_km": args.translation_step_km,
        "fuel_capacity_delta_v": args.fuel_capacity_delta_v,
        "agent_spawn_radius_km": args.agent_spawn_radius_km,
        "target_position_noise_km": args.target_position_noise_km,
        "random_seed": args.seed,
    }

    if args.debris_cache:
        try:
            records = load_cached_records(args.debris_cache.resolve())
            target_positions = records_to_relative_positions(
                records,
                world_range_km=args.world_range_km,
                max_targets=env_config["max_targets"],
            )
            if target_positions:
                env_config["target_positions"] = target_positions
                env_config["num_targets"] = min(len(target_positions), env_config["max_targets"])
                LOGGER.info(
                    "Loaded %d target seeds from %s",
                    len(target_positions),
                    args.debris_cache,
                )
        except Exception as exc:  # noqa: BLE001 - keep training usable even if cache is bad.
            LOGGER.warning("Unable to load debris cache %s: %s", args.debris_cache, exc)

    return env_config


def configure_parallelism(config: PPOConfig, args: argparse.Namespace) -> PPOConfig:
    """Apply rollout parallelism with compatibility across RLlib versions."""

    if hasattr(config, "env_runners"):
        try:
            return config.env_runners(
                num_env_runners=args.num_workers,
                num_envs_per_env_runner=args.num_envs_per_worker,
            )
        except TypeError:
            pass

    if hasattr(config, "rollouts"):
        return config.rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
        )

    return config


def extract_metric(result: Mapping[str, Any], *paths: str) -> Optional[float]:
    """Return the first metric found across a list of dotted result paths."""

    for path in paths:
        current: Any = result
        found = True
        for token in path.split("."):
            if isinstance(current, Mapping) and token in current:
                current = current[token]
            else:
                found = False
                break
        if found and current is not None:
            return float(current)
    return None


def normalize_checkpoint_path(raw_checkpoint_path: Optional[str]) -> Optional[str]:
    if raw_checkpoint_path is None:
        return None
    text = str(raw_checkpoint_path).strip()
    if not text:
        return None
    if text.startswith("Checkpoint(") and "path=" in text:
        extracted = text.split("path=", maxsplit=1)[1].rstrip(")")
        return extracted.strip()
    return text


def load_manifest_if_present(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Unable to read manifest %s: %s", manifest_path, exc)
        return {}


def find_manifest_for_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    for parent in checkpoint_path.resolve().parents:
        candidate = parent / "training_manifest.json"
        if candidate.exists():
            return candidate
    return None


def describe_manifest_compatibility(
    manifest: Mapping[str, Any],
    env_config: Mapping[str, Any],
) -> Optional[str]:
    if not manifest:
        return None

    topology = str(manifest.get("training_topology", "")).strip()
    if topology != TRAINING_TOPOLOGY:
        if not topology:
            return (
                "the existing manifest predates the decentralized rewrite and most likely "
                "points to an old CTDE checkpoint"
            )
        return f"the existing manifest was trained with '{topology}', not '{TRAINING_TOPOLOGY}'"

    previous_env = manifest.get("env_config", {})
    mismatches = []
    for field_name in ("num_agents", "max_targets"):
        if field_name in previous_env and previous_env[field_name] != env_config.get(field_name):
            mismatches.append(
                f"{field_name}: previous={previous_env[field_name]} current={env_config.get(field_name)}"
            )

    if mismatches:
        return "the saved checkpoint is incompatible with the current observation/action layout (" + ", ".join(
            mismatches
        ) + ")"

    policy_ids = manifest.get("policy_ids")
    if isinstance(policy_ids, list) and len(policy_ids) != int(env_config.get("num_agents", 0)):
        return (
            "the saved checkpoint contains a different number of decentralized policies "
            f"({len(policy_ids)} vs {env_config.get('num_agents')})"
        )

    return None


def validate_resume_checkpoint(
    checkpoint_path: Path,
    *,
    env_config: Mapping[str, Any],
    strict: bool,
) -> Optional[Path]:
    manifest_path = find_manifest_for_checkpoint(checkpoint_path)
    manifest = load_manifest_if_present(manifest_path) if manifest_path else {}
    compatibility_issue = describe_manifest_compatibility(manifest, env_config)
    if compatibility_issue is None:
        return checkpoint_path

    if strict:
        raise ValueError(
            f"Checkpoint {checkpoint_path} cannot be resumed: {compatibility_issue}."
        )

    LOGGER.info("Skipping auto-resume from %s because %s.", checkpoint_path, compatibility_issue)
    return None


def resolve_resume_checkpoint(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    env_config: Mapping[str, Any],
) -> Optional[Path]:
    if args.fresh_start:
        return None

    if args.resume_from_checkpoint:
        candidate = args.resume_from_checkpoint.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {candidate}")
        return validate_resume_checkpoint(candidate, env_config=env_config, strict=True)

    manifest = load_manifest_if_present(manifest_path)
    compatibility_issue = describe_manifest_compatibility(manifest, env_config)
    if compatibility_issue is not None:
        if manifest:
            LOGGER.info(
                "Existing manifest in %s is being ignored because %s.",
                manifest_path,
                compatibility_issue,
            )
        return None

    latest_checkpoint = normalize_checkpoint_path(manifest.get("latest_checkpoint"))
    best_checkpoint = normalize_checkpoint_path(manifest.get("best_checkpoint"))

    for raw_path in (latest_checkpoint, best_checkpoint):
        if not raw_path:
            continue
        candidate = Path(raw_path).resolve()
        if candidate.exists():
            return validate_resume_checkpoint(candidate, env_config=env_config, strict=False)

    return None


def save_checkpoint(algo, checkpoint_dir: Path) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_result = algo.save(checkpoint_dir=str(checkpoint_dir))
    checkpoint_handle = getattr(save_result, "checkpoint", save_result)
    resolved_path = getattr(checkpoint_handle, "path", checkpoint_handle)
    return str(resolved_path)


def write_manifest(
    *,
    manifest_path: Path,
    env_config: Mapping[str, Any],
    latest_checkpoint: str,
    best_checkpoint: str,
    policy_ids: list[str],
    args: argparse.Namespace,
    best_reward: Optional[float],
    resumed_from_checkpoint: Optional[str],
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PPO",
        "training_topology": TRAINING_TOPOLOGY,
        "observation_layout": "decentralized_local_target_matrix",
        "policy_ids": list(policy_ids),
        "latest_checkpoint": latest_checkpoint,
        "best_checkpoint": best_checkpoint,
        "best_reward": best_reward,
        "env_config": dict(env_config),
        "debris_cache": str(args.debris_cache.resolve()) if args.debris_cache else None,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "seed": args.seed,
        "resumed_from_checkpoint": resumed_from_checkpoint,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    manifest_path = output_dir / "training_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = build_env_config(args)
    existing_manifest = load_manifest_if_present(manifest_path)
    manifest_issue = describe_manifest_compatibility(existing_manifest, env_config)
    if manifest_issue is not None:
        existing_manifest = {}

    resume_checkpoint = resolve_resume_checkpoint(
        args,
        manifest_path=manifest_path,
        env_config=env_config,
    )

    dummy_env = OrbitalDebrisRemovalEnv(env_config)
    observation_space = dummy_env.observation_space
    action_space = dummy_env.action_space
    dummy_env.close()

    register_env(ENV_NAME, lambda config: OrbitalDebrisRemovalEnv(config))

    num_gpus = 1 if args.use_gpu and torch.cuda.is_available() else 0
    if args.use_gpu and num_gpus == 0:
        LOGGER.warning("GPU was requested but no CUDA device is available. Falling back to CPU.")

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    base_config = PPOConfig()
    if hasattr(base_config, "api_stack"):
        base_config = base_config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )

    policy_specs = build_policy_specs(
        num_agents=args.num_agents,
        observation_space=observation_space,
        action_space=action_space,
    )
    policy_ids = list(policy_specs.keys())

    config = (
        base_config.environment(
            env=ENV_NAME,
            env_config=env_config,
            disable_env_checking=True,
        )
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .training(
            model={
                "fcnet_hiddens": [512, 512, 256],
                "vf_share_layers": False,
            },
            gamma=0.995,
            lr=args.learning_rate,
            lr_schedule=[[0, args.learning_rate], [args.stop_timesteps, 1e-5]],
            lambda_=0.97,
            clip_param=0.2,
            entropy_coeff=0.05,
            entropy_coeff_schedule=[[0, 0.05], [int(args.stop_timesteps * 0.8), 0.005]],
            vf_loss_coeff=1.0,
            grad_clip=0.5,
            use_critic=True,
            use_gae=True,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
        )
        .multi_agent(
            policies=policy_specs,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policy_ids,
        )
        .debugging(seed=args.seed)
    )
    config = configure_parallelism(config, args)

    algo = config.build()
    resumed_from_checkpoint = str(resume_checkpoint) if resume_checkpoint else None
    if resume_checkpoint:
        LOGGER.info("Restoring trainer state from %s", resume_checkpoint)
        algo.restore(str(resume_checkpoint))

    best_reward: Optional[float] = (
        float(existing_manifest["best_reward"])
        if existing_manifest.get("best_reward") is not None
        else None
    )
    best_checkpoint = (
        normalize_checkpoint_path(existing_manifest.get("best_checkpoint"))
        or resumed_from_checkpoint
        or ""
    )
    latest_checkpoint = (
        normalize_checkpoint_path(existing_manifest.get("latest_checkpoint"))
        or resumed_from_checkpoint
        or ""
    )

    try:
        for iteration in range(1, args.stop_iters + 1):
            result = algo.train()
            mean_reward = extract_metric(
                result,
                "env_runners.episode_return_mean",
                "sampler_results.episode_reward_mean",
                "episode_reward_mean",
            )
            timesteps_total = extract_metric(result, "timesteps_total") or 0.0

            summary = {
                "iteration": iteration,
                "mean_reward": mean_reward,
                "timesteps_total": int(timesteps_total),
                "episodes_this_iter": int(
                    extract_metric(result, "episodes_this_iter") or 0.0
                ),
            }
            print(json.dumps(summary))

            if mean_reward is not None and (best_reward is None or mean_reward > best_reward):
                best_reward = mean_reward
                best_checkpoint = save_checkpoint(algo, checkpoint_dir)
                latest_checkpoint = best_checkpoint
                write_manifest(
                    manifest_path=manifest_path,
                    env_config=env_config,
                    latest_checkpoint=latest_checkpoint,
                    best_checkpoint=best_checkpoint,
                    policy_ids=policy_ids,
                    args=args,
                    best_reward=best_reward,
                    resumed_from_checkpoint=resumed_from_checkpoint,
                )
                LOGGER.info(
                    "Saved new best checkpoint at %s with mean reward %.4f",
                    best_checkpoint,
                    best_reward,
                )

            if timesteps_total >= args.stop_timesteps:
                LOGGER.info(
                    "Reached stop_timesteps=%d at iteration %d.",
                    args.stop_timesteps,
                    iteration,
                )
                break

        latest_checkpoint = save_checkpoint(algo, checkpoint_dir)
        if not best_checkpoint:
            best_checkpoint = latest_checkpoint
        write_manifest(
            manifest_path=manifest_path,
            env_config=env_config,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
            policy_ids=policy_ids,
            args=args,
            best_reward=best_reward,
            resumed_from_checkpoint=resumed_from_checkpoint,
        )

        LOGGER.info("Training complete. Latest checkpoint: %s", latest_checkpoint)
        LOGGER.info("Best checkpoint: %s", best_checkpoint)
    finally:
        algo.stop()
        ray.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
