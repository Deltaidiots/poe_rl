from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from sb3_contrib.ppo_mask import MaskablePPO

from .envs.ring_crafting_env import GoalSpec
from .train_ppo import build_ring_engine, make_env, wrap_with_mask


class EpisodeResult(TypedDict):
    base: str
    reward: float
    steps: int
    success: bool
    actions: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a saved PPO policy on random ring bases")
    parser.add_argument(
        "--model-path",
        default="artifacts/ppo_ring_policy.zip",
        help="Path to the trained policy ZIP produced by MaskablePPO",
    )
    parser.add_argument(
        "--mods",
        default="src/poe_rl/data/static/poec_data.json",
        help="Path to poec_data.json dump used to build the crafting database",
    )
    parser.add_argument("--episodes", type=int, default=10, help="How many evaluation episodes to run")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for base sampling and env")
    parser.add_argument("--max-steps", type=int, default=200, help="Episode step cap inside the env")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="If set, run the policy deterministically (default: stochastic)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="If >0, randomly sample this many unique bases instead of using the full base list",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file with overrides (model_path, goal spec, etc.)",
    )
    parser.add_argument(
        "--store-results",
        action="store_true",
        help="If set, persist evaluation logs and JSON payload alongside the checkpoint run",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to store evaluation artifacts (default: run_root/custom_eval)",
    )
    return parser


def _choose_bases(all_bases: Sequence[str], rng: random.Random, sample_count: int) -> list[str]:
    if sample_count <= 0 or sample_count >= len(all_bases):
        return list(all_bases)
    return rng.sample(list(all_bases), sample_count)


def run_episode(
    model: MaskablePPO,
    env,
    base_id: str,
    *,
    deterministic: bool,
    rng: random.Random,
) -> EpisodeResult:
    base_env = env.unwrapped
    base_env.set_base(base_id)
    obs, info = env.reset(seed=rng.randint(0, 1_000_000))
    done = False
    total_reward = 0.0
    steps = 0
    actions: list[str] = []
    while not done:
        mask = info.get("action_mask")
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        action_name = base_env.get_action_name(int(action))
        actions.append(action_name)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        steps += 1
    success = bool(info.get("success"))
    return {
        "base": base_id,
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "actions": actions,
    }


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation config must be a JSON object")
    return payload


def _resolve_option(
    name: str,
    *,
    args: argparse.Namespace,
    defaults: argparse.Namespace,
    config: Mapping[str, Any],
) -> Any:
    cli_value = getattr(args, name)
    default_value = getattr(defaults, name)
    if cli_value != default_value:
        return cli_value
    if name in config:
        return config[name]
    return default_value


def _build_goal_spec(payload: Mapping[str, Any] | None) -> GoalSpec | None:
    if not payload:
        return None
    allowed = {field.name for field in fields(GoalSpec)}
    goal_kwargs = {key: payload[key] for key in payload.keys() & allowed}
    return GoalSpec(**goal_kwargs)


def _resolve_output_base(model_path: str, override_dir: str | None) -> Path:
    if override_dir:
        return Path(override_dir)
    model_parent = Path(model_path).resolve().parent
    run_root = model_parent.parent if model_parent.name == "checkpoints" else model_parent
    return run_root / "custom_eval"


def _summarize_results(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {
            "episodes": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "median_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
        }
    total = len(results)
    rewards = sorted(result["reward"] for result in results)
    successes = sum(1 for result in results if result["success"])
    mid = total // 2
    if total % 2 == 0:
        median = (rewards[mid - 1] + rewards[mid]) / 2
    else:
        median = rewards[mid]
    return {
        "episodes": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total,
        "avg_reward": sum(rewards) / total,
        "median_reward": median,
        "max_reward": rewards[-1],
        "min_reward": rewards[0],
    }


def _write_results(
    output_base: Path,
    log_lines: Sequence[str],
    metadata: Mapping[str, Any],
    results: Sequence[EpisodeResult],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S%z")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "summary": _summarize_results(results),
        "episodes": list(results),
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    args = parser.parse_args()
    config = _load_config(args.config)

    model_path = _resolve_option("model_path", args=args, defaults=defaults, config=config)
    mods_path = _resolve_option("mods", args=args, defaults=defaults, config=config)
    episodes = int(_resolve_option("episodes", args=args, defaults=defaults, config=config))
    seed = int(_resolve_option("seed", args=args, defaults=defaults, config=config))
    max_steps = int(_resolve_option("max_steps", args=args, defaults=defaults, config=config))
    deterministic = bool(_resolve_option("deterministic", args=args, defaults=defaults, config=config))
    sample_count = int(_resolve_option("sample_count", args=args, defaults=defaults, config=config))
    goal_spec = _build_goal_spec(config.get("goal"))
    store_results = bool(_resolve_option("store_results", args=args, defaults=defaults, config=config))
    output_dir_override = _resolve_option("output_dir", args=args, defaults=defaults, config=config)
    output_dir_override = output_dir_override or None

    engine, base_id = build_ring_engine(str(mods_path), seed)
    env = wrap_with_mask(make_env(engine, base_id, max_steps, seed, goal=goal_spec))
    model = MaskablePPO.load(str(model_path), env=env)
    rng = random.Random(seed)
    base_ids = _choose_bases(list(engine.db.bases_by_id.keys()), rng, sample_count)
    results: list[EpisodeResult] = []
    log_lines: list[str] = []
    header = f"Loaded model from {model_path} -> evaluating on {len(base_ids)} bases"
    print(header)
    log_lines.append(header)
    for episode in range(episodes):
        chosen_base = rng.choice(base_ids)
        result = run_episode(model, env, chosen_base, deterministic=deterministic, rng=rng)
        trace = " -> ".join(result["actions"])
        outcome = "SUCCESS" if result["success"] else "FAIL"
        line = (
            f"Episode {episode:02d} base={result['base']} {outcome} "
            f"reward={result['reward']:.2f} steps={result['steps']}\n  actions: {trace}"
        )
        print(line)
        log_lines.append(line)
        results.append(result)
    env.close()

    if store_results:
        output_base = _resolve_output_base(str(model_path), output_dir_override)
        metadata = {
            "model_path": str(model_path),
            "mods_path": str(mods_path),
            "episodes": episodes,
            "seed": seed,
            "max_steps": max_steps,
            "deterministic": deterministic,
            "sample_count": sample_count,
            "goal": asdict(goal_spec) if goal_spec else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": args.config,
        }
        final_dir = _write_results(output_base, log_lines, metadata, results)
        print(f"Stored evaluation artifacts under {final_dir}")


if __name__ == "__main__":
    main()
