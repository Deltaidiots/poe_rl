from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from sb3_contrib.ppo_mask import MaskablePPO

from .train_ppo import build_ring_engine, make_env, wrap_with_mask


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
) -> dict[str, object]:
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


def main() -> None:
    args = parse_args()
    engine, base_id = build_ring_engine(args.mods, args.seed)
    env = wrap_with_mask(make_env(engine, base_id, args.max_steps, args.seed))
    model = MaskablePPO.load(args.model_path, env=env)
    rng = random.Random(args.seed)
    base_ids = _choose_bases(list(engine.db.bases_by_id.keys()), rng, args.sample_count)
    print(f"Loaded model from {args.model_path} -> evaluating on {len(base_ids)} bases")
    for episode in range(args.episodes):
        chosen_base = rng.choice(base_ids)
        result = run_episode(model, env, chosen_base, deterministic=args.deterministic, rng=rng)
        trace = " -> ".join(result["actions"])
        outcome = "SUCCESS" if result["success"] else "FAIL"
        print(
            f"Episode {episode:02d} base={result['base']} {outcome} "
            f"reward={result['reward']:.2f} steps={result['steps']}\n  actions: {trace}"
        )
    env.close()


if __name__ == "__main__":
    main()
