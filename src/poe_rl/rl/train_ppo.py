from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import List, Sequence, Tuple, TypedDict, cast

import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from ..data.loader import load_essences, load_item_bases, load_mods
from ..data.models import ItemBase
from ..engine.core import CraftingEngine
from ..engine.database import CraftingDatabase
from ..rl.envs.ring_crafting_env import (
    GoalSpec,
    RewardConfig,
    RingCraftingEnvV1,
    reward_config_from_dict,
    reward_config_to_dict,
)
from ..main import build_price_provider

DEFAULT_DATA_PATH = Path("src/poe_rl/data/static/poec_data.json")
DEFAULT_MODEL_PATH = Path("artifacts/ppo_ring_policy.zip")
DEFAULT_LOG_DIR = Path("runs/ppo")


class EpisodeStat(TypedDict):
    reward: float
    steps: int
    success: float
    actions: list[str]
    base_id: str


def _log_success_traces(stats: List[EpisodeStat], seed: int, label: str, log_path: Path) -> None:
    lines: List[str] = []
    for idx, entry in enumerate(stats):
        if entry["success"]:
            trace = " -> ".join(entry["actions"])
            line = (
                f"{label} seed={seed} episode={idx} base={entry['base_id']} "
                f"reward={entry['reward']:.2f} steps={entry['steps']} actions={trace}"
            )
            print(f"Success trace: {line}")
            lines.append(line)
    if not lines:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    print(f"Logged {len(lines)} success traces to {log_path}")


def _log_eval_traces(
    stats: List[EpisodeStat],
    label: str,
    log_path: Path,
    *,
    summary: str | None = None,
) -> None:
    if not stats and not summary:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        if summary:
            handle.write(f"[{label}] {summary}\n")
        for idx, entry in enumerate(stats):
            outcome = "success" if entry["success"] else "fail"
            trace = " -> ".join(entry["actions"])
            line = (
                f"[{label}] episode={idx} base={entry['base_id']} {outcome} "
                f"reward={entry['reward']:.2f} steps={entry['steps']} actions={trace}"
            )
            handle.write(line + "\n")


class TensorboardMetricsCallback(BaseCallback):
    """Log episodic success rate/score to TensorBoard for easier monitoring."""

    def __init__(self, log_interval: int = 25, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_interval = log_interval
        self._pending_success: list[float] = []
        self._pending_scores: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True
        for done, info in zip(dones, infos):
            if done:
                self._pending_success.append(1.0 if info.get("success") else 0.0)
                self._pending_scores.append(float(info.get("score", 0.0)))
        if len(self._pending_success) >= self.log_interval:
            self._dump_metrics()
        return True

    def _on_training_end(self) -> None:
        if self._pending_success:
            self._dump_metrics()

    def _dump_metrics(self) -> None:
        success_rate = sum(self._pending_success) / len(self._pending_success)
        avg_score = (
            sum(self._pending_scores) / len(self._pending_scores)
            if self._pending_scores
            else 0.0
        )
        self.logger.record("custom/success_rate", success_rate)
        self.logger.record("custom/avg_score", avg_score)
        self._pending_success.clear()
        self._pending_scores.clear()


class PeriodicEvaluationCallback(BaseCallback):
    """Run evaluation rollouts every N timesteps and log summary metrics."""

    def __init__(
        self,
        eval_env: gym.Env,
        *,
        eval_episodes: int,
        eval_frequency: int,
        seed: int,
        log_path: Path,
        base_choices: Sequence[str] | None = None,
        rng_seed: int | None = None,
        log_traces: bool = False,
        trace_log_path: Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_frequency = eval_frequency
        self.seed = seed
        self.log_path = log_path
        self.base_choices = base_choices
        self.log_traces = log_traces
        self.trace_log_path = trace_log_path
        self._rng = random.Random(rng_seed if rng_seed is not None else seed)
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.eval_frequency <= 0:
            return True
        if (self.num_timesteps - self._last_eval_step) < self.eval_frequency:
            return True
        self._last_eval_step = self.num_timesteps
        model = cast(MaskablePPO, self.model)
        stats = evaluate_policy(
            model,
            self.eval_env,
            self.eval_episodes,
            base_choices=self.base_choices,
            rng=self._rng,
            log_all_traces=self.log_traces,
        )
        avg_reward = sum(entry["reward"] for entry in stats) / len(stats)
        avg_steps = sum(entry["steps"] for entry in stats) / len(stats)
        success_rate = sum(entry["success"] for entry in stats) / len(stats)
        label = f"periodic_eval@{self.num_timesteps}"
        if self.verbose:
            print(
                f"[{label}] reward={avg_reward:.2f}, steps={avg_steps:.1f}, success_rate={success_rate:.2%}"
            )
        self.logger.record("custom/eval_reward", avg_reward)
        self.logger.record("custom/eval_steps", avg_steps)
        self.logger.record("custom/eval_success_rate", success_rate)
        if self.trace_log_path is not None:
            summary = f"reward={avg_reward:.2f}, steps={avg_steps:.1f}, success_rate={success_rate:.2%}"
            _log_eval_traces(stats, label, self.trace_log_path, summary=summary)
        _log_success_traces(stats, self.seed, label, self.log_path)
        return True


def _is_ring_base(base: ItemBase) -> bool:
    if base.item_class and base.item_class.lower() == "ring":
        return True
    return any(tag.lower() == "ring" for tag in base.tags)


def _select_ring_base(bases_by_id: dict[str, ItemBase], preferred_base_id: str = "12") -> str:
    if preferred_base_id in bases_by_id:
        return preferred_base_id
    if not bases_by_id:
        raise ValueError("Ring base database is empty")
    candidates = sorted(bases_by_id.values(), key=lambda base: (base.drop_level, base.name))
    return candidates[-1].base_id


def build_ring_engine(mods_path: str, seed: int) -> Tuple[CraftingEngine, str]:
    mods = load_mods(mods_path)
    ring_mods = [mod for mod in mods if any(cls.lower() == "ring" for cls in mod.item_classes)]
    if not ring_mods:
        raise ValueError("No ring modifiers found in static data; cannot build engine")

    essences = load_essences(mods_path)
    bases = load_item_bases(mods_path)
    ring_bases = [base for base in bases if _is_ring_base(base)]
    if not ring_bases:
        raise ValueError("No ring bases found in static data; cannot build engine")

    bases_by_id = {base.base_id: base for base in ring_bases}
    db = CraftingDatabase(
        mods_by_id={mod.mod_id: mod for mod in ring_mods},
        bases_by_id=bases_by_id,
        essences_by_id={ess.essence_id: ess for ess in essences},
    )

    price_provider = build_price_provider()
    engine = CraftingEngine(db=db, price_provider=price_provider, rng=random.Random(seed))
    base_id = _select_ring_base(bases_by_id)
    return engine, base_id


def _mask_fn(env: gym.Env) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    if not hasattr(base_env, "get_action_mask"):
        raise AttributeError("Environment must expose get_action_mask() for action masking")
    return base_env.get_action_mask()  # type: ignore[no-any-return]


def make_env(
    engine: CraftingEngine,
    base_id: str,
    max_steps: int,
    seed: int,
    reward_config: RewardConfig | None = None,
    reward_profile: str | None = None,
    goal: GoalSpec | None = None,
) -> RingCraftingEnvV1:
    env = RingCraftingEnvV1(
        engine,
        base_id=base_id,
        max_steps=max_steps,
        goal=goal,
        reward_config=reward_config,
        reward_profile=reward_profile,
    )
    env.reset(seed=seed)
    return env


def wrap_with_mask(env: RingCraftingEnvV1) -> gym.Env:
    return ActionMasker(env, _mask_fn)


def evaluate_policy(
    model: MaskablePPO,
    env: gym.Env,
    episodes: int,
    *,
    base_choices: Sequence[str] | None = None,
    rng: random.Random | None = None,
    log_all_traces: bool = False,
) -> list[EpisodeStat]:
    results: list[EpisodeStat] = []
    base_env = cast(RingCraftingEnvV1, env.unwrapped)
    if not hasattr(base_env, "get_action_name"):
        raise AttributeError("Environment must expose get_action_name() for action logging")
    rng = rng or random.Random(0)
    choice_pool = list(base_choices) if base_choices else None
    for episode in range(episodes):
        if choice_pool:
            chosen_base = rng.choice(choice_pool)
            base_env.set_base(chosen_base)
        else:
            chosen_base = base_env.base_id
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False
        action_trace: list[str] = []
        while not done:
            mask = info.get("action_mask")
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action_name = base_env.get_action_name(int(action))
            action_trace.append(action_name)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
            success = info.get("success", success)
        episode_stat: EpisodeStat = {
            "reward": total_reward,
            "steps": steps,
            "success": float(success),
            "actions": action_trace,
            "base_id": chosen_base,
        }
        results.append(episode_stat)
        if log_all_traces:
            outcome = "success" if success else "fail"
            trace = " -> ".join(action_trace)
            print(
                f"Eval episode {episode} base={chosen_base} {outcome} reward={total_reward:.2f} "
                f"steps={steps} actions={trace}"
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for PoE2 ring crafting")
    parser.add_argument("--mods", default=str(DEFAULT_DATA_PATH), help="Path to poec_data.json dump")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total PPO timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for engine and PPO")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate")
    parser.add_argument("--max-steps", type=int, default=200, help="Episode step cap inside the env")
    parser.add_argument("--save-path", default=str(DEFAULT_MODEL_PATH), help="Where to store the trained policy")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="TensorBoard log directory")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes to run after training for quick sanity check")
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=0,
        help="Timesteps between periodic eval sweeps during training (0 disables)",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=100_000,
        help="Save an intermediate checkpoint every N timesteps (0 disables)",
    )
    parser.add_argument(
        "--eval-random-bases",
        action="store_true",
        help="When set, evaluation episodes sample random ring bases",
    )
    parser.add_argument(
        "--eval-print-traces",
        action="store_true",
        help="Print action traces for every evaluation episode",
    )
    parser.add_argument(
        "--reward-profile",
        default=None,
        help="Named reward profile defined in ring_crafting_env.REWARD_PROFILES",
    )
    parser.add_argument(
        "--reward-config-json",
        default=None,
        help="Path to JSON file describing a custom RewardConfig (overrides --reward-profile)",
    )
    return parser.parse_args()


def _load_reward_config(path: str) -> RewardConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return reward_config_from_dict(payload)


def _next_run_name(log_dir: Path, prefix: str) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)")
    next_index = 1
    for entry in log_dir.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)
    return f"{prefix}_{next_index}"


def _persist_reward_metadata(
    reward_config: RewardConfig,
    args: argparse.Namespace,
    save_path: Path,
    run_dir: Path,
) -> None:
    snapshot = {
        "timestamp": time.time(),
        "reward_config": reward_config_to_dict(reward_config),
        "train_args": vars(args),
        "run_dir": str(run_dir),
    }
    model_path = save_path
    reward_file = model_path.with_name(f"{model_path.stem}.reward.json")
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    reward_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_file = run_dir / "reward_config.json"
    run_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def _configure_run_logger(run_dir: Path) -> sb3_logger.Logger:
    """Force SB3 to log TensorBoard/CSV artifacts inside the chosen run_dir."""
    format_strings = ["stdout", "log", "csv", "tensorboard"]
    return sb3_logger.configure(folder=str(run_dir), format_strings=format_strings)


def main() -> None:
    args = parse_args()
    log_dir_path = Path(args.log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    engine, base_id = build_ring_engine(args.mods, args.seed)

    reward_config_override: RewardConfig | None = None
    if args.reward_config_json:
        reward_config_override = _load_reward_config(args.reward_config_json)

    run_name = _next_run_name(log_dir_path, "MaskablePPO")
    run_dir = log_dir_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = wrap_with_mask(
        make_env(
            engine,
            base_id,
            args.max_steps,
            args.seed,
            reward_config=reward_config_override,
            reward_profile=args.reward_profile,
        )
    )
    active_reward_config = cast(RingCraftingEnvV1, env.unwrapped).reward_config
    checkpoint_frequency = max(0, args.checkpoint_frequency)
    base_choices: Sequence[str] | None = None
    if args.eval_random_bases:
        base_choices = list(engine.db.bases_by_id.keys())

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        verbose=1,
        seed=args.seed,
        ent_coef=0.01,
        tensorboard_log=None,
    )
    run_logger = _configure_run_logger(run_dir)
    model.set_logger(run_logger)
    callbacks: List[BaseCallback] = [TensorboardMetricsCallback()]

    _persist_reward_metadata(
        active_reward_config,
        args,
        Path(args.save_path),
        run_dir,
    )

    if checkpoint_frequency > 0:
        checkpoint_prefix = Path(args.save_path).stem or "ppo_checkpoint"
        checkpoint_path = run_dir / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_frequency,
                save_path=str(checkpoint_path),
                name_prefix=checkpoint_prefix,
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=1,
            )
        )
    periodic_eval_env: gym.Env | None = None
    success_log_path = run_dir / "success_actions.log"
    eval_trace_log_path = run_dir / "eval_traces.log"
    if args.eval_frequency > 0 and args.eval_episodes > 0:
        periodic_eval_env = wrap_with_mask(
            make_env(
                engine,
                base_id,
                args.max_steps,
                args.seed + 10_000,
                reward_config=active_reward_config,
            )
        )
        callbacks.append(
            PeriodicEvaluationCallback(
                periodic_eval_env,
                eval_episodes=args.eval_episodes,
                eval_frequency=args.eval_frequency,
                seed=args.seed,
                log_path=success_log_path,
                base_choices=base_choices,
                rng_seed=args.seed + 777,
                log_traces=args.eval_print_traces,
                trace_log_path=eval_trace_log_path,
                verbose=1,
            )
        )
    callback: BaseCallback | CallbackList
    if len(callbacks) == 1:
        callback = callbacks[0]
    else:
        callback = CallbackList(callbacks)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_path)
    run_model_path = run_dir / Path(args.save_path).name
    model.save(str(run_model_path))
    if periodic_eval_env is not None:
        periodic_eval_env.close()

    if args.eval_episodes > 0:
        eval_env = wrap_with_mask(
            make_env(
                engine,
                base_id,
                args.max_steps,
                args.seed + 1,
                reward_config=active_reward_config,
            )
        )
        stats = evaluate_policy(
            model,
            eval_env,
            args.eval_episodes,
            base_choices=base_choices,
            rng=random.Random(args.seed + 9999),
            log_all_traces=args.eval_print_traces,
        )
        avg_reward = sum(entry["reward"] for entry in stats) / len(stats)
        avg_steps = sum(entry["steps"] for entry in stats) / len(stats)
        success_rate = sum(entry["success"] for entry in stats) / len(stats)
        print(
            f"Eval over {args.eval_episodes} episodes -> reward={avg_reward:.2f}, steps={avg_steps:.1f}, success_rate={success_rate:.2%}"
        )
        _log_eval_traces(
            stats,
            "final_eval",
            eval_trace_log_path,
            summary=f"reward={avg_reward:.2f}, steps={avg_steps:.1f}, success_rate={success_rate:.2%}",
        )
        _log_success_traces(stats, args.seed, "final_eval", success_log_path)
        summary_path = run_dir / "final_eval.json"
        summary = {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "episodes": stats,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        eval_env.close()

    env.close()


if __name__ == "__main__":
    main()
