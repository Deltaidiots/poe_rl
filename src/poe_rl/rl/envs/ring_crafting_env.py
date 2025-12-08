from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...data.models import Item, Mod
from ...engine.actions import ACTIONS, CraftingAction, create_essence_actions
from ...engine.core import CraftingEngine

# --------------------------------------------------------------------------------------
# v1 Design Constants
# --------------------------------------------------------------------------------------

# Life/res group definitions derived from ring_inspector.py results
LIFE_GROUPS: Tuple[str, ...] = (
    "AttackDamage",
    "IncreasedLife",
    "LifeGainedFromEnemyDeath",
    "LifeLeech",
    "LifeRegeneration",
    "RecoverPercentMaxLifeOnKill",
)

RES_GROUPS: Tuple[str, ...] = (
    "AllResistances",
    "ChaosResistance",
    "ColdAndChaosDamageResistance",
    "ColdResistance",
    "FireAndChaosDamageResistance",
    "FireResistance",
    "LightningAndChaosDamageResistance",
    "LightningResistance",
)

TAG_VOCAB: Tuple[str, ...] = (
    "life",
    "resistance",
    "elemental",
    "chaos",
    "attack",
    "spell",
    "physical",
    "minion",
)

# Action roster (order matters – aligns with design spec indices)
BASE_ACTION_SEQUENCE: Tuple[str, ...] = (
    "Orb of Transmutation",
    "Greater Orb of Transmutation",
    "Perfect Orb of Transmutation",
    "Orb of Augmentation",
    "Greater Orb of Augmentation",
    "Perfect Orb of Augmentation",
    "Regal Orb",
    "Greater Regal Orb",
    "Perfect Regal Orb",
    "Orb of Alchemy",
    "Chaos Orb",
    "Greater Chaos Orb",
    "Perfect Chaos Orb",
    "Exalted Orb",
    "Greater Exalted Orb",
    "Perfect Exalted Orb",
    "Orb of Annulment",
)

ESSENCE_ACTION_SEQUENCE: Tuple[str, ...] = (
    "Essence of the Body",
    "Lesser Essence of the Body",
    "Essence of Grounding",
    "Essence of Insulation",
    "Essence of Ruin",
    "Essence of Thawing",
    "Greater Essence of Grounding",
    "Greater Essence of Insulation",
    "Greater Essence of Ruin",
    "Greater Essence of Thawing",
    "Lesser Essence of Grounding",
    "Lesser Essence of Insulation",
    "Lesser Essence of Ruin",
    "Lesser Essence of Thawing",
)

OMEN_ACTION_SEQUENCE: Tuple[str, ...] = (
    "Omen of Dextral Exaltation",
    "Omen of Sinistral Exaltation",
    "Omen of Greater Exaltation",
    "Omen of Homogenising Exaltation",
    "Omen of Dextral Annulment",
    "Omen of Sinistral Annulment",
    "Omen of Greater Annulment",
)

STOP_ACTION_NAME = "Stop"

# Observation layout sizing
NUM_LIFE_GROUPS = len(LIFE_GROUPS)
NUM_RES_GROUPS = len(RES_GROUPS)
NUM_TAG_FEATURES = len(TAG_VOCAB) * 2
GLOBAL_FEATURES = 12  # rarity (3) + affix slots (4) + corruption/fracture (3) + ilvl/quality (2)
GROUP_FEATURES = 4 * (NUM_LIFE_GROUPS + NUM_RES_GROUPS)
AGG_FEATURES = 7
GOAL_FEATURES = 8
OBSERVATION_SIZE = GLOBAL_FEATURES + GROUP_FEATURES + AGG_FEATURES + NUM_TAG_FEATURES + GOAL_FEATURES

# Reward constants
VALUE_NORMALISER = 150.0
TIER_NORMALISER = 5  # assume most mods have <= 6 tiers
NO_PROGRESS_EPS = 1e-4


@dataclass(frozen=True)
class StatTarget:
    name: str
    groups: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    required_count: int = 0
    max_tier: int = TIER_NORMALISER
    weight: float = 1.0
    quality_weight: float = 0.5
    use_prefixes_only: bool = False
    use_suffixes_only: bool = False

    def __post_init__(self) -> None:
        if self.use_prefixes_only and self.use_suffixes_only:
            raise ValueError(f"StatTarget '{self.name}' cannot force both prefix and suffix only matching")


@dataclass(frozen=True)
class TargetEvalResult:
    name: str
    count: int
    met: bool
    progress_score: float
    quality_score: float
    combined_score: float
    required_count: int
    weight: float


@dataclass(frozen=True)
class ProgressTargetConfig:
    target_name: str
    new_mod_reward: float
    loss_penalty: float
    tier_improvement_reward: float
    tier_loss_penalty: float
    milestone_bonuses: Tuple[float, ...]


@dataclass(frozen=True)
class TargetSnapshot:
    mod_ids: Set[str]
    group_best_tiers: Dict[str, float]


@dataclass(frozen=True)
class RewardConfig:
    name: str = "default"
    shaping_coefficient: float = 2.0
    success_bonus: float = 20.0
    stop_fail_penalty: float = 1.0
    cost_coefficient: float = 0.005
    step_penalty: float = 0.008
    impossible_state_penalty: float = 4.0
    new_res_mod_reward: float = 1.2
    res_tier_improvement_reward: float = 0.85
    res_milestone_bonuses: Tuple[float, ...] = (0.5, 1.0, 2.0)
    res_loss_penalty: float = 0.35
    res_tier_loss_penalty: float = 0.1
    no_progress_penalty: float = 0.01
    repeat_action_penalty: float = 0.04
    repeat_penalty_streak_threshold: int = 3
    unconsumed_omen_penalty: float = 0.5
    omen_consumption_bonus: float = 0.5
    omen_progress_bonus: float = 0.6
    essence_progress_bonus: float = 0.6
    essence_first_mod_bonus: float = 0.4
    state_change_reward: float = 0.0
    # New fields for stagnation handling
    max_steps_without_progress: int = 40
    stagnation_penalty: float = 0.2
    progress_targets: Tuple[ProgressTargetConfig, ...] = ()


DEFAULT_REWARD_CONFIG = RewardConfig()
REWARD_PROFILES: Dict[str, RewardConfig] = {
    DEFAULT_REWARD_CONFIG.name: DEFAULT_REWARD_CONFIG,
    "cost_sensitive": RewardConfig(
        name="cost_sensitive",
        cost_coefficient=0.02,
        success_bonus=8.0,
        res_milestone_bonuses=(0.6, 1.2, 2.0),
        res_loss_penalty=0.3,
    ),
}
REPEAT_PENALTY_ACTIONS: Tuple[str, ...] = (
    "Chaos Orb",
    "Greater Chaos Orb",
    "Perfect Chaos Orb",
    "Exalted Orb",
    "Greater Exalted Orb",
    "Perfect Exalted Orb",
    *OMEN_ACTION_SEQUENCE,
)


@dataclass(frozen=True)
class ProgressStats:
    target_name: str
    prev_count: int
    current_count: int
    new_mods: int
    lost_mods: int
    tier_gain: float
    tier_loss: float
    milestone_bonus: float

    @property
    def made_positive_progress(self) -> bool:
        return (self.new_mods > 0) or (self.tier_gain > 0.0) or (self.milestone_bonus > 0.0)

    @property
    def created_first_mod(self) -> bool:
        return self.prev_count == 0 and self.current_count > 0


@dataclass(frozen=True)
class GoalSpec:
    life_required_count: int = 1
    life_max_tier: int = TIER_NORMALISER
    res_required_count: int = 2
    res_max_tier: int = 3
    life_weight: float = 0.5
    res_weight: float = 0.5


@dataclass
class GoalEvalResult:
    life_count: int
    res_count: int
    life_met: bool
    res_met: bool
    life_score: float
    res_score: float
    res_quality_score: float
    life_weight: float
    res_weight: float
    targets: Dict[str, TargetEvalResult] = field(default_factory=dict)
    composite_score_value: float = 0.0

    @property
    def success(self) -> bool:
        if self.targets:
            return all(target.met for target in self.targets.values())
        return self.life_met and self.res_met

    @property
    def composite_score(self) -> float:
        return self.composite_score_value


@dataclass
class ActionSpec:
    name: str
    action: CraftingAction


class RingCraftingEnvV1(gym.Env[np.ndarray, int]):
    """Gymnasium-compatible environment for the v1 PoE2 ring crafting PPO spec."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        engine: CraftingEngine,
        *,
        base_id: str = "12",
        item_ilvl: int = 80,
        goal: GoalSpec | None = None,
        stat_targets: Optional[Sequence[StatTarget]] = None,
        max_steps: int = 200,
        reward_profile: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.base_id = base_id
        self.item_ilvl = item_ilvl
        self.max_steps = max_steps
        self.goal = goal or GoalSpec()
        self.reward_config = self._resolve_reward_config(reward_config, reward_profile)
        self._stat_targets = tuple(stat_targets) if stat_targets else self._build_stat_targets(self.goal)
        if not self._stat_targets:
            raise ValueError("RingCraftingEnvV1 requires at least one stat target")
        self._target_lookup: Dict[str, StatTarget] = {target.name: target for target in self._stat_targets}
        self._target_group_sets: Dict[str, Set[str]] = {target.name: set(target.groups) for target in self._stat_targets}
        self._target_tag_sets: Dict[str, Set[str]] = {target.name: {tag.lower() for tag in target.tags} for target in self._stat_targets}
        self._progress_target_configs = self._resolve_progress_targets(self.reward_config)
        self._essence_progress_target = (
            next((target.name for target in self._stat_targets if target.name == "res"), self._stat_targets[0].name)
        )
        self._actions = self._build_actions()
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._rng = self.engine.rng
        self._item: Optional[Item] = None
        self._steps = 0
        self._last_score = 0.0
        self._last_action_mask = np.ones(len(self._actions), dtype=np.int8)
        self._prev_action_name: Optional[str] = None
        self._repeat_action_name: Optional[str] = None
        self._repeat_action_count: int = 0
        self._used_omens: Set[str] = set()
        # New episode-level tracking
        self._episode_raw_return: float = 0.0
        self._last_progress_step: int = 0

    def _resolve_reward_config(
        self,
        override: Optional[RewardConfig],
        profile_name: Optional[str],
    ) -> RewardConfig:
        if override is not None:
            return override
        if profile_name:
            profile = REWARD_PROFILES.get(profile_name)
            if profile is None:
                raise KeyError(f"Unknown reward profile '{profile_name}'")
            return profile
        return DEFAULT_REWARD_CONFIG

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._item = self.engine.create_item(self.base_id, self.item_ilvl)
        self._item.prefix_ids.clear()
        self._item.suffix_ids.clear()
        self._item.rarity = "Normal"
        self._item.active_omens.clear()
        self._item.corrupted = False
        self._item.fractured_mods.clear()
        self._steps = 0
        initial_snapshots = self._capture_target_snapshots(self._item)
        self._last_score = self._evaluate_goal(self._item, snapshots=initial_snapshots).composite_score
        self._prev_action_name = None
        self._reset_repeat_tracker()
        self._used_omens.clear()
        # Reset episode tracking
        self._episode_raw_return = 0.0
        self._last_progress_step = 0
        observation = self._encode_observation(self._item)
        self._last_action_mask = self._compute_action_mask(self._item)
        info = {"action_mask": self._last_action_mask.copy()}
        return observation, info

    def step(self, action_idx: int):  # type: ignore[override]
        if not (0 <= action_idx < len(self._actions)):
            raise ValueError(f"Action index {action_idx} is out of bounds")
        assert self._item is not None, "Environment must be reset before stepping"
        action_spec = self._actions[action_idx]
        last_action_name = action_spec.name

        cfg = self.reward_config

        # Handle Stop action explicitly
        if action_spec.name == STOP_ACTION_NAME:
            eval_result = self._evaluate_goal(self._item)
            reward = cfg.success_bonus if eval_result.success else -cfg.stop_fail_penalty
            if self._item.active_omens:
                reward -= cfg.unconsumed_omen_penalty * len(self._item.active_omens)
            terminated = True
            truncated = False
            observation = self._encode_observation(self._item)
            self._last_action_mask = np.zeros(len(self._actions), dtype=np.int8)
            self._prev_action_name = last_action_name
            self._reset_repeat_tracker()
            self._episode_raw_return += reward
            info = {
                "action_mask": self._last_action_mask.copy(),
                "success": eval_result.success,
                "score": eval_result.composite_score,
                "cost": 0.0,
                "last_action_name": last_action_name,
                "raw_episode_return": self._episode_raw_return,
            }
            return observation, reward, terminated, truncated, info

        # Guard against invalid actions (should be masked, but belt-and-braces)
        if not self._is_action_valid(action_spec.action, self._item, action_spec.name):
            reward = -cfg.stop_fail_penalty
            repeat_count = self._update_repeat_tracker(last_action_name, made_progress=False)
            if (
                last_action_name in REPEAT_PENALTY_ACTIONS
                and repeat_count >= cfg.repeat_penalty_streak_threshold
            ):
                streak_excess = repeat_count - cfg.repeat_penalty_streak_threshold + 1
                reward -= cfg.repeat_action_penalty * streak_excess
            observation = self._encode_observation(self._item)
            self._last_action_mask = self._compute_action_mask(self._item)
            self._prev_action_name = last_action_name
            self._episode_raw_return += reward
            info = {
                "action_mask": self._last_action_mask.copy(),
                "invalid_action": True,
                "cost": 0.0,
                "success": False,
                "score": self._last_score,
                "last_action_name": last_action_name,
            }
            return observation, reward, False, False, info

        prev_state_sig = self._item_state_signature(self._item)
        prev_snapshots = self._capture_target_snapshots(self._item)
        prev_active_omens = set(self._item.active_omens)

        try:
            new_item, cost = self.engine.apply(self._item, action_spec.action)
        except ValueError as exc:
            # Action failed due to unmet hidden condition
            reward = -cfg.stop_fail_penalty
            repeat_count = self._update_repeat_tracker(last_action_name, made_progress=False)
            if (
                last_action_name in REPEAT_PENALTY_ACTIONS
                and repeat_count >= cfg.repeat_penalty_streak_threshold
            ):
                streak_excess = repeat_count - cfg.repeat_penalty_streak_threshold + 1
                reward -= cfg.repeat_action_penalty * streak_excess
            observation = self._encode_observation(self._item)
            self._last_action_mask = self._compute_action_mask(self._item)
            self._prev_action_name = last_action_name
            self._episode_raw_return += reward
            info = {
                "action_mask": self._last_action_mask.copy(),
                "invalid_action": True,
                "error": str(exc),
                "cost": 0.0,
                "success": False,
                "score": self._last_score,
                "last_action_name": last_action_name,
            }
            return observation, reward, False, False, info

        self._steps += 1
        new_snapshots = self._capture_target_snapshots(new_item)
        eval_result = self._evaluate_goal(new_item, snapshots=new_snapshots)
        new_score = eval_result.composite_score
        score_delta = new_score - self._last_score
        shaping = cfg.shaping_coefficient * score_delta
        currency_penalty = cfg.cost_coefficient * float(cost)
        step_penalty = cfg.step_penalty
        progress_reward, progress_stats = self._progress_shaping(prev_snapshots, new_snapshots)
        any_progress = any(stat.made_positive_progress for stat in progress_stats.values())
        state_changed = prev_state_sig != self._item_state_signature(new_item)
        consumed_omens = prev_active_omens - set(new_item.active_omens)
        omen_bonus = 0.0
        if consumed_omens:
            omen_bonus = cfg.omen_consumption_bonus * len(consumed_omens)
            if any_progress:
                omen_bonus += cfg.omen_progress_bonus * len(consumed_omens)
        essence_bonus = 0.0
        if last_action_name in ESSENCE_ACTION_SEQUENCE:
            if any_progress:
                essence_bonus += cfg.essence_progress_bonus
            essence_target_stats = progress_stats.get(self._essence_progress_target)
            if essence_target_stats and essence_target_stats.created_first_mod:
                essence_bonus += cfg.essence_first_mod_bonus
        reward = shaping + progress_reward + omen_bonus + essence_bonus - currency_penalty - step_penalty
        if state_changed:
            reward += cfg.state_change_reward
        made_progress = (abs(score_delta) >= NO_PROGRESS_EPS) or any_progress
        if not made_progress:
            reward -= cfg.no_progress_penalty
        else:
            # Update "last progress" step index
            self._last_progress_step = self._steps
        repeat_count = self._update_repeat_tracker(last_action_name, made_progress)
        if (
            last_action_name in REPEAT_PENALTY_ACTIONS
            and not made_progress
            and repeat_count >= cfg.repeat_penalty_streak_threshold
        ):
            streak_excess = repeat_count - cfg.repeat_penalty_streak_threshold + 1
            reward -= cfg.repeat_action_penalty * streak_excess

        terminated = False
        truncated = False

        if eval_result.success:
            reward += cfg.success_bonus
            terminated = True
        elif self._is_impossible_state(new_item, eval_result):
            reward -= cfg.impossible_state_penalty
            truncated = True
        elif self._steps >= self.max_steps:
            truncated = True

        # Stagnation termination: only if we haven't already terminated/truncated
        if not terminated and not truncated:
            steps_since_progress = self._steps - self._last_progress_step
            if steps_since_progress >= cfg.max_steps_without_progress:
                truncated = True
                reward -= cfg.stagnation_penalty

        if last_action_name in OMEN_ACTION_SEQUENCE:
            self._used_omens.add(last_action_name)

        self._item = new_item
        self._last_score = new_score
        observation = self._encode_observation(new_item)
        self._last_action_mask = self._compute_action_mask(new_item)

        if terminated or truncated:
            omen_count = len(new_item.active_omens)
            if omen_count:
                reward -= cfg.unconsumed_omen_penalty * omen_count
            self._reset_repeat_tracker()
        self._episode_raw_return += reward

        self._prev_action_name = last_action_name
        info = {
            "action_mask": self._last_action_mask.copy(),
            "success": eval_result.success,
            "score": new_score,
            "cost": cost,
            "last_action_name": last_action_name,
            "omens_consumed": len(consumed_omens),
        }
        if terminated or truncated:
            info["raw_episode_return"] = self._episode_raw_return
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        """Expose the latest action mask for sb3-contrib's ActionMasker wrapper."""
        return self._last_action_mask.copy()

    def get_action_name(self, action_idx: int) -> str:
        if not (0 <= action_idx < len(self._actions)):
            raise ValueError(f"Action index {action_idx} is out of bounds")
        return self._actions[action_idx].name

    def _reset_repeat_tracker(self) -> None:
        self._repeat_action_name = None
        self._repeat_action_count = 0

    def _update_repeat_tracker(self, action_name: str, made_progress: bool) -> int:
        if made_progress or action_name != self._repeat_action_name:
            self._repeat_action_name = action_name
            self._repeat_action_count = 1
        else:
            self._repeat_action_count += 1
        return self._repeat_action_count

    def set_base(self, base_id: str) -> None:
        if base_id not in self.engine.db.bases_by_id:
            raise ValueError(f"Unknown base_id '{base_id}' for this database")
        self.base_id = base_id

    def _is_impossible_state(self, item: Item, eval_result: GoalEvalResult) -> bool:
        """Return True when the item can no longer meet the configured goal."""
        for target in self._stat_targets:
            target_eval = eval_result.targets.get(target.name)
            if target_eval is None or target_eval.met:
                continue
            if target.use_suffixes_only and not item.has_open_suffix():
                return True
            if target.use_prefixes_only and not item.has_open_prefix():
                return True
        return False

    def _build_actions(self) -> List[ActionSpec]:
        action_lookup = {action.name: action for action in ACTIONS}
        essence_lookup = {a.name: a for a in create_essence_actions(self.engine.db)}
        ordered_names: List[str] = [
            *BASE_ACTION_SEQUENCE,
            *ESSENCE_ACTION_SEQUENCE,
            *OMEN_ACTION_SEQUENCE,
            STOP_ACTION_NAME,
        ]
        action_specs: List[ActionSpec] = []
        for name in ordered_names:
            action = action_lookup.get(name) or essence_lookup.get(name)
            if action is None:
                raise KeyError(f"Crafting action '{name}' is not available in the current database")
            action_specs.append(ActionSpec(name=name, action=action))
        return action_specs

    def _compute_action_mask(self, item: Item) -> np.ndarray:
        mask = np.zeros(len(self._actions), dtype=np.int8)
        for idx, spec in enumerate(self._actions):
            if spec.name == STOP_ACTION_NAME:
                mask[idx] = 1
            elif self._is_action_valid(spec.action, item, spec.name):
                mask[idx] = 1
        return mask

    def _is_action_valid(
        self,
        action: CraftingAction,
        item: Item,
        action_name: Optional[str] = None,
    ) -> bool:
        name = action_name or action.name
        if name in OMEN_ACTION_SEQUENCE and name in self._used_omens:
            return False
        return all(req.check(item) for req in action.requirements)

    def _build_stat_targets(self, goal: GoalSpec) -> Tuple[StatTarget, ...]:
        return (
            StatTarget(
                name="life",
                groups=LIFE_GROUPS,
                required_count=goal.life_required_count,
                max_tier=goal.life_max_tier,
                weight=goal.life_weight,
                quality_weight=0.0,
                use_prefixes_only=True,
            ),
            StatTarget(
                name="res",
                groups=RES_GROUPS,
                required_count=goal.res_required_count,
                max_tier=goal.res_max_tier,
                weight=goal.res_weight,
                quality_weight=0.5,
                use_suffixes_only=True,
            ),
        )

    def _capture_target_snapshots(self, item: Item) -> Dict[str, TargetSnapshot]:
        snapshots: Dict[str, TargetSnapshot] = {}
        for target in self._stat_targets:
            mod_ids: Set[str] = set()
            best_tiers: Dict[str, float] = {}
            for mod_id in self._iter_target_mod_ids(item, target):
                mod = self.engine.db.mods_by_id.get(mod_id)
                if mod is None:
                    continue
                if not self._mod_matches_target(mod, target):
                    continue
                tier_value = self._resolve_mod_tier(mod)
                if tier_value > target.max_tier:
                    continue
                mod_ids.add(mod_id)
                prev = best_tiers.get(mod.group)
                if prev is None or tier_value < prev:
                    best_tiers[mod.group] = tier_value
            snapshots[target.name] = TargetSnapshot(mod_ids=mod_ids, group_best_tiers=best_tiers)
        return snapshots

    def _iter_target_mod_ids(self, item: Item, target: StatTarget) -> Iterable[str]:
        if target.use_suffixes_only and not target.use_prefixes_only:
            pools = (item.suffix_ids,)
        elif target.use_prefixes_only and not target.use_suffixes_only:
            pools = (item.prefix_ids,)
        else:
            pools = (item.prefix_ids, item.suffix_ids)
        for pool in pools:
            for mod_id in pool:
                yield mod_id

    def _mod_matches_target(self, mod: Mod, target: StatTarget) -> bool:
        group_set = self._target_group_sets.get(target.name)
        if group_set and mod.group in group_set:
            return True
        tag_set = self._target_tag_sets.get(target.name)
        if tag_set:
            mod_tags = {tag.lower() for tag in mod.tags}
            if mod_tags & tag_set:
                return True
        return False

    @staticmethod
    def _empty_snapshot() -> TargetSnapshot:
        return TargetSnapshot(mod_ids=set(), group_best_tiers={})

    def _target_quality_score(self, snapshot: TargetSnapshot) -> float:
        total = 0.0
        count = 0
        for tier in snapshot.group_best_tiers.values():
            total += self._normalise_tier(tier)
            count += 1
        if count == 0:
            return 0.0
        return min(total / 3.0, 1.0)

    def _resolve_progress_targets(self, cfg: RewardConfig) -> Tuple[ProgressTargetConfig, ...]:
        if cfg.progress_targets:
            return cfg.progress_targets
        return (
            ProgressTargetConfig(
                target_name="res",
                new_mod_reward=cfg.new_res_mod_reward,
                loss_penalty=cfg.res_loss_penalty,
                tier_improvement_reward=cfg.res_tier_improvement_reward,
                tier_loss_penalty=cfg.res_tier_loss_penalty,
                milestone_bonuses=cfg.res_milestone_bonuses,
            ),
        )

    def _encode_observation(self, item: Item) -> np.ndarray:
        group_summary, tag_summary = self._summarise_mods(item)
        eval_result = self._evaluate_goal(item)
        values: List[float] = []
        # Global features
        values.extend(self._rarity_one_hot(item.rarity))
        values.append(item.num_prefixes / 3.0)
        values.append(item.num_suffixes / 3.0)
        values.append(1.0 if item.has_open_prefix() else 0.0)
        values.append(1.0 if item.has_open_suffix() else 0.0)
        values.append(1.0 if item.corrupted else 0.0)
        values.append(len(item.fractured_mods) / 3.0)
        values.append(1.0 if self._has_fractured_priority_mod(item) else 0.0)
        values.append(item.ilvl / 100.0)
        values.append(item.quality / 20.0)
        # Group-specific features
        for group in LIFE_GROUPS:
            values.extend(self._group_features(group_summary.get(group)))
        for group in RES_GROUPS:
            values.extend(self._group_features(group_summary.get(group)))
        # Aggregate features
        values.extend(
            [
                1.0 if eval_result.life_count > 0 else 0.0,
                1.0 if eval_result.life_met else 0.0,
                self._quality_from_groups(group_summary, LIFE_GROUPS),
                min(eval_result.res_count / 3.0, 1.0),
                min(eval_result.res_count / (self.goal.res_required_count or 1), 1.0),
                1.0 if eval_result.res_met else 0.0,
                self._quality_from_groups(group_summary, RES_GROUPS),
            ]
        )
        # Tag features
        total_slots = max(1, item.num_prefixes + item.num_suffixes)
        for tag in TAG_VOCAB:
            count = tag_summary.get(tag, 0)
            values.append(1.0 if count > 0 else 0.0)
            values.append(min(count / total_slots, 1.0))
        # Goal features
        values.extend(
            [
                self.goal.life_required_count / 3.0,
                self._normalise_tier(self.goal.life_max_tier),
                0.0,  # placeholder for value threshold (unused in v1)
                self.goal.life_weight,
                self.goal.res_required_count / 3.0,
                self._normalise_tier(self.goal.res_max_tier),
                0.0,  # placeholder for res value threshold
                self.goal.res_weight,
            ]
        )
        assert len(values) == OBSERVATION_SIZE, f"Expected {OBSERVATION_SIZE} features, got {len(values)}"
        return np.array(values, dtype=np.float32)

    def _summarise_mods(self, item: Item) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
        summary: Dict[str, Dict[str, float]] = {}
        tags: Dict[str, int] = {}
        for mod_id in item.prefix_ids + item.suffix_ids:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod is None:
                continue
            entry = summary.setdefault(mod.group, {"count": 0.0, "best_tier": math.inf, "best_value": 0.0})
            entry["count"] += 1.0
            tier_value = self._resolve_mod_tier(mod)
            entry["best_tier"] = min(entry["best_tier"], tier_value)
            value = self._extract_mod_value(mod)
            entry["best_value"] = max(entry["best_value"], value)
            for tag in mod.tags:
                key = tag.lower()
                tags[key] = tags.get(key, 0) + 1
        return summary, tags

    @staticmethod
    def _extract_mod_value(mod: Mod) -> float:
        if mod.max_value is not None:
            return float(mod.max_value)
        if mod.min_value is not None:
            return float(mod.min_value)
        if "value_max" in mod.stat_values:
            return float(mod.stat_values["value_max"])
        if "value_min" in mod.stat_values:
            return float(mod.stat_values["value_min"])
        if mod.stat_values:
            return float(next(iter(mod.stat_values.values())))
        return 0.0

    def _evaluate_goal(
        self,
        item: Item,
        *,
        snapshots: Optional[Dict[str, TargetSnapshot]] = None,
    ) -> GoalEvalResult:
        target_snapshots = snapshots or self._capture_target_snapshots(item)
        target_results: Dict[str, TargetEvalResult] = {}
        total_weight = sum(max(target.weight, 0.0) for target in self._stat_targets)
        weighted_score = 0.0
        for target in self._stat_targets:
            snapshot = target_snapshots.get(target.name, self._empty_snapshot())
            count = len(snapshot.mod_ids)
            required = max(target.required_count, 0)
            met = True if required == 0 else count >= required
            if required == 0:
                progress_score = 1.0
            else:
                progress_score = min(count / required, 1.0)
            quality_score = self._target_quality_score(snapshot)
            quality_weight = min(max(target.quality_weight, 0.0), 1.0)
            combined_score = ((1.0 - quality_weight) * progress_score) + (quality_weight * quality_score)
            target_results[target.name] = TargetEvalResult(
                name=target.name,
                count=count,
                met=met,
                progress_score=progress_score,
                quality_score=quality_score,
                combined_score=combined_score,
                required_count=required,
                weight=target.weight,
            )
            weighted_score += max(target.weight, 0.0) * combined_score

        normaliser = total_weight if total_weight > 1e-6 else 1e-6
        composite = weighted_score / normaliser

        life_eval = target_results.get("life")
        res_eval = target_results.get("res")

        life_count = life_eval.count if life_eval else 0
        res_count = res_eval.count if res_eval else 0
        life_met = life_eval.met if life_eval else (self.goal.life_required_count <= 0)
        res_met = res_eval.met if res_eval else (self.goal.res_required_count <= 0)
        life_score = life_eval.combined_score if life_eval else (1.0 if self.goal.life_required_count <= 0 else 0.0)
        res_score = res_eval.combined_score if res_eval else (1.0 if self.goal.res_required_count <= 0 else 0.0)
        res_quality_score = res_eval.quality_score if res_eval else 0.0

        return GoalEvalResult(
            life_count=life_count,
            res_count=res_count,
            life_met=life_met,
            res_met=res_met,
            life_score=life_score,
            res_score=res_score,
            res_quality_score=res_quality_score,
            life_weight=self.goal.life_weight,
            res_weight=self.goal.res_weight,
            targets=target_results,
            composite_score_value=composite,
        )

    @staticmethod
    def _resolve_mod_tier(mod: Mod) -> float:
        return float(mod.tier) if mod.tier is not None else float(TIER_NORMALISER)

    @staticmethod
    def _rarity_one_hot(rarity: str) -> List[float]:
        if rarity == "Normal":
            return [1.0, 0.0, 0.0]
        if rarity == "Magic":
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def _group_features(self, entry: Optional[Dict[str, float]]) -> List[float]:
        if not entry:
            return [0.0, 0.0, 0.0, 0.0]
        present = 1.0 if entry["count"] > 0 else 0.0
        count_norm = min(entry["count"] / 3.0, 1.0)
        tier_norm = self._normalise_tier(entry["best_tier"])
        value_norm = min(entry["best_value"] / VALUE_NORMALISER, 1.0)
        return [present, count_norm, tier_norm, value_norm]

    @staticmethod
    def _normalise_tier(tier: float) -> float:
        if tier <= 0:
            return 1.0
        tier = min(tier, float(TIER_NORMALISER))
        return 1.0 - (tier - 1.0) / (TIER_NORMALISER - 1.0)

    def _quality_from_groups(self, summary: Dict[str, Dict[str, float]], groups: Sequence[str]) -> float:
        best = 0.0
        for group in groups:
            entry = summary.get(group)
            if entry:
                best = max(best, self._normalise_tier(entry["best_tier"]))
        return best

    def _has_fractured_priority_mod(self, item: Item) -> bool:
        targets = tuple(self._stat_targets)
        for mod_id in item.fractured_mods:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod and any(self._mod_matches_target(mod, target) for target in targets):
                return True
        return False

    @staticmethod
    def _item_state_signature(item: Item) -> Tuple[
        Tuple[str, ...],
        Tuple[str, ...],
        str,
        Tuple[str, ...],
        bool,
        Tuple[str, ...],
    ]:
        return (
            tuple(item.prefix_ids),
            tuple(item.suffix_ids),
            item.rarity,
            tuple(sorted(item.active_omens)),
            item.corrupted,
            tuple(sorted(item.fractured_mods)),
        )

    def _progress_shaping(
        self,
        prev_snapshots: Dict[str, TargetSnapshot],
        current_snapshots: Dict[str, TargetSnapshot],
    ) -> Tuple[float, Dict[str, ProgressStats]]:
        """Reward incremental progress events per configured target."""

        reward = 0.0
        stats_by_target: Dict[str, ProgressStats] = {}
        for cfg in self._progress_target_configs:
            target = self._target_lookup.get(cfg.target_name)
            if target is None:
                continue
            prev_snapshot = prev_snapshots.get(target.name, self._empty_snapshot())
            current_snapshot = current_snapshots.get(target.name, self._empty_snapshot())
            prev_count = len(prev_snapshot.mod_ids)
            current_count = len(current_snapshot.mod_ids)
            new_mods = max(0, current_count - prev_count)
            lost_mods = max(0, prev_count - current_count)
            if new_mods:
                reward += cfg.new_mod_reward * new_mods
            if lost_mods:
                reward -= cfg.loss_penalty * lost_mods
            tier_gain = 0.0
            tier_loss = 0.0
            for group, new_tier in current_snapshot.group_best_tiers.items():
                prev_tier = prev_snapshot.group_best_tiers.get(group)
                if prev_tier is None:
                    continue
                if new_tier < prev_tier:
                    tier_gain += prev_tier - new_tier
                elif new_tier > prev_tier:
                    tier_loss += new_tier - prev_tier
            if tier_gain:
                reward += cfg.tier_improvement_reward * tier_gain
            if tier_loss:
                reward -= cfg.tier_loss_penalty * tier_loss
            milestone_bonus = 0.0
            for idx, bonus in enumerate(cfg.milestone_bonuses, start=1):
                if prev_count < idx <= current_count:
                    milestone_bonus += bonus
            reward += milestone_bonus
            stats_by_target[target.name] = ProgressStats(
                target_name=target.name,
                prev_count=prev_count,
                current_count=current_count,
                new_mods=new_mods,
                lost_mods=lost_mods,
                tier_gain=tier_gain,
                tier_loss=tier_loss,
                milestone_bonus=milestone_bonus,
            )
        return reward, stats_by_target


def reward_config_to_dict(config: RewardConfig) -> Dict[str, Any]:
    data = asdict(config)
    return data  # tuples become lists for JSON compatibility


def reward_config_from_dict(payload: Mapping[str, Any]) -> RewardConfig:
    field_names = {field.name for field in fields(RewardConfig)}
    kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in field_names:
            continue
        if key == "res_milestone_bonuses":
            value = tuple(value)
        if key == "progress_targets":
            value = tuple(ProgressTargetConfig(**entry) for entry in value)
        kwargs[key] = value
    return RewardConfig(**kwargs)

    # ------------------------------------------------------------------
    # Rendering helpers (optional)
    # ------------------------------------------------------------------
    def render(self):  # type: ignore[override]
        if self._item is None:
            print("<env not reset>")
            return
        eval_result = self._evaluate_goal(self._item)
        print(
            f"Step {self._steps}: Rarity={self._item.rarity}, "
            f"Prefixes={self._item.prefix_ids}, Suffixes={self._item.suffix_ids}, "
            f"Score={eval_result.composite_score:.3f}"
        )

    def close(self):  # type: ignore[override]
        return None


__all__ = [
    "RingCraftingEnvV1",
    "GoalSpec",
    "StatTarget",
    "TargetEvalResult",
    "ProgressTargetConfig",
    "RewardConfig",
    "ProgressStats",
    "reward_config_to_dict",
    "reward_config_from_dict",
]
