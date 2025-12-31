"""
Gymnasium environment for PoE2 ring crafting with PPO/MaskablePPO.

This environment simulates crafting a ring in Path of Exile 2, with the goal
of achieving specific mod configurations (e.g., +Life and +Resistance mods).

Key Features:
------------
- **Action Masking:** Invalid actions are masked to prevent impossible crafts
- **99-Dimensional Observation:** Item state, mod groups, goal progress
- **110 Discrete Actions:** Base currency, 81 essences, 11 omens, stop
- **Configurable Rewards:** RewardConfig allows tuning shaping signals

Environment Design:
------------------
- Observation includes: rarity, affix slots, mod group features, goal progress
- Actions are validated before execution (action mask)
- Episode ends when: goal achieved, max steps reached, or stop action
- Reward combines: progress bonuses, tier improvements, cost penalties

Usage Example:
    from poe_rl.rl.envs.ring_crafting_env import RingCraftingEnvV1
    from poe_rl.engine.core import CraftingEngine
    from sb3_contrib import MaskablePPO

    env = RingCraftingEnvV1(engine=engine)
    model = MaskablePPO("MlpPolicy", env)
    model.learn(total_timesteps=100000)

See Also:
    - docs/RL_JOURNEY.md for reward function evolution
    - docs/ARCHITECTURE.md for system design
"""

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

# Static essence sequence for life/res goal - includes all tier variants
# Use DYNAMIC_ESSENCE_ACTIONS=True to dynamically include all available essences
DYNAMIC_ESSENCE_ACTIONS = True

ESSENCE_ACTION_SEQUENCE: Tuple[str, ...] = (
    # Life essences (Essence of the Body - all tiers)
    "Lesser Essence of the Body",
    "Essence of the Body",
    "Greater Essence of the Body",
    "Perfect Essence of the Body",
    # Resistance essences - Grounding (Lightning Res)
    "Lesser Essence of Grounding",
    "Essence of Grounding",
    "Greater Essence of Grounding",
    "Perfect Essence of Grounding",
    # Resistance essences - Insulation (Cold Res)
    "Lesser Essence of Insulation",
    "Essence of Insulation",
    "Greater Essence of Insulation",
    "Perfect Essence of Insulation",
    # Resistance essences - Ruin (Chaos Res)
    "Lesser Essence of Ruin",
    "Essence of Ruin",
    "Greater Essence of Ruin",
    "Perfect Essence of Ruin",
    # Resistance essences - Thawing (Fire Res)
    "Lesser Essence of Thawing",
    "Essence of Thawing",
    "Greater Essence of Thawing",
    "Perfect Essence of Thawing",
)

OMEN_ACTION_SEQUENCE: Tuple[str, ...] = (
    "Omen of Dextral Exaltation",
    "Omen of Sinistral Exaltation",
    "Omen of Greater Exaltation",
    "Omen of Homogenising Exaltation",
    "Omen of Dextral Annulment",
    "Omen of Sinistral Annulment",
    "Omen of Greater Annulment",
    "Omen of Dextral Crystallisation",
    "Omen of Sinistral Crystallisation",
    "Omen of Dextral Erasure",
    "Omen of Sinistral Erasure",
)

STOP_ACTION_NAME = "Stop"

CHAOS_ACTIONS: Tuple[str, ...] = (
    "Chaos Orb",
    "Greater Chaos Orb",
    "Perfect Chaos Orb",
)
EXALTED_ACTIONS: Tuple[str, ...] = (
    "Exalted Orb",
    "Greater Exalted Orb",
    "Perfect Exalted Orb",
)
REGAL_ACTIONS: Tuple[str, ...] = (
    "Regal Orb",
    "Greater Regal Orb",
    "Perfect Regal Orb",
)
ANNULMENT_ACTIONS: Tuple[str, ...] = ("Orb of Annulment",)

EXALTATION_OMENS: Set[str] = {
    "Omen of Dextral Exaltation",
    "Omen of Sinistral Exaltation",
    "Omen of Greater Exaltation",
    "Omen of Homogenising Exaltation",
}
REGAL_OMENS: Set[str] = {
    "Omen of Dextral Coronation",
    "Omen of Sinistral Coronation",
    "Omen of Homogenising Coronation",
}
CHAOS_OMENS: Set[str] = {
    "Omen of Dextral Erasure",
    "Omen of Sinistral Erasure",
    "Omen of Whittling",
}
ANNULMENT_OMENS: Set[str] = {
    "Omen of Dextral Annulment",
    "Omen of Sinistral Annulment",
    "Omen of Greater Annulment",
    "Omen of Light",
}
ESSENCE_KEYWORD = "Essence"

SLOT_PHASE_RESET_ACTIONS: Set[str] = {
    "Orb of Scouring",
    "Orb of Transmutation",
    "Orb of Augmentation",
    "Orb of Alchemy",
    "Orb of Chance",
}
SLOT_PHASE_RESET_ACTIONS.update(ANNULMENT_ACTIONS)
SLOT_PHASE_RESET_ACTIONS.update(BASE_ACTION_SEQUENCE[:3])
SLOT_PHASE_RESET_ACTIONS.update(ESSENCE_ACTION_SEQUENCE)

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
class RewardChannelWeights:
    general: float = 1.0
    target_specific: float = 1.0
    omen: float = 1.0
    cost: float = 1.0
    guardrail: float = 1.0


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
    omen_timely_window: int = 5
    omen_timely_consumption_bonus: float = 0.35
    state_change_reward: float = 0.0
    early_rare_penalty: float = 0.6
    deterministic_roll_bonus: float = 0.4
    deterministic_cost_discount: float = 0.6
    deterministic_bonus_max_streak: int = 2
    family_cost_free_uses: int = 4
    family_cost_ramp: float = 0.03
    slot_congestion_grace_steps: int = 40
    slot_congestion_penalty: float = 0.2
    slot_congestion_growth: float = 0.02
    slot_churn_threshold: int = 3
    slot_churn_penalty: float = 0.15
    structural_affix_gain_reward: float = 0.4
    structural_affix_loss_penalty: float = 0.25
    structural_quality_reward: float = 1.0
    structural_quality_penalty: float = 0.6
    structural_balance_penalty: float = 0.1
    # Essence-specific incentives: reward guaranteed/targeted outcomes
    essence_guaranteed_mod_bonus: float = 0.8  # bonus for using essence (guaranteed mod)
    essence_goal_aligned_bonus: float = 1.2  # extra bonus if essence adds life/res
    essence_early_use_bonus: float = 0.5  # bonus for using essence in first N steps
    essence_early_use_window: int = 10  # steps within which early_use_bonus applies
    # RNG exploitation penalties: discourage pure chaos spam
    chaos_spam_threshold: int = 5  # start penalizing after this many chaos uses
    chaos_spam_penalty_ramp: float = 0.05  # penalty per chaos use over threshold
    rng_action_base_penalty: float = 0.02  # small per-use cost for purely RNG actions
    channels: RewardChannelWeights = field(default_factory=RewardChannelWeights)


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
REPEAT_PENALTY_FAMILIES: Tuple[str, ...] = (
    "chaos",
    "exalt",
    "annul",
    "regal",
    "essence",
    "omen",
)


@dataclass(frozen=True)
class ProgressStats:
    prev_count: int
    current_count: int
    new_res_mods: int
    lost_res_mods: int
    tier_gain: float
    tier_loss: float
    milestone_bonus: float

    @property
    def made_positive_progress(self) -> bool:
        return (self.new_res_mods > 0) or (self.tier_gain > 0.0) or (self.milestone_bonus > 0.0)

    @property
    def created_first_res_mod(self) -> bool:
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

    @property
    def success(self) -> bool:
        return self.life_met and self.res_met

    @property
    def composite_score(self) -> float:
        total_weight = max(self.life_weight + self.res_weight, 1e-6)
        weighted = (self.life_weight * self.life_score) + (self.res_weight * self.res_score)
        return weighted / total_weight


@dataclass
class ActionSpec:
    name: str
    action: CraftingAction


class RingCraftingEnvV1(gym.Env[np.ndarray, int]):
    """Gymnasium-compatible environment for the v1 PoE2 ring crafting PPO spec."""

    metadata = {"render_modes": ["human"], "render_fps": 4}
    _REWARD_COMPONENT_KEYS: Tuple[str, ...] = (
        "general",
        "target_specific",
        "omen",
        "cost",
        "guardrail",
    )

    def __init__(
        self,
        engine: CraftingEngine,
        *,
        base_id: str = "12",
        item_ilvl: int = 80,
        goal: GoalSpec | None = None,
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
        self._used_omens: Set[str] = set()
        self._made_progress: bool = False
        self._active_omen_steps: Dict[str, int] = {}
        self._omens_applied_total = 0
        self._omens_consumed_total = 0
        self._repeat_family: Optional[str] = None
        self._repeat_family_count: int = 0
        self._family_use_counts: Dict[str, int] = {}
        self._deterministic_streak = 0
        self._affix_congestion_steps: Dict[str, int] = {"prefix": 0, "suffix": 0}
        self._affix_churn_scores: Dict[str, int] = {"prefix": 0, "suffix": 0}
        self._last_affix_counts: Dict[str, int] = {"prefix": 0, "suffix": 0}

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
        self._last_score = self._evaluate_goal(self._item).composite_score
        self._prev_action_name = None
        self._reset_repeat_tracker()
        self._used_omens.clear()
        self._made_progress = False
        self._active_omen_steps.clear()
        self._omens_applied_total = 0
        self._omens_consumed_total = 0
        self._family_use_counts.clear()
        self._deterministic_streak = 0
        self._affix_congestion_steps = {"prefix": 0, "suffix": 0}
        self._affix_churn_scores = {"prefix": 0, "suffix": 0}
        self._last_affix_counts = {"prefix": 0, "suffix": 0}
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
        action_family = self._action_family(last_action_name)

        # Handle Stop action explicitly
        cfg = self.reward_config
        components = self._init_reward_components()
        if action_spec.name == STOP_ACTION_NAME:
            eval_result = self._evaluate_goal(self._item)
            if eval_result.success:
                components["general"] += cfg.success_bonus
            else:
                components["guardrail"] -= cfg.stop_fail_penalty
            self._apply_unused_omen_penalty(components)
            terminated = True
            truncated = False
            observation = self._encode_observation(self._item)
            self._last_action_mask = np.zeros(len(self._actions), dtype=np.int8)
            self._prev_action_name = last_action_name
            self._reset_repeat_tracker()
            reward = self._combine_reward_components(components)
            info = {
                "action_mask": self._last_action_mask.copy(),
                "success": eval_result.success,
                "score": eval_result.composite_score,
                "cost": 0.0,
                "last_action_name": last_action_name,
                "omen_consumption_rate": self._omen_consumption_rate(),
                "reward_components": components.copy(),
            }
            return observation, reward, terminated, truncated, info

        # Guard against invalid actions (should be masked, but belt-and-braces)
        if not self._is_action_valid(action_spec.action, self._item, action_spec.name):
            components["guardrail"] -= cfg.stop_fail_penalty
            repeat_count = self._update_repeat_tracker(action_family, made_progress=False)
            if (
                action_family in REPEAT_PENALTY_FAMILIES
                and repeat_count >= cfg.repeat_penalty_streak_threshold
            ):
                streak_excess = repeat_count - cfg.repeat_penalty_streak_threshold + 1
                components["guardrail"] -= cfg.repeat_action_penalty * streak_excess
            observation = self._encode_observation(self._item)
            self._last_action_mask = self._compute_action_mask(self._item)
            self._prev_action_name = last_action_name
            reward = self._combine_reward_components(components)
            info = {
                "action_mask": self._last_action_mask.copy(),
                "invalid_action": True,
                "cost": 0.0,
                "success": False,
                "score": self._last_score,
                "last_action_name": last_action_name,
                "reward_components": components.copy(),
            }
            return observation, reward, False, False, info

        prev_state_sig = self._item_state_signature(self._item)
        prev_rarity = self._item.rarity
        prev_res_ids, prev_res_tiers = self._res_mod_snapshot(self._item)
        prev_active_omens = set(self._item.active_omens)
        prev_affix_counts = {
            "prefix": self._item.num_prefixes,
            "suffix": self._item.num_suffixes,
        }
        prev_structure_quality = self._structure_quality_score(self._item)

        try:
            new_item, cost = self.engine.apply(self._item, action_spec.action)
        except ValueError as exc:
            # Action failed due to unmet hidden condition
            components["guardrail"] -= cfg.stop_fail_penalty
            repeat_count = self._update_repeat_tracker(action_family, made_progress=False)
            if (
                action_family in REPEAT_PENALTY_FAMILIES
                and repeat_count >= cfg.repeat_penalty_streak_threshold
            ):
                streak_excess = repeat_count - cfg.repeat_penalty_streak_threshold + 1
                components["guardrail"] -= cfg.repeat_action_penalty * streak_excess
            observation = self._encode_observation(self._item)
            self._last_action_mask = self._compute_action_mask(self._item)
            self._prev_action_name = last_action_name
            reward = self._combine_reward_components(components)
            info = {
                "action_mask": self._last_action_mask.copy(),
                "invalid_action": True,
                "error": str(exc),
                "cost": 0.0,
                "success": False,
                "score": self._last_score,
                "last_action_name": last_action_name,
                "reward_components": components.copy(),
            }
            return observation, reward, False, False, info

        self._steps += 1
        eval_result = self._evaluate_goal(new_item)
        new_score = eval_result.composite_score
        score_delta = new_score - self._last_score
        shaping = cfg.shaping_coefficient * score_delta
        family_use_count = self._family_use_counts.get(action_family, 0)
        deterministic_roll = self._is_deterministic_roll(last_action_name, prev_active_omens)
        if deterministic_roll:
            self._deterministic_streak += 1
        else:
            self._deterministic_streak = 0
        deterministic_bonus_allowed = (
            deterministic_roll and self._deterministic_streak <= cfg.deterministic_bonus_max_streak
        )
        cost_multiplier = self._cost_multiplier(
            last_action_name,
            action_family,
            family_use_count,
            deterministic_bonus_allowed,
            deterministic_roll,
        )
        currency_penalty = cfg.cost_coefficient * float(cost) * cost_multiplier
        step_penalty = cfg.step_penalty
        progress_reward, progress_stats = self._progress_shaping(prev_res_ids, prev_res_tiers, new_item)
        new_affix_counts = {
            "prefix": new_item.num_prefixes,
            "suffix": new_item.num_suffixes,
        }
        new_structure_quality = self._structure_quality_score(new_item)
        structural_reward = self._structural_progress_reward(
            prev_affix_counts,
            new_affix_counts,
            prev_structure_quality,
            new_structure_quality,
        )
        state_changed = prev_state_sig != self._item_state_signature(new_item)
        current_active_omens = set(new_item.active_omens)
        consumed_omens = prev_active_omens - current_active_omens
        applied_omens = current_active_omens - prev_active_omens
        if applied_omens:
            self._track_omen_applications(applied_omens)
        omen_bonus = self._compute_omen_bonus(consumed_omens, progress_stats)
        components["general"] += shaping + structural_reward
        if deterministic_bonus_allowed:
            components["general"] += cfg.deterministic_roll_bonus
        components["target_specific"] += progress_reward
        components["omen"] += omen_bonus
        components["cost"] -= currency_penalty + step_penalty
        
        # Essence incentives: reward guaranteed/targeted outcomes
        is_essence_action = ESSENCE_KEYWORD in last_action_name
        if is_essence_action:
            # Base bonus for using essence (guaranteed mod outcome)
            components["general"] += cfg.essence_guaranteed_mod_bonus
            # Extra bonus if used early in episode (encourages strategic planning)
            if self._steps <= cfg.essence_early_use_window:
                components["general"] += cfg.essence_early_use_bonus
            # Extra bonus if essence contributes to goal (life/res)
            if progress_stats.made_positive_progress:
                components["target_specific"] += cfg.essence_goal_aligned_bonus
        
        # Chaos spam penalty: discourage pure RNG exploitation
        if action_family == "chaos":
            chaos_uses = self._family_use_counts.get("chaos", 0) + 1
            if chaos_uses > cfg.chaos_spam_threshold:
                excess = chaos_uses - cfg.chaos_spam_threshold
                components["guardrail"] -= cfg.chaos_spam_penalty_ramp * excess
            # Small base penalty for purely RNG actions (incentivize deterministic alternatives)
            if not deterministic_roll:
                components["cost"] -= cfg.rng_action_base_penalty
        
        if state_changed:
            components["general"] += cfg.state_change_reward
        made_progress = (abs(score_delta) >= NO_PROGRESS_EPS) or progress_stats.made_positive_progress
        if made_progress:
            self._made_progress = True
        else:
            components["guardrail"] -= cfg.no_progress_penalty
        repeat_count = self._update_repeat_tracker(action_family, made_progress)
        if (
            action_family in REPEAT_PENALTY_FAMILIES
            and not made_progress
            and repeat_count >= cfg.repeat_penalty_streak_threshold
        ):
            streak_excess = repeat_count - cfg.repeat_penalty_streak_threshold + 1
            components["guardrail"] -= cfg.repeat_action_penalty * streak_excess
        self._family_use_counts[action_family] = family_use_count + 1

        if (
            prev_rarity != "Rare"
            and new_item.rarity == "Rare"
            and not self._made_progress
        ):
            components["guardrail"] -= cfg.early_rare_penalty
        self._apply_slot_efficiency_penalty(new_item, last_action_name, components)
        terminated = False
        truncated = False

        if eval_result.success:
            components["general"] += cfg.success_bonus
            terminated = True
        elif self._is_impossible_state(new_item, eval_result):
            components["guardrail"] -= cfg.impossible_state_penalty
            truncated = True
        elif self._steps >= self.max_steps:
            truncated = True

        if last_action_name in OMEN_ACTION_SEQUENCE:
            self._used_omens.add(last_action_name)

        self._item = new_item
        self._last_score = new_score
        observation = self._encode_observation(new_item)
        self._last_action_mask = self._compute_action_mask(new_item)
        if terminated or truncated:
            self._apply_unused_omen_penalty(components, active_count=len(new_item.active_omens))
            self._reset_repeat_tracker()
        reward = self._combine_reward_components(components)
        self._prev_action_name = last_action_name
        info = {
            "action_mask": self._last_action_mask.copy(),
            "success": eval_result.success,
            "score": new_score,
            "cost": cost,
            "last_action_name": last_action_name,
            "omens_consumed": len(consumed_omens),
            "omen_consumption_rate": self._omen_consumption_rate(),
            "reward_components": components.copy(),
        }
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
        self._repeat_family = None
        self._repeat_family_count = 0

    def _update_repeat_tracker(self, action_family: str, made_progress: bool) -> int:
        if made_progress or action_family != self._repeat_family:
            self._repeat_family = action_family
            self._repeat_family_count = 1
        else:
            self._repeat_family_count += 1
        return self._repeat_family_count

    def set_base(self, base_id: str) -> None:
        if base_id not in self.engine.db.bases_by_id:
            raise ValueError(f"Unknown base_id '{base_id}' for this database")
        self.base_id = base_id

    def _is_impossible_state(self, item: Item, eval_result: GoalEvalResult) -> bool:
        """Return True when the item can no longer meet the configured goal."""
        if self.goal.res_required_count <= 0:
            return False
        if item.num_suffixes >= 3 and eval_result.res_count < self.goal.res_required_count:
            return True
        return False

    def _build_actions(self) -> List[ActionSpec]:
        action_lookup = {action.name: action for action in ACTIONS}
        essence_lookup = {a.name: a for a in create_essence_actions(self.engine.db)}
        
        # Determine essence sequence: either static or dynamic from database
        if DYNAMIC_ESSENCE_ACTIONS:
            # Include all available essences from the engine's database
            essence_names = sorted(essence_lookup.keys())
        else:
            # Use the static sequence (for backward compatibility)
            essence_names = list(ESSENCE_ACTION_SEQUENCE)
        
        ordered_names: List[str] = [
            *BASE_ACTION_SEQUENCE,
            *essence_names,
            *OMEN_ACTION_SEQUENCE,
            STOP_ACTION_NAME,
        ]
        action_specs: List[ActionSpec] = []
        for name in ordered_names:
            action = action_lookup.get(name) or essence_lookup.get(name)
            if action is None:
                # Skip missing actions in dynamic mode, raise in static mode
                if DYNAMIC_ESSENCE_ACTIONS and name in essence_names:
                    continue  # Essence not available for this item class
                raise KeyError(f"Crafting action '{name}' is not available in the current database")
            action_specs.append(ActionSpec(name=name, action=action))
        return action_specs

    def _init_reward_components(self) -> Dict[str, float]:
        return {key: 0.0 for key in self._REWARD_COMPONENT_KEYS}

    def _combine_reward_components(self, components: Mapping[str, float]) -> float:
        channels = self.reward_config.channels
        return (
            channels.general * components["general"]
            + channels.target_specific * components["target_specific"]
            + channels.omen * components["omen"]
            + channels.cost * components["cost"]
            + channels.guardrail * components["guardrail"]
        )

    def _compute_action_mask(self, item: Item) -> np.ndarray:
        mask = np.zeros(len(self._actions), dtype=np.int8)
        for idx, spec in enumerate(self._actions):
            if spec.name == STOP_ACTION_NAME:
                mask[idx] = 1
            elif self._is_action_valid(spec.action, item, spec.name):
                mask[idx] = 1
        return mask

    def _action_family(self, action_name: str) -> str:
        if action_name == STOP_ACTION_NAME:
            return "stop"
        if action_name in CHAOS_ACTIONS:
            return "chaos"
        if action_name in EXALTED_ACTIONS:
            return "exalt"
        if action_name in ANNULMENT_ACTIONS:
            return "annul"
        if action_name in REGAL_ACTIONS:
            return "regal"
        if action_name in OMEN_ACTION_SEQUENCE or action_name.startswith("Omen"):
            return "omen"
        if ESSENCE_KEYWORD in action_name:
            return "essence"
        if "Alchemy" in action_name:
            return "alchemy"
        if "Augmentation" in action_name or "Transmutation" in action_name:
            return "magic_upgrade"
        if "Scouring" in action_name or "Chance" in action_name:
            return "reset"
        return action_name.lower()

    def _track_omen_applications(self, applied: Set[str]) -> None:
        for omen in applied:
            self._active_omen_steps[omen] = self._steps
            self._omens_applied_total += 1

    def _compute_omen_bonus(self, consumed: Set[str], progress_stats: ProgressStats) -> float:
        if not consumed:
            return 0.0
        cfg = self.reward_config
        bonus = cfg.omen_consumption_bonus * len(consumed)
        if progress_stats.made_positive_progress:
            bonus += cfg.omen_progress_bonus * len(consumed)
        for omen in consumed:
            self._omens_consumed_total += 1
            applied_step = self._active_omen_steps.pop(omen, None)
            if applied_step is None:
                continue
            lag = max(1, self._steps - applied_step)
            if lag <= cfg.omen_timely_window:
                bonus += cfg.omen_timely_consumption_bonus
        return bonus

    def _apply_unused_omen_penalty(
        self,
        components: Dict[str, float],
        *,
        active_count: Optional[int] = None,
    ) -> None:
        cfg = self.reward_config
        if active_count is None:
            active_count = len(self._item.active_omens) if self._item else 0
        total_applied = self._omens_applied_total
        if total_applied <= 0:
            if active_count > 0:
                components["omen"] -= cfg.unconsumed_omen_penalty * active_count
            return
        unused = max(total_applied - self._omens_consumed_total, active_count)
        if unused <= 0:
            return
        unused_fraction = unused / total_applied
        components["omen"] -= cfg.unconsumed_omen_penalty * unused_fraction

    def _omen_consumption_rate(self) -> float:
        if self._omens_applied_total == 0:
            return 0.0
        return self._omens_consumed_total / self._omens_applied_total

    def _is_deterministic_roll(self, action_name: str, prev_active_omens: Set[str]) -> bool:
        if ESSENCE_KEYWORD in action_name:
            return True
        if action_name in EXALTED_ACTIONS and prev_active_omens & EXALTATION_OMENS:
            return True
        if action_name in REGAL_ACTIONS and prev_active_omens & REGAL_OMENS:
            return True
        if action_name in CHAOS_ACTIONS and prev_active_omens & CHAOS_OMENS:
            return True
        if action_name in ANNULMENT_ACTIONS and prev_active_omens & ANNULMENT_OMENS:
            return True
        return False

    def _cost_multiplier(
        self,
        action_name: str,
        action_family: str,
        family_use_count: int,
        deterministic_discount_allowed: bool,
        deterministic_roll: bool,
    ) -> float:
        cfg = self.reward_config
        multiplier = 1.0
        if deterministic_roll and deterministic_discount_allowed:
            multiplier *= cfg.deterministic_cost_discount
        free_uses = max(0, cfg.family_cost_free_uses)
        if family_use_count >= free_uses:
            # family_use_count reflects prior uses; include current action in the surcharge
            overage = family_use_count - free_uses + 1
            multiplier *= 1.0 + max(0.0, cfg.family_cost_ramp) * overage
        return multiplier

    def _apply_slot_efficiency_penalty(
        self,
        item: Item,
        action_name: str,
        components: Dict[str, float],
    ) -> None:
        cfg = self.reward_config
        resets_affixes = action_name in SLOT_PHASE_RESET_ACTIONS or ESSENCE_KEYWORD in action_name
        counts = {
            "prefix": item.num_prefixes,
            "suffix": item.num_suffixes,
        }
        for kind, count in counts.items():
            max_slots = 3
            if resets_affixes:
                self._affix_congestion_steps[kind] = 0
            if count < max_slots:
                self._affix_congestion_steps[kind] = 0
            else:
                self._affix_congestion_steps[kind] += 1
                if self._affix_congestion_steps[kind] > cfg.slot_congestion_grace_steps:
                    excess = self._affix_congestion_steps[kind] - cfg.slot_congestion_grace_steps
                    growth = 1.0 + (cfg.slot_congestion_growth * excess)
                    penalty = cfg.slot_congestion_penalty * growth
                    components["guardrail"] -= penalty

            prev_count = self._last_affix_counts[kind]
            if count == max_slots and prev_count < max_slots:
                self._affix_churn_scores[kind] += 1
                if self._affix_churn_scores[kind] >= cfg.slot_churn_threshold:
                    churn_penalty = cfg.slot_churn_penalty * self._affix_churn_scores[kind]
                    components["guardrail"] -= churn_penalty
            else:
                self._affix_churn_scores[kind] = max(0, self._affix_churn_scores[kind] - 1)
            self._last_affix_counts[kind] = count

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

    def _evaluate_goal(self, item: Item) -> GoalEvalResult:
        life_count = self._count_matching_mods(item.prefix_ids, LIFE_GROUPS, self.goal.life_max_tier)
        res_count = self._count_matching_mods(item.suffix_ids, RES_GROUPS, self.goal.res_max_tier)
        life_met = life_count >= self.goal.life_required_count
        res_met = res_count >= self.goal.res_required_count
        if self.goal.life_required_count > 0:
            life_score = min(life_count / self.goal.life_required_count, 1.0)
        else:
            life_score = 1.0

        if self.goal.res_required_count > 0:
            res_progress = min(res_count / self.goal.res_required_count, 1.0)
            res_quality = self._res_quality_score(item.suffix_ids)
            res_score = 0.5 * (res_progress + res_quality)
        else:
            res_quality = 1.0
            res_score = 1.0
        return GoalEvalResult(
            life_count=life_count,
            res_count=res_count,
            life_met=life_met,
            res_met=res_met,
            life_score=life_score,
            res_score=res_score,
            res_quality_score=res_quality,
            life_weight=self.goal.life_weight,
            res_weight=self.goal.res_weight,
        )

    def _count_matching_mods(self, mod_ids: Iterable[str], groups: Sequence[str], tier_limit: int) -> int:
        group_set = set(groups)
        count = 0
        for mod_id in mod_ids:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod and mod.group in group_set and self._resolve_mod_tier(mod) <= tier_limit:
                count += 1
        return count

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

    def _res_quality_score(self, mod_ids: Iterable[str]) -> float:
        total = 0.0
        count = 0
        for mod_id in mod_ids:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod and mod.group in RES_GROUPS:
                total += self._normalise_tier(self._resolve_mod_tier(mod))
                count += 1
        if count == 0:
            return 0.0
        return min(total / 3.0, 1.0)

    def _structure_quality_score(self, item: Item) -> float:
        total = 0.0
        count = 0
        for mod_id in item.prefix_ids + item.suffix_ids:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod is None:
                continue
            total += self._normalise_tier(self._resolve_mod_tier(mod))
            count += 1
        if count == 0:
            return 0.0
        return total / count

    def _has_fractured_priority_mod(self, item: Item) -> bool:
        for mod_id in item.fractured_mods:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod and (mod.group in LIFE_GROUPS or mod.group in RES_GROUPS):
                return True
        return False

    def _res_mod_snapshot(self, item: Item) -> Tuple[Set[str], Dict[str, float]]:
        mod_ids: Set[str] = set()
        best_tiers: Dict[str, float] = {}
        for mod_id in item.suffix_ids:
            mod = self.engine.db.mods_by_id.get(mod_id)
            if mod and mod.group in RES_GROUPS:
                mod_ids.add(mod_id)
                tier_value = self._resolve_mod_tier(mod)
                current_best = best_tiers.get(mod.group)
                if current_best is None or tier_value < current_best:
                    best_tiers[mod.group] = tier_value
        return mod_ids, best_tiers

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
        prev_res_ids: Set[str],
        prev_res_tiers: Dict[str, float],
        item: Item,
    ) -> Tuple[float, ProgressStats]:
        """Reward incremental progress events while capturing granular stats."""

        current_ids, current_tiers = self._res_mod_snapshot(item)
        cfg = self.reward_config
        reward = 0.0

        prev_count = len(prev_res_ids)
        current_count = len(current_ids)

        new_res_mods = max(0, current_count - prev_count)
        if new_res_mods:
            reward += cfg.new_res_mod_reward * new_res_mods

        lost_res_mods = max(0, prev_count - current_count)
        if lost_res_mods:
            reward -= cfg.res_loss_penalty * lost_res_mods

        tier_gain = 0.0
        tier_loss = 0.0
        for group, new_tier in current_tiers.items():
            prev_tier = prev_res_tiers.get(group)
            if prev_tier is None:
                continue
            if new_tier < prev_tier:
                tier_gain += prev_tier - new_tier
            elif new_tier > prev_tier:
                tier_loss += new_tier - prev_tier

        if tier_gain:
            reward += cfg.res_tier_improvement_reward * tier_gain
        if tier_loss:
            reward -= cfg.res_tier_loss_penalty * tier_loss

        milestone_bonus = 0.0
        for idx, bonus in enumerate(cfg.res_milestone_bonuses, start=1):
            if prev_count < idx <= current_count:
                milestone_bonus += bonus
        reward += milestone_bonus

        stats = ProgressStats(
            prev_count=prev_count,
            current_count=current_count,
            new_res_mods=new_res_mods,
            lost_res_mods=lost_res_mods,
            tier_gain=tier_gain,
            tier_loss=tier_loss,
            milestone_bonus=milestone_bonus,
        )
        return reward, stats

    def _structural_progress_reward(
        self,
        prev_affix_counts: Mapping[str, int],
        new_affix_counts: Mapping[str, int],
        prev_quality: float,
        new_quality: float,
    ) -> float:
        cfg = self.reward_config
        reward = 0.0
        prev_total = prev_affix_counts["prefix"] + prev_affix_counts["suffix"]
        new_total = new_affix_counts["prefix"] + new_affix_counts["suffix"]
        delta = new_total - prev_total
        if delta > 0:
            reward += cfg.structural_affix_gain_reward * delta
        elif delta < 0:
            reward -= cfg.structural_affix_loss_penalty * (-delta)

        quality_delta = new_quality - prev_quality
        if quality_delta > 0:
            reward += cfg.structural_quality_reward * quality_delta
        elif quality_delta < 0:
            reward -= cfg.structural_quality_penalty * (-quality_delta)

        prev_balance = abs(prev_affix_counts["prefix"] - prev_affix_counts["suffix"])
        new_balance = abs(new_affix_counts["prefix"] - new_affix_counts["suffix"])
        if new_balance > prev_balance:
            reward -= cfg.structural_balance_penalty * (new_balance - prev_balance)
        return reward


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
        if key == "channels" and isinstance(value, Mapping):
            value = RewardChannelWeights(**value)
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
    "RewardChannelWeights",
    "RewardConfig",
    "ProgressStats",
    "reward_config_to_dict",
    "reward_config_from_dict",
]
