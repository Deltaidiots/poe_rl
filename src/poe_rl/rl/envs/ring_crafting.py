from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ...data.models import Item
from ...engine.actions import (
    ACTIONS,
    CraftingAction,
    RarityReq,
    RemoveAllMods,
    SetRarity,
    AddModGroup,
    AddRandomMod,
    get_action_by_name,
    create_essence_actions,
)
from ...engine.core import CraftingEngine


@dataclass
class RingState:
    """Represents the simplified state of a ring for RL purposes.

    We track the counts of attack and resistance mods and the number
    of open prefix and suffix slots.  Additional features could be
    added here (e.g. counts of other mod types or mod tiers), but
    these four integers allow a compact representation for a broad
    range of crafting goals.
    """

    attack_count: int
    res_count: int
    open_prefixes: int
    open_suffixes: int

    def to_index(self) -> int:
        """Convert this state into an integer index for Q‑learning.

        The space is limited by the maximum number of prefixes/suffixes
        (3 each) and desired counts (0–3).  We map (attack, res,
        open_prefixes, open_suffixes) into a unique integer in
        [0, 4*4*4*4) = 256.
        """
        # Bound counts within 0–3 for indexing
        a = max(0, min(3, self.attack_count))
        r = max(0, min(3, self.res_count))
        op = max(0, min(3, self.open_prefixes))
        os = max(0, min(3, self.open_suffixes))
        return ((a * 4 + r) * 4 + op) * 4 + os

    @classmethod
    def from_item(cls, item: Item) -> "RingState":
        """Construct the RingState from a full Item instance.

        We count attack and resistance mods by examining each mod ID
        attached to the item and checking if its group contains
        "attack" or "resistance".  The number of open prefixes and
        suffixes is derived from the base's maxima minus the counts.
        """
        attack_count = 0
        res_count = 0
        # Note: ItemBase doesn't have max_prefixes/suffixes fields in the current model.
        # We assume 3/3 as standard for PoE rares.
        # If ItemBase is updated to include these, we should use them.
        # For now, hardcode 3.
        max_prefixes = 3
        max_suffixes = 3
        
        for mod_id in item.prefix_ids + item.suffix_ids:
            if "attack" in mod_id.lower():
                attack_count += 1
            if "res" in mod_id.lower() or "resistance" in mod_id.lower():
                res_count += 1
        
        open_prefixes = max(0, max_prefixes - len(item.prefix_ids))
        open_suffixes = max(0, max_suffixes - len(item.suffix_ids))
        return cls(attack_count, res_count, open_prefixes, open_suffixes)

    def __str__(self) -> str:
        return f"A:{self.attack_count} R:{self.res_count} OP:{self.open_prefixes} OS:{self.open_suffixes}"


class RingCraftEnv:
    """Reinforcement learning environment for crafting rings with attack/res goals.

    The environment uses the `CraftingEngine` to simulate the effects of
    currency actions.  It defines a discrete state space and a finite
    action set.  The agent's objective is to craft a ring with at
    least three attack and three resistance mods (tier ≥ 3).  The
    environment provides rewards based on currency costs and the
    attainment of the goal.
    """

    def __init__(
        self,
        engine: CraftingEngine,
        attack_group: str,
        res_group: str,
        target_tier_min: int = 3,
        max_steps: int = 20,
    ) -> None:
        self.engine = engine
        self.attack_group = attack_group
        self.res_group = res_group
        self.target_tier_min = target_tier_min
        self.max_steps = max_steps
        
        # Build action list
        self.actions: List[CraftingAction] = []
        
        # Add all standard actions (Currencies, Omens)
        self.actions.extend(ACTIONS)
        
        # Add all Essence actions
        essence_actions = create_essence_actions(engine.db)
        self.actions.extend(essence_actions)
            
        self.reset()

    def reset(self) -> RingState:
        """Reset the environment to the initial state (white ring)."""
        # Create a new base ring: we pick the first ring base from the engine's DB
        # Check for item_class "1" (Ring ID) or tag "ring"
        bases = [b for b in self.engine.db.bases_by_id.values() if b.item_class == "1" or "ring" in b.tags]
        if not bases:
            raise ValueError("No ring bases available in the crafting database")
        base = bases[0]
        self.item = Item(base=base, ilvl=86)  # high ilvl to unlock all tiers
        self.steps = 0
        # Reset engine's random seed if needed
        return RingState.from_item(self.item)

    def step(self, action_idx: int) -> Tuple[RingState, float, bool]:
        """Apply an action and return the next state, reward, done flag."""
        if action_idx < 0 or action_idx >= len(self.actions):
            raise ValueError(f"Invalid action index {action_idx}")
        action = self.actions[action_idx]
        # If stop action: mark done and compute final reward
        if action.name == "Stop":
            # Evaluate success
            state = RingState.from_item(self.item)
            done = True
            success = state.attack_count >= 3 and state.res_count >= 3
            reward = 0.0
            if success:
                reward += 500.0
            else:
                reward -= 20.0  # penalty for stopping without success
            return state, reward, True
        # Otherwise apply currency and update item
        try:
            new_item, cost = self.engine.apply(self.item, action)
        except ValueError:
            # Invalid action (e.g. wrong rarity)
            return RingState.from_item(self.item), -10.0, False
            
        self.item = new_item
        self.steps += 1
        state = RingState.from_item(self.item)
        # Compute reward: negative currency cost plus progress bonus
        reward = -cost
        # Reward for each additional desired mod of tier ≥ target_tier_min
        # We check newly added mods only by comparing counts before/after
        # For simplicity, we use state counts only (not tier).  Tier filtering is complex; here we assume all mods meet tier requirement.
        # You can extend this by storing previous state and comparing.
        # Check success within steps
        done = False
        if state.attack_count >= 3 and state.res_count >= 3:
            reward += 500.0
            done = True
        # Terminate if max steps
        if self.steps >= self.max_steps:
            done = True
        return state, reward, done

    @property
    def num_states(self) -> int:
        return 4 * 4 * 4 * 4  # attack (0-3), res (0-3), open prefix (0-3), open suffix (0-3)

    @property
    def num_actions(self) -> int:
        return len(self.actions)
