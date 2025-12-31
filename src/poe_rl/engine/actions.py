from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Set, Tuple

from ..data.models import Item, Mod
from .database import CraftingDatabase


ADD_FAILURE_MESSAGE = "No mod available to add in the pool"
ADD_FAILURE_MULTI_MESSAGE = "Could not generate mod"
REMOVE_FAILURE_MESSAGE = "Could not remove a mod"
ESSENCE_CONFLICT_MESSAGE = "The item already has a mod of this type"
ESSENCE_SPACE_MESSAGE = "Could not find space to execute the action"
ESSENCE_UNSUPPORTED_MESSAGE = "This essence could not add any mods to this item"
DIVINE_NO_IMPLICIT_MESSAGE = "No implicit modifiers to reroll"
DIVINE_NO_EXPLICIT_MESSAGE = "No explicit modifiers to reroll"
FRACTURE_EXISTS_MESSAGE = "Item already has a fractured modifier"
FRACTURE_MINIMUM_MESSAGE = "Not enough modifiers to fracture"
FRACTURE_NO_ELIGIBLE_MESSAGE = "No eligible modifier to fracture"

CORRUPTED_ESSENCE_NAMES = {
    "Essence of Horror",
    "Essence of Insanity",
    "Essence of Delirium",
    "Essence of Hysteria",
    "Essence of the Abyss",
}

# --- Helper Functions ---

def check_conflicting_omens(item: Item, omen1: str, omen2: str) -> None:
    if omen1 in item.active_omens and omen2 in item.active_omens:
        raise ValueError(f"Conflicting omens: {omen1} and {omen2}")

def select_mod(
    item: Item,
    db: CraftingDatabase,
    rng: random.Random,
    force_type: Optional[str] = None,  # "Prefix" or "Suffix"
    tags_filter: Optional[Set[str]] = None,
    exclude_groups: Optional[Set[str]] = None,
    min_ilvl: Optional[int] = None,
) -> Optional[Mod]:
    """
    Select a mod from the database based on weights and constraints.
    """
    def filtered_candidates(want_prefix_flag: bool) -> List[Mod]:
        pool = db.get_mod_candidates(item, want_prefix=want_prefix_flag)
        if exclude_groups:
            pool = [m for m in pool if m.group not in exclude_groups]
        if tags_filter:
            pool = [m for m in pool if any(t in tags_filter for t in m.tags)]
        if min_ilvl is not None:
            pool = [m for m in pool if m.ilvl_required >= min_ilvl]
        return pool

    can_prefix = item.has_open_prefix()
    can_suffix = item.has_open_suffix()
    if not can_prefix and not can_suffix:
        return None

    attempt_order: List[bool] = []
    if force_type == "Prefix":
        if not can_prefix:
            return None
        attempt_order = [True]
    elif force_type == "Suffix":
        if not can_suffix:
            return None
        attempt_order = [False]
    else:
        if can_prefix and can_suffix:
            first = rng.choice([True, False])
            attempt_order = [first, not first]
        elif can_prefix:
            attempt_order = [True]
        else:
            attempt_order = [False]

    candidates: List[Mod] = []
    for want_prefix in attempt_order:
        candidates = filtered_candidates(want_prefix)
        if candidates:
            break

    if not candidates:
        return None

    # Calculate weights
    weights = []
    for mod in candidates:
        w = 0
        for tag in item.base.tags:
            w += mod.weight_tags.get(tag, 0)
        if w <= 0:
            w = 0 
        weights.append(w)
    
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    # Weighted Choice
    r = rng.uniform(0, total_weight)
    cumulative = 0.0
    for mod, w in zip(candidates, weights):
        cumulative += w
        if r <= cumulative:
            return mod
    
    return candidates[-1]


# --- Protocols ---

class Effect(Protocol):
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        ...

class Requirement(Protocol):
    def check(self, item: Item) -> bool:
        ...
    
    def error_message(self) -> str:
        ...

# --- Requirements ---

@dataclass
class CorruptedReq:
    def check(self, item: Item) -> bool:
        return not item.corrupted
    
    def error_message(self) -> str:
        return "Item is corrupted"

@dataclass
class RarityReq:
    allowed: Set[str] # "Normal", "Magic", "Rare"

    def check(self, item: Item) -> bool:
        return item.rarity in self.allowed
    
    def error_message(self) -> str:
        return f"Item rarity must be one of: {', '.join(self.allowed)}"

@dataclass
class OpenSlotReq:
    def check(self, item: Item) -> bool:
        return item.has_open_prefix() or item.has_open_suffix()
    
    def error_message(self) -> str:
        return "Item has no open affix slots"

@dataclass
class HasModReq:
    def check(self, item: Item) -> bool:
        return len(item.prefix_ids) > 0 or len(item.suffix_ids) > 0
    
    def error_message(self) -> str:
        return "Item has no modifiers to remove"

@dataclass
class ItemLevelReq:
    min_ilvl: int

    def check(self, item: Item) -> bool:
        return item.ilvl >= self.min_ilvl
    
    def error_message(self) -> str:
        return f"Item level must be at least {self.min_ilvl}"


@dataclass
class UniqueOmenReq:
    omen_name: str

    def check(self, item: Item) -> bool:
        return self.omen_name not in item.active_omens

    def error_message(self) -> str:
        return f"{self.omen_name} is already active"




# --- Effects ---

@dataclass
class SetQuality:
    amount: int

    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        item.quality = max(item.quality, self.amount)

@dataclass
class AddRandomMod:
    count_min: int = 1
    count_max: int = 1
    force_type: Optional[str] = None # "Prefix" or "Suffix"
    min_ilvl: Optional[int] = None
    allowed_omens: List[str] = field(default_factory=list)
    failure_message: Optional[str] = None

    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # Check for conflicts
        if "Omen of Dextral Alchemy" in self.allowed_omens:
             check_conflicting_omens(item, "Omen of Dextral Alchemy", "Omen of Sinistral Alchemy")
        if "Omen of Dextral Exaltation" in self.allowed_omens:
             check_conflicting_omens(item, "Omen of Dextral Exaltation", "Omen of Sinistral Exaltation")
        if "Omen of Dextral Coronation" in self.allowed_omens:
             check_conflicting_omens(item, "Omen of Dextral Coronation", "Omen of Sinistral Coronation")

        # Snapshot state for revert on failure
        original_prefixes = list(item.prefix_ids)
        original_suffixes = list(item.suffix_ids)
        omens_to_remove = []

        count = rng.randint(self.count_min, self.count_max)
        
        # Gather existing groups to avoid duplicates
        used_groups = set()
        for mid in item.prefix_ids + item.suffix_ids:
            m = db.mods_by_id.get(mid)
            if m: used_groups.add(m.group)

        # --- Omen Handling ---
        current_force_type = self.force_type
        tags_filter = None
        alchemy_omen_type = None

        homogenising_tags_cache: Optional[Set[str]] = None

        def get_homogenising_tags() -> Optional[Set[str]]:
            nonlocal homogenising_tags_cache
            if homogenising_tags_cache is None:
                tags: Set[str] = set()
                for mid in original_prefixes + original_suffixes:
                    mod = db.mods_by_id.get(mid)
                    if mod:
                        tags.update(mod.tags)
                homogenising_tags_cache = tags if tags else None
            return homogenising_tags_cache

        # Alchemy Omens
        if self.count_max > 1:
            if "Omen of Dextral Alchemy" in self.allowed_omens and "Omen of Dextral Alchemy" in item.active_omens:
                alchemy_omen_type = "Suffix"
                omens_to_remove.append("Omen of Dextral Alchemy")
            elif "Omen of Sinistral Alchemy" in self.allowed_omens and "Omen of Sinistral Alchemy" in item.active_omens:
                alchemy_omen_type = "Prefix"
                omens_to_remove.append("Omen of Sinistral Alchemy")

        # Exalt/Regal Omens (Single Mod actions)
        if self.count_max == 1:
            # Exaltation
            if "Omen of Dextral Exaltation" in self.allowed_omens and "Omen of Dextral Exaltation" in item.active_omens:
                current_force_type = "Prefix"
                omens_to_remove.append("Omen of Dextral Exaltation")
            elif "Omen of Sinistral Exaltation" in self.allowed_omens and "Omen of Sinistral Exaltation" in item.active_omens:
                current_force_type = "Suffix"
                omens_to_remove.append("Omen of Sinistral Exaltation")
            
            if "Omen of Greater Exaltation" in self.allowed_omens and "Omen of Greater Exaltation" in item.active_omens:
                count = 2
                omens_to_remove.append("Omen of Greater Exaltation")
            
            if "Omen of Homogenising Exaltation" in self.allowed_omens and "Omen of Homogenising Exaltation" in item.active_omens:
                existing_tags = get_homogenising_tags()
                if existing_tags:
                    tags_filter = set(existing_tags)
                omens_to_remove.append("Omen of Homogenising Exaltation")

            # Regal
            if "Omen of Dextral Coronation" in self.allowed_omens and "Omen of Dextral Coronation" in item.active_omens:
                current_force_type = "Suffix"
                omens_to_remove.append("Omen of Dextral Coronation")
            elif "Omen of Sinistral Coronation" in self.allowed_omens and "Omen of Sinistral Coronation" in item.active_omens:
                current_force_type = "Prefix"
                omens_to_remove.append("Omen of Sinistral Coronation")
            
            if "Omen of Homogenising Coronation" in self.allowed_omens and "Omen of Homogenising Coronation" in item.active_omens:
                existing_tags = get_homogenising_tags()
                if existing_tags:
                    tags_filter = set(existing_tags)
                omens_to_remove.append("Omen of Homogenising Coronation")

        mods_added = 0
        for _ in range(count):
            loop_force_type = current_force_type
            
            # Alchemy Omen Logic
            if alchemy_omen_type == "Suffix":
                if len(item.suffix_ids) < 3:
                    loop_force_type = "Suffix"
            elif alchemy_omen_type == "Prefix":
                if len(item.prefix_ids) < 3:
                    loop_force_type = "Prefix"

            mod = select_mod(item, db, rng, force_type=loop_force_type, exclude_groups=used_groups, min_ilvl=self.min_ilvl, tags_filter=tags_filter)
            if mod:
                if mod.is_prefix:
                    if item.has_open_prefix():
                        item.prefix_ids.append(mod.mod_id)
                        used_groups.add(mod.group)
                        mods_added += 1
                    else:
                        break
                else:
                    if item.has_open_suffix():
                        item.suffix_ids.append(mod.mod_id)
                        used_groups.add(mod.group)
                        mods_added += 1
                    else:
                        break
            else:
                break
        
        if mods_added < count:
            # Revert entirely to mirror simulator behaviour when multi-roll omens fail
            item.prefix_ids = original_prefixes
            item.suffix_ids = original_suffixes
            failure_message = self.failure_message
            if failure_message is None:
                failure_message = ADD_FAILURE_MULTI_MESSAGE if count > 1 else ADD_FAILURE_MESSAGE
            raise ValueError(failure_message)

        # Success: Consume omens
        for omen in omens_to_remove:
            if omen in item.active_omens:
                item.active_omens.remove(omen)

@dataclass
class RemoveAllMods:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        protected = set(item.fractured_mods)
        item.prefix_ids = [mid for mid in item.prefix_ids if mid in protected]
        item.suffix_ids = [mid for mid in item.suffix_ids if mid in protected]

@dataclass
class Reforge:
    rarity: str = "Rare"
    min_ilvl: Optional[int] = None
    
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # Handle Omen of Resurgence
        if "Omen of Resurgence" in item.active_omens:
            current_count = len(item.prefix_ids) + len(item.suffix_ids)
            item.active_omens.remove("Omen of Resurgence")
            item.prefix_ids.clear()
            item.suffix_ids.clear()
            item.rarity = self.rarity
            AddRandomMod(count_min=current_count, count_max=current_count, min_ilvl=self.min_ilvl).apply(item, db, rng)
            return

        # Handle Omen of Whittling (Reforge keeping 1 prefix 1 suffix)
        if "Omen of Whittling" in item.active_omens:
            item.active_omens.remove("Omen of Whittling")
            item.rarity = self.rarity
            # Keep 1 prefix 1 suffix if available
            p = rng.choice(item.prefix_ids) if item.prefix_ids else None
            s = rng.choice(item.suffix_ids) if item.suffix_ids else None
            item.prefix_ids.clear()
            item.suffix_ids.clear()
            if p: item.prefix_ids.append(p)
            if s: item.suffix_ids.append(s)
            return

        # Standard Reforge
        item.prefix_ids.clear()
        item.suffix_ids.clear()
        item.rarity = self.rarity
        AddRandomMod(count_min=4, count_max=6, min_ilvl=self.min_ilvl).apply(item, db, rng)

@dataclass
class Corrupt:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        item.corrupted = True
        # Simplified Vaal Orb logic:
        # 25% Brick to Rare (reroll all)
        # 75% No change
        roll = rng.random()
        if roll < 0.25:
            item.rarity = "Rare"
            item.prefix_ids.clear()
            item.suffix_ids.clear()
            # Add 4-6 mods
            count = rng.randint(4, 6)
            AddRandomMod(count_min=count, count_max=count).apply(item, db, rng)

@dataclass
class ChanceOrbEffect:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # 80% Magic, 19% Rare, 1% Unique
        roll = rng.random()
        if roll < 0.80:
            item.rarity = "Magic"
            AddRandomMod(count_min=1, count_max=2).apply(item, db, rng)
        else:
            item.rarity = "Rare"
            AddRandomMod(count_min=4, count_max=6).apply(item, db, rng)

@dataclass
class SetRarity:
    rarity: str

    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        item.rarity = self.rarity

@dataclass
class AddModFromExistingTags:
    """For Omen of Homogenising Exaltation/Coronation"""
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # Gather tags from existing mods
        existing_tags = set()
        for mid in item.prefix_ids + item.suffix_ids:
            m = db.mods_by_id.get(mid)
            if m:
                for tag in m.tags:
                    existing_tags.add(tag)
        
        if not existing_tags:
            return # No tags to match

        # Select a mod that shares at least one tag
        mod = select_mod(item, db, rng, tags_filter=existing_tags)
        if mod:
            if mod.is_prefix and item.has_open_prefix():
                item.prefix_ids.append(mod.mod_id)
            elif not mod.is_prefix and item.has_open_suffix():
                item.suffix_ids.append(mod.mod_id)

@dataclass
class RemoveRandomMod:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        check_conflicting_omens(item, "Omen of Dextral Annulment", "Omen of Sinistral Annulment")

        # Determine candidates pool
        candidates = list(item.prefix_ids + item.suffix_ids)
        omens_to_consume: List[str] = []
        
        # Handle Restriction Omens (Dextral, Sinistral, Light)
        if "Omen of Dextral Annulment" in item.active_omens:
            candidates = list(item.suffix_ids)
            omens_to_consume.append("Omen of Dextral Annulment")
        elif "Omen of Sinistral Annulment" in item.active_omens:
            candidates = list(item.prefix_ids)
            omens_to_consume.append("Omen of Sinistral Annulment")
        elif "Omen of Light" in item.active_omens:
            # Remove Desecrated Mod
            # Assuming they have a "desecrated" tag
            candidates = []
            for mid in item.prefix_ids + item.suffix_ids:
                m = db.mods_by_id.get(mid)
                if m and "desecrated" in m.tags:
                    candidates.append(mid)
            omens_to_consume.append("Omen of Light")

        # Handle Count Omens (Greater Annulment)
        count = 1
        if "Omen of Greater Annulment" in item.active_omens:
            count = 2
            omens_to_consume.append("Omen of Greater Annulment")

        candidates = [mid for mid in candidates if mid not in item.fractured_mods]

        if not candidates or len(candidates) < count:
            raise ValueError(REMOVE_FAILURE_MESSAGE)

        # Remove mods
        for _ in range(count):
            to_remove = rng.choice(candidates)
            
            if to_remove in item.prefix_ids:
                item.prefix_ids.remove(to_remove)
            elif to_remove in item.suffix_ids:
                item.suffix_ids.remove(to_remove)
            
            candidates.remove(to_remove)

        for omen in omens_to_consume:
            if omen in item.active_omens:
                item.active_omens.remove(omen)

@dataclass
class SetOmen:
    omen_name: str
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        item.active_omens.add(self.omen_name)

@dataclass
class AddModGroup:
    group: str
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # Find all mods in this group
        candidates = [m for m in db.mods_by_id.values() if m.group == self.group]
        if not candidates: return
        
        mod = rng.choice(candidates)
        
        if mod.is_prefix:
            if item.has_open_prefix():
                item.prefix_ids.append(mod.mod_id)
        else:
            if item.has_open_suffix():
                item.suffix_ids.append(mod.mod_id)

@dataclass
class AddSpecificMod:
    mod_id: str
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        mod = db.mods_by_id.get(self.mod_id)
        if not mod: return
        
        # Check if group exists
        for mid in item.prefix_ids + item.suffix_ids:
            m = db.mods_by_id.get(mid)
            if m and m.group == mod.group:
                return # Conflict

        if mod.is_prefix and item.has_open_prefix():
            item.prefix_ids.append(mod.mod_id)
        elif not mod.is_prefix and item.has_open_suffix():
            item.suffix_ids.append(mod.mod_id)


# --- Main Action Class ---

@dataclass
class CraftingAction:
    name: str
    currency_key: str
    effects: List[Effect]
    requirements: List[Requirement] = field(default_factory=list)

    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> Item:
        # Check requirements
        for req in self.requirements:
            if not req.check(item):
                raise ValueError(req.error_message())
        
        new_item = item.copy()
        for effect in self.effects:
            effect.apply(new_item, db, rng)
        return new_item



@dataclass
class ScourEffect:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # Handle Omen of Resurgence
        if "Omen of Resurgence" in item.active_omens:
            item.active_omens.remove("Omen of Resurgence")
            # Reroll as Rare with same number of mods (or 4-6?)
            # Reforge logic used current_count. Let's stick to that or 4-6.
            # Usually "Reforge" implies getting a new Rare.
            # Let's assume it acts like a Chaos Orb (PoE 1) / Alchemy.
            # But preserving mod count is a specific behavior.
            # Let's use 4-6 as safe bet for "Rare", or current_count if we want to be fancy.
            # Given "Resurgence", maybe it brings back the item?
            # Let's use 4-6 mods (Alchemy equivalent).
            item.prefix_ids.clear()
            item.suffix_ids.clear()
            item.rarity = "Rare"
            AddRandomMod(count_min=4, count_max=6).apply(item, db, rng)
            return

        # Standard Scour
        protected = set(item.fractured_mods)
        preserved_prefixes = [mid for mid in item.prefix_ids if mid in protected]
        preserved_suffixes = [mid for mid in item.suffix_ids if mid in protected]

        item.prefix_ids = preserved_prefixes
        item.suffix_ids = preserved_suffixes

        prefix_count = len(preserved_prefixes)
        suffix_count = len(preserved_suffixes)
        total = prefix_count + suffix_count

        if total == 0:
            item.rarity = "Normal"
        elif prefix_count > 1 or suffix_count > 1:
            item.rarity = "Rare"
        elif prefix_count == 1 and suffix_count == 1 and protected:
            item.rarity = "Rare"
        else:
            item.rarity = "Magic"


@dataclass
class FractureEffect:
    min_affixes: int = 4

    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        total_mods = item.num_prefixes + item.num_suffixes
        if total_mods < self.min_affixes:
            raise ValueError(FRACTURE_MINIMUM_MESSAGE)

        if item.fractured_mods:
            raise ValueError(FRACTURE_EXISTS_MESSAGE)

        eligible = [mid for mid in item.prefix_ids + item.suffix_ids if mid not in item.fractured_mods]
        if not eligible:
            raise ValueError(FRACTURE_NO_ELIGIBLE_MESSAGE)

        to_fracture = rng.choice(eligible)
        item.fractured_mods.append(to_fracture)

@dataclass
class ChaosOrbEffect:
    min_ilvl: Optional[int] = None

    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        if item.rarity != "Rare":
            return
        
        check_conflicting_omens(item, "Omen of Dextral Erasure", "Omen of Sinistral Erasure")

        # Handle Omens
        remove_candidates = list(item.prefix_ids + item.suffix_ids)
        omens_to_consume: List[str] = []
        
        if "Omen of Dextral Erasure" in item.active_omens:
            remove_candidates = list(item.suffix_ids)
            omens_to_consume.append("Omen of Dextral Erasure")
        elif "Omen of Sinistral Erasure" in item.active_omens:
            remove_candidates = list(item.prefix_ids)
            omens_to_consume.append("Omen of Sinistral Erasure")
        elif "Omen of Whittling" in item.active_omens:
            # Remove lowest level mod
            # We need to look up mods to find their level/req
            # Assuming 'level' or 'ilvl' is available in Mod. 
            # If not, we might pick randomly or use a placeholder.
            # Let's assume we pick randomly for now if we can't determine level, 
            # or try to find the mod with lowest ilvl_required.
            candidates_with_lvl = []
            for mid in remove_candidates:
                m = db.mods_by_id.get(mid)
                if m:
                    candidates_with_lvl.append((mid, m.ilvl_required))
            
            if candidates_with_lvl:
                # Find min level
                min_lvl = min(c[1] for c in candidates_with_lvl)
                # Filter candidates with that level
                lowest_candidates = [c[0] for c in candidates_with_lvl if c[1] == min_lvl]
                remove_candidates = lowest_candidates
            
            omens_to_consume.append("Omen of Whittling")

        remove_candidates = [mid for mid in remove_candidates if mid not in item.fractured_mods]

        if not remove_candidates:
            raise ValueError(REMOVE_FAILURE_MESSAGE)

        to_remove = rng.choice(remove_candidates)
        
        # Check if we can add a mod after removal (to prevent "Remove Only" bug)
        temp_item = item.copy()
        if to_remove in temp_item.prefix_ids:
            temp_item.prefix_ids.remove(to_remove)
        elif to_remove in temp_item.suffix_ids:
            temp_item.suffix_ids.remove(to_remove)
            
        # Check if a valid mod exists for the new state
        check_mod = select_mod(temp_item, db, rng, min_ilvl=self.min_ilvl)
        if not check_mod:
            raise ValueError(ADD_FAILURE_MESSAGE)

        # Proceed with actual removal
        if to_remove in item.prefix_ids:
            item.prefix_ids.remove(to_remove)
        elif to_remove in item.suffix_ids:
            item.suffix_ids.remove(to_remove)
        
        # Add 1 random mod
        AddRandomMod(count_min=1, count_max=1, min_ilvl=self.min_ilvl).apply(item, db, rng)

        for omen in omens_to_consume:
            if omen in item.active_omens:
                item.active_omens.remove(omen)

@dataclass
class EssenceEffect:
    essence_id: str
    
    def _resolve_guaranteed_mod(
        self,
        base_mod_id: str,
        item_class: str,
        db: CraftingDatabase,
        essence_name: str,
    ) -> Optional[Tuple[str, Mod]]:
        """
        Resolve the guaranteed mod ID from the essence data to an actual mod in the database.
        
        The essence's guaranteed_mods contains base mod IDs like '5039', but the database
        keys are composite like '5039_Ring_1' (base_id + item_class + tier).
        
        Tier selection based on essence type:
        - Perfect/Corrupted: Best tier (tier 1)
        - Greater: Mid-high tier (tier 2-3)
        - Normal (no prefix): Mid tier
        - Lesser: Lower tier
        """
        # First try direct lookup (in case the ID already includes class/tier)
        if base_mod_id in db.mods_by_id:
            mod = db.mods_by_id[base_mod_id]
            return (base_mod_id, mod)
        
        # Find all mods matching the base ID prefix for this item class
        # Pattern: {base_mod_id}_{ItemClass}_{tier}
        prefix = f"{base_mod_id}_"
        candidates: List[Tuple[str, Mod]] = []
        for mod_id, mod in db.mods_by_id.items():
            if mod_id.startswith(prefix):
                # Check if this mod applies to rings (or the relevant item class)
                if any(cls.lower() == item_class.lower() for cls in mod.item_classes):
                    candidates.append((mod_id, mod))
                # Also match by item_class name in the mod_id (e.g., "5039_Ring_1")
                elif f"_{item_class}_" in mod_id or mod_id.endswith(f"_{item_class}"):
                    candidates.append((mod_id, mod))
        
        if not candidates:
            # Try looser match - just prefix
            for mod_id, mod in db.mods_by_id.items():
                if mod_id.startswith(prefix):
                    candidates.append((mod_id, mod))
        
        if not candidates:
            return None
        
        # Sort by tier (lower tier number = better)
        candidates.sort(key=lambda x: x[1].tier if x[1].tier is not None else 999)
        
        # Select tier based on essence type
        is_perfect = "Perfect" in essence_name
        is_corrupted = essence_name in CORRUPTED_ESSENCE_NAMES
        is_greater = "Greater" in essence_name
        is_lesser = "Lesser" in essence_name
        
        if is_perfect or is_corrupted:
            # Best tier (first after sorting = lowest tier number)
            return candidates[0]
        elif is_greater:
            # Mid-high tier (second best if available)
            idx = min(1, len(candidates) - 1)
            return candidates[idx]
        elif is_lesser:
            # Lower tier (towards end)
            idx = max(0, len(candidates) - 2)
            return candidates[idx]
        else:
            # Normal essence - mid tier
            idx = len(candidates) // 2
            return candidates[idx]
    
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        if not db.essences_by_id:
            raise ValueError(ESSENCE_UNSUPPORTED_MESSAGE)

        essence = db.essences_by_id.get(self.essence_id)
        if not essence:
            raise ValueError(ESSENCE_UNSUPPORTED_MESSAGE)
            
        is_perfect = "Perfect" in essence.name
        is_corrupted = essence.name in CORRUPTED_ESSENCE_NAMES
        
        # Determine Guaranteed Mod - use item_class_id from the base if available
        # The essence stores guaranteed_mods keyed by item class ID (e.g., "1" for Ring)
        item_class_key = item.base.item_class_id or item.base.item_class
        guaranteed_mods = essence.guaranteed_mods.get(item_class_key)
        
        # Fallback: try the item_class name if ID lookup failed
        if not guaranteed_mods and item.base.item_class_id:
            guaranteed_mods = essence.guaranteed_mods.get(item.base.item_class)
        
        base_mod_id = None
        if guaranteed_mods:
            base_mod_id = guaranteed_mods[0]  # Assume first
        
        if not base_mod_id:
            raise ValueError(ESSENCE_UNSUPPORTED_MESSAGE)
        
        # Resolve the base mod ID to an actual mod in the database
        resolved = self._resolve_guaranteed_mod(
            base_mod_id, 
            item.base.item_class,
            db,
            essence.name,
        )
        if not resolved:
            raise ValueError(ESSENCE_UNSUPPORTED_MESSAGE)
        
        guaranteed_mod_id, guaranteed_mod = resolved

        original_prefixes = list(item.prefix_ids)
        original_suffixes = list(item.suffix_ids)
        original_rarity = item.rarity

        def revert_state() -> None:
            item.prefix_ids = list(original_prefixes)
            item.suffix_ids = list(original_suffixes)
            item.rarity = original_rarity

        def has_group_conflict(group: str) -> bool:
            for mid in item.prefix_ids + item.suffix_ids:
                mod = db.mods_by_id.get(mid)
                if mod and mod.group == group:
                    return True
            return False

        omens_to_consume: List[str] = []

        if is_perfect or is_corrupted:
            # Target: Rare
            if item.rarity != "Rare":
                return 
            
            check_conflicting_omens(item, "Omen of Dextral Crystallisation", "Omen of Sinistral Crystallisation")

            # Handle Omen of Crystallisation
            remove_candidates = list(item.prefix_ids + item.suffix_ids)
            if "Omen of Dextral Crystallisation" in item.active_omens:
                remove_candidates = list(item.suffix_ids)
                omens_to_consume.append("Omen of Dextral Crystallisation")
            elif "Omen of Sinistral Crystallisation" in item.active_omens:
                remove_candidates = list(item.prefix_ids)
                omens_to_consume.append("Omen of Sinistral Crystallisation")
            
            remove_candidates = [mid for mid in remove_candidates if mid not in item.fractured_mods]

            if not remove_candidates:
                raise ValueError(REMOVE_FAILURE_MESSAGE)

            # Remove 1 mod
            to_remove = rng.choice(remove_candidates)
            if to_remove in item.prefix_ids:
                item.prefix_ids.remove(to_remove)
            elif to_remove in item.suffix_ids:
                item.suffix_ids.remove(to_remove)
            
            if has_group_conflict(guaranteed_mod.group):
                revert_state()
                raise ValueError(ESSENCE_CONFLICT_MESSAGE)

            target_has_space = item.has_open_prefix() if guaranteed_mod.is_prefix else item.has_open_suffix()
            if not target_has_space:
                revert_state()
                raise ValueError(ESSENCE_SPACE_MESSAGE)

            if guaranteed_mod.is_prefix:
                item.prefix_ids.append(guaranteed_mod_id)
            else:
                item.suffix_ids.append(guaranteed_mod_id)
                    
        else:
            # Lesser/Greater/Normal
            # Target: Magic only (PoE 2 Rule)
            if item.rarity != "Magic":
                return 
            
            if has_group_conflict(guaranteed_mod.group):
                raise ValueError(ESSENCE_CONFLICT_MESSAGE)

            # Upgrade to Rare, Add Guaranteed (Regal-like)
            item.rarity = "Rare"
            
            target_has_space = item.has_open_prefix() if guaranteed_mod.is_prefix else item.has_open_suffix()
            if not target_has_space:
                revert_state()
                raise ValueError(ESSENCE_SPACE_MESSAGE)

            # Add Guaranteed
            if guaranteed_mod.is_prefix:
                item.prefix_ids.append(guaranteed_mod_id)
            else:
                item.suffix_ids.append(guaranteed_mod_id)

        for omen in omens_to_consume:
            if omen in item.active_omens:
                item.active_omens.remove(omen)


@dataclass
class DivineEffect:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        if "Omen of the Blessed" in item.active_omens:
            if not item.base.implicits:
                raise ValueError(DIVINE_NO_IMPLICIT_MESSAGE)
            item.active_omens.remove("Omen of the Blessed")
            return

        if not item.prefix_ids and not item.suffix_ids:
            raise ValueError(DIVINE_NO_EXPLICIT_MESSAGE)
        # No stat values are tracked in this simulator, so we simply validate that
        # reroll conditions are met without mutating affixes.


@dataclass
class OmenEffect:
    omen_name: str
    
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> None:
        # Placeholder for Omen logic
        pass

# --- Action Registry ---

ACTIONS: List[CraftingAction] = [
    # --- Basics ---
    CraftingAction(
        name="Orb of Transmutation",
        currency_key="Orb of Transmutation",
        requirements=[RarityReq({"Normal"}), CorruptedReq()],
        effects=[SetRarity("Magic"), AddRandomMod(count_min=1, count_max=1)]
    ),
    CraftingAction(
        name="Greater Orb of Transmutation",
        currency_key="Greater Orb of Transmutation",
        requirements=[RarityReq({"Normal"}), CorruptedReq(), ItemLevelReq(55)],
        effects=[SetRarity("Magic"), AddRandomMod(count_min=1, count_max=1, min_ilvl=55)]
    ),
    CraftingAction(
        name="Perfect Orb of Transmutation",
        currency_key="Perfect Orb of Transmutation",
        requirements=[RarityReq({"Normal"}), CorruptedReq(), ItemLevelReq(70)],
        effects=[SetRarity("Magic"), AddRandomMod(count_min=1, count_max=1, min_ilvl=70)]
    ),
    CraftingAction(
        name="Orb of Augmentation",
        currency_key="Orb of Augmentation",
        requirements=[RarityReq({"Magic"}), OpenSlotReq(), CorruptedReq()],
        effects=[AddRandomMod(count_min=1, count_max=1)]
    ),
    CraftingAction(
        name="Greater Orb of Augmentation",
        currency_key="Greater Orb of Augmentation",
        requirements=[RarityReq({"Magic"}), OpenSlotReq(), CorruptedReq(), ItemLevelReq(55)],
        effects=[AddRandomMod(count_min=1, count_max=1, min_ilvl=55)]
    ),
    CraftingAction(
        name="Perfect Orb of Augmentation",
        currency_key="Perfect Orb of Augmentation",
        requirements=[RarityReq({"Magic"}), OpenSlotReq(), CorruptedReq(), ItemLevelReq(70)],
        effects=[AddRandomMod(count_min=1, count_max=1, min_ilvl=70)]
    ),
    CraftingAction(
        name="Orb of Alteration",
        currency_key="Orb of Alteration",
        requirements=[RarityReq({"Magic"}), CorruptedReq()],
        effects=[RemoveAllMods(), AddRandomMod(count_min=1, count_max=1)]
    ),
    CraftingAction(
        name="Regal Orb",
        currency_key="Regal Orb",
        requirements=[RarityReq({"Magic"}), CorruptedReq()],
        effects=[SetRarity("Rare"), AddRandomMod(count_min=1, count_max=1, allowed_omens=[
            "Omen of Homogenising Coronation", "Omen of Dextral Coronation", "Omen of Sinistral Coronation"
        ])]
    ),
    CraftingAction(
        name="Greater Regal Orb",
        currency_key="Greater Regal Orb",
        requirements=[RarityReq({"Magic"}), CorruptedReq(), ItemLevelReq(35)],
        effects=[SetRarity("Rare"), AddRandomMod(count_min=1, count_max=1, min_ilvl=35, allowed_omens=[
            "Omen of Homogenising Coronation", "Omen of Dextral Coronation", "Omen of Sinistral Coronation"
        ])]
    ),
    CraftingAction(
        name="Perfect Regal Orb",
        currency_key="Perfect Regal Orb",
        requirements=[RarityReq({"Magic"}), CorruptedReq(), ItemLevelReq(50)],
        effects=[SetRarity("Rare"), AddRandomMod(count_min=1, count_max=1, min_ilvl=50, allowed_omens=[
            "Omen of Homogenising Coronation", "Omen of Dextral Coronation", "Omen of Sinistral Coronation"
        ])]
    ),
    CraftingAction(
        name="Orb of Alchemy",
        currency_key="Orb of Alchemy",
        requirements=[RarityReq({"Normal", "Magic"}), CorruptedReq()],
        effects=[RemoveAllMods(), SetRarity("Rare"), AddRandomMod(count_min=4, count_max=6, allowed_omens=[
            "Omen of Dextral Alchemy", "Omen of Sinistral Alchemy"
        ])]
    ),
    CraftingAction(
        name="Chaos Orb",
        currency_key="Chaos Orb",
        requirements=[RarityReq({"Rare"}), CorruptedReq()],
        effects=[ChaosOrbEffect()]
    ),
    CraftingAction(
        name="Greater Chaos Orb",
        currency_key="Greater Chaos Orb",
        requirements=[RarityReq({"Rare"}), CorruptedReq(), ItemLevelReq(35)],
        effects=[ChaosOrbEffect(min_ilvl=35)]
    ),
    CraftingAction(
        name="Perfect Chaos Orb",
        currency_key="Perfect Chaos Orb",
        requirements=[RarityReq({"Rare"}), CorruptedReq(), ItemLevelReq(50)],
        effects=[ChaosOrbEffect(min_ilvl=50)]
    ),
    CraftingAction(
        name="Exalted Orb",
        currency_key="Exalted Orb",
        requirements=[RarityReq({"Rare"}), OpenSlotReq(), CorruptedReq()],
        effects=[AddRandomMod(count_min=1, count_max=1, allowed_omens=[
            "Omen of Dextral Exaltation", "Omen of Sinistral Exaltation", 
            "Omen of Greater Exaltation", "Omen of Homogenising Exaltation"
        ])]
    ),
    CraftingAction(
        name="Greater Exalted Orb",
        currency_key="Greater Exalted Orb",
        requirements=[RarityReq({"Rare"}), OpenSlotReq(), CorruptedReq(), ItemLevelReq(35)],
        effects=[AddRandomMod(count_min=1, count_max=1, min_ilvl=35, allowed_omens=[
            "Omen of Dextral Exaltation", "Omen of Sinistral Exaltation", 
            "Omen of Greater Exaltation", "Omen of Homogenising Exaltation"
        ])]
    ),
    CraftingAction(
        name="Perfect Exalted Orb",
        currency_key="Perfect Exalted Orb",
        requirements=[RarityReq({"Rare"}), OpenSlotReq(), CorruptedReq(), ItemLevelReq(50)],
        effects=[AddRandomMod(count_min=1, count_max=1, min_ilvl=50, allowed_omens=[
            "Omen of Dextral Exaltation", "Omen of Sinistral Exaltation", 
            "Omen of Greater Exaltation", "Omen of Homogenising Exaltation"
        ])]
    ),
    CraftingAction(
        name="Orb of Annulment",
        currency_key="Orb of Annulment",
        requirements=[HasModReq(), CorruptedReq()],
        effects=[RemoveRandomMod()]
    ),
    CraftingAction(
        name="Orb of Scouring",
        currency_key="Orb of Scouring",
        requirements=[RarityReq({"Magic", "Rare"}), CorruptedReq()],
        effects=[ScourEffect()]
    ),
    CraftingAction(
        name="Orb of Chance",
        currency_key="Orb of Chance",
        requirements=[RarityReq({"Normal"}), CorruptedReq()],
        effects=[ChanceOrbEffect()]
    ),
    CraftingAction(
        name="Vaal Orb",
        currency_key="Vaal Orb",
        requirements=[CorruptedReq()],
        effects=[Corrupt()]
    ),
    CraftingAction(
        name="Divine Orb",
        currency_key="Divine Orb",
        requirements=[RarityReq({"Magic", "Rare"}), CorruptedReq()],
        effects=[DivineEffect()]
    ),
    CraftingAction(
        name="Fracturing Orb",
        currency_key="Fracturing Orb",
        requirements=[RarityReq({"Rare"}), HasModReq(), CorruptedReq()],
        effects=[FractureEffect()]
    ),
    CraftingAction(
        name="Catalyst",
        currency_key="Adaptive Catalyst",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq()],
        effects=[SetQuality(20)]
    ),
    
    # --- Omens ---
    CraftingAction(
        name="Omen of Dextral Exaltation",
        currency_key="Omen of Dextral Exaltation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Dextral Exaltation")],
        effects=[SetOmen("Omen of Dextral Exaltation")]
    ),
    CraftingAction(
        name="Omen of Sinistral Exaltation",
        currency_key="Omen of Sinistral Exaltation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sinistral Exaltation")],
        effects=[SetOmen("Omen of Sinistral Exaltation")]
    ),
    CraftingAction(
        name="Omen of Greater Exaltation",
        currency_key="Omen of Greater Exaltation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Greater Exaltation")],
        effects=[SetOmen("Omen of Greater Exaltation")]
    ),
    CraftingAction(
        name="Omen of Homogenising Exaltation",
        currency_key="Omen of Homogenising Exaltation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Homogenising Exaltation")],
        effects=[SetOmen("Omen of Homogenising Exaltation")]
    ),
    CraftingAction(
        name="Omen of Dextral Annulment",
        currency_key="Omen of Dextral Annulment",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Dextral Annulment")],
        effects=[SetOmen("Omen of Dextral Annulment")]
    ),
    CraftingAction(
        name="Omen of Sinistral Annulment",
        currency_key="Omen of Sinistral Annulment",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sinistral Annulment")],
        effects=[SetOmen("Omen of Sinistral Annulment")]
    ),
    CraftingAction(
        name="Omen of Greater Annulment",
        currency_key="Omen of Greater Annulment",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Greater Annulment")],
        effects=[SetOmen("Omen of Greater Annulment")]
    ),
    CraftingAction(
        name="Omen of Light",
        currency_key="Omen of Light",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Light")],
        effects=[SetOmen("Omen of Light")]
    ),
    CraftingAction(
        name="Omen of Whittling",
        currency_key="Omen of Whittling",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Whittling")],
        effects=[SetOmen("Omen of Whittling")]
    ),
    CraftingAction(
        name="Omen of Dextral Erasure",
        currency_key="Omen of Dextral Erasure",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Dextral Erasure")],
        effects=[SetOmen("Omen of Dextral Erasure")]
    ),
    CraftingAction(
        name="Omen of Sinistral Erasure",
        currency_key="Omen of Sinistral Erasure",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sinistral Erasure")],
        effects=[SetOmen("Omen of Sinistral Erasure")]
    ),
    CraftingAction(
        name="Omen of Dextral Coronation",
        currency_key="Omen of Dextral Coronation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Dextral Coronation")],
        effects=[SetOmen("Omen of Dextral Coronation")]
    ),
    CraftingAction(
        name="Omen of Sinistral Coronation",
        currency_key="Omen of Sinistral Coronation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sinistral Coronation")],
        effects=[SetOmen("Omen of Sinistral Coronation")]
    ),
    CraftingAction(
        name="Omen of Homogenising Coronation",
        currency_key="Omen of Homogenising Coronation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Homogenising Coronation")],
        effects=[SetOmen("Omen of Homogenising Coronation")]
    ),
    CraftingAction(
        name="Omen of Dextral Crystallisation",
        currency_key="Omen of Dextral Crystallisation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Dextral Crystallisation")],
        effects=[SetOmen("Omen of Dextral Crystallisation")]
    ),
    CraftingAction(
        name="Omen of Sinistral Crystallisation",
        currency_key="Omen of Sinistral Crystallisation",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sinistral Crystallisation")],
        effects=[SetOmen("Omen of Sinistral Crystallisation")]
    ),
    CraftingAction(
        name="Omen of the Blessed",
        currency_key="Omen of the Blessed",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of the Blessed")],
        effects=[SetOmen("Omen of the Blessed")]
    ),
    CraftingAction(
        name="Omen of Dextral Alchemy",
        currency_key="Omen of Dextral Alchemy",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Dextral Alchemy")],
        effects=[SetOmen("Omen of Dextral Alchemy")]
    ),
    CraftingAction(
        name="Omen of Sinistral Alchemy",
        currency_key="Omen of Sinistral Alchemy",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sinistral Alchemy")],
        effects=[SetOmen("Omen of Sinistral Alchemy")]
    ),
    CraftingAction(
        name="Omen of Liege",
        currency_key="Omen of Liege",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Liege")],
        effects=[SetOmen("Omen of Liege")]
    ),
    CraftingAction(
        name="Omen of Sovereign",
        currency_key="Omen of Sovereign",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq(), UniqueOmenReq("Omen of Sovereign")],
        effects=[SetOmen("Omen of Sovereign")]
    ),
    CraftingAction(
        name="Omen of Blackblooded",
        currency_key="Omen of Blackblooded",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq()],
        effects=[SetOmen("Omen of Blackblooded")]
    ),
    CraftingAction(
        name="Omen of Resurgence",
        currency_key="Omen of Resurgence",
        requirements=[RarityReq({"Normal", "Magic", "Rare"}), CorruptedReq()],
        effects=[SetOmen("Omen of Resurgence")]
    ),
    
    # --- Stop ---
    CraftingAction(
        name="Stop",
        currency_key="none",
        requirements=[],
        effects=[]
    )
]

def get_action_by_name(name: str) -> Optional[CraftingAction]:
    for a in ACTIONS:
        if a.name == name:
            return a
    return None

def create_essence_actions(db: CraftingDatabase) -> List[CraftingAction]:
    """
    Generates Essence actions based on available essences in the database.
    """
    actions = []
    if not db.essences_by_id:
        return actions
        
    for essence in db.essences_by_id.values():
        requires_rare = "Perfect" in essence.name or essence.name in CORRUPTED_ESSENCE_NAMES
        rarity_targets = {"Rare"} if requires_rare else {"Magic"}
        actions.append(CraftingAction(
            name=essence.name,
            currency_key=essence.name, # Assuming price key matches name
            requirements=[CorruptedReq(), RarityReq(rarity_targets)], # Enforce rarity semantics
            effects=[
                EssenceEffect(essence_id=essence.essence_id)
            ]
        ))
    return actions

