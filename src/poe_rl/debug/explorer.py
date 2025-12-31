"""
Data exploration utilities for understanding PoE2 crafting data.

This module provides a unified interface for querying mods, essences,
item bases, and understanding the ID format conventions used throughout
the codebase.

ID Format Reference:
-------------------
- Mod IDs: "{base_mod_id}_{ItemClass}_{tier}" e.g., "5039_Ring_1"
- Essence IDs: Numeric strings e.g., "3187"  
- Base IDs: Numeric strings e.g., "12" (Pearl Ring)
- Item Class IDs: Numeric "1" (Ring), "2" (Amulet), etc.
- Item Class Names: "Ring", "Amulet", etc.

Example Usage:
-------------
    from poe_rl.debug import DataExplorer
    from poe_rl.data.loader import load_mods, load_essences, load_item_bases

    # Load data
    DATA_PATH = "src/poe_rl/data/static/poec_data.json"
    mods = load_mods(DATA_PATH)
    essences = load_essences(DATA_PATH)
    bases = load_item_bases(DATA_PATH)

    # Create explorer
    explorer = DataExplorer(mods, essences, bases)

    # Find essences by name
    life_essences = explorer.find_essences("life")

    # Get all mods for an essence
    essence = explorer.get_essence("3187")
    ring_mods = explorer.get_essence_mods_for_class(essence, "Ring")

    # Find mods by stat type
    res_mods = explorer.find_mods_by_stat("resistance")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator
from collections import defaultdict

from ..data.models import Mod, Essence, ItemBase


# Item class ID to Name mapping (from PoE2 / Craft of Exile)
ITEM_CLASS_MAP = {
    "1": "Ring",
    "2": "Amulet",
    "3": "Belt",
    "4": "Boots",
    "5": "Gloves",
    "6": "Helmet",
    "7": "Body Armour",
    "8": "Shield",
    "9": "Quiver",
    "10": "Staff",
    "11": "Wand",
    "12": "Dagger",
    "13": "Claw",
    "14": "Sword",
    "15": "Axe",
    "16": "Mace",
    "17": "Bow",
    "18": "Flask",
}

ITEM_NAME_TO_CLASS_ID = {v: k for k, v in ITEM_CLASS_MAP.items()}


@dataclass
class ModMatch:
    """Result of a mod search with context about the match."""
    mod: Mod
    match_reason: str


@dataclass
class EssenceMatch:
    """Result of an essence search with context."""
    essence: Essence
    tier: str  # "lesser", "greater", or "perfect"


class DataExplorer:
    """
    Unified interface for exploring PoE2 crafting data.
    
    Provides search and lookup utilities for mods, essences, and item bases
    with helpful context about ID formats and relationships.
    """
    
    def __init__(
        self,
        mods: List[Mod],
        essences: List[Essence],
        bases: List[ItemBase],
    ):
        """
        Initialize the explorer with loaded data.
        
        Args:
            mods: List of Mod objects from load_mods()
            essences: List of Essence objects from load_essences()
            bases: List of ItemBase objects from load_item_bases()
        """
        # Build lookup indices
        self._mods_by_id: Dict[str, Mod] = {m.mod_id: m for m in mods}
        self._essences_by_id: Dict[str, Essence] = {e.essence_id: e for e in essences}
        self._essences_by_name: Dict[str, Essence] = {e.name.lower(): e for e in essences}
        self._bases_by_id: Dict[str, ItemBase] = {b.base_id: b for b in bases}
        self._bases_by_name: Dict[str, ItemBase] = {}
        for b in bases:
            self._bases_by_name[b.name.lower()] = b
        
        # Group mods by base mod ID (prefix before _ItemClass_tier)
        self._mods_by_base_id: Dict[str, List[Mod]] = defaultdict(list)
        for mod in mods:
            base_mod_id = self._extract_base_mod_id(mod.mod_id)
            self._mods_by_base_id[base_mod_id].append(mod)
        
        # Build essence tier groupings
        self._essence_tiers = self._build_essence_tiers()
    
    def _extract_base_mod_id(self, mod_id: str) -> str:
        """
        Extract the base mod ID from a fully qualified mod ID.
        
        "5039_Ring_1" -> "5039"
        "4823_Amulet_3" -> "4823"
        """
        parts = mod_id.split("_")
        if len(parts) >= 1:
            return parts[0]
        return mod_id
    
    def _build_essence_tiers(self) -> Dict[str, str]:
        """
        Build a mapping of essence_id -> tier name.
        
        Tier detection is based on name patterns:
        - "Lesser Essence of X" -> "lesser"
        - "Greater Essence of X" -> "greater"
        - "Perfect Essence of X" or "Essence of X" (no prefix) -> "perfect"
        """
        tiers = {}
        for eid, e in self._essences_by_id.items():
            name_lower = e.name.lower()
            if name_lower.startswith("lesser"):
                tiers[eid] = "lesser"
            elif name_lower.startswith("greater"):
                tiers[eid] = "greater"
            else:
                tiers[eid] = "perfect"
        return tiers
    
    # =========================================================================
    # Item Class Utilities
    # =========================================================================
    
    def class_id_to_name(self, class_id: str) -> str:
        """Convert item class ID to name. '1' -> 'Ring'"""
        return ITEM_CLASS_MAP.get(class_id, f"Unknown({class_id})")
    
    def class_name_to_id(self, class_name: str) -> Optional[str]:
        """Convert item class name to ID. 'Ring' -> '1'"""
        return ITEM_NAME_TO_CLASS_ID.get(class_name)
    
    # =========================================================================
    # Essence Lookups
    # =========================================================================
    
    def get_essence(self, essence_id: str) -> Optional[Essence]:
        """Get an essence by its ID."""
        return self._essences_by_id.get(essence_id)
    
    def get_essence_by_name(self, name: str) -> Optional[Essence]:
        """Get an essence by exact name (case-insensitive)."""
        return self._essences_by_name.get(name.lower())
    
    def find_essences(self, query: str) -> List[EssenceMatch]:
        """
        Find essences matching a query string.
        
        Args:
            query: Partial name to search for (case-insensitive)
        
        Returns:
            List of matching essences with their tier classification
        """
        query_lower = query.lower()
        matches = []
        for e in self._essences_by_id.values():
            if query_lower in e.name.lower():
                tier = self._essence_tiers.get(e.essence_id, "unknown")
                matches.append(EssenceMatch(essence=e, tier=tier))
        return sorted(matches, key=lambda m: m.essence.name)
    
    def get_essence_tier(self, essence_id: str) -> str:
        """Get the tier of an essence: 'lesser', 'greater', or 'perfect'."""
        return self._essence_tiers.get(essence_id, "unknown")
    
    def list_essences_by_tier(self, tier: str = "all") -> List[Essence]:
        """
        List all essences, optionally filtered by tier.
        
        Args:
            tier: "lesser", "greater", "perfect", or "all"
        """
        if tier == "all":
            return list(self._essences_by_id.values())
        return [
            e for eid, e in self._essences_by_id.items()
            if self._essence_tiers.get(eid) == tier
        ]
    
    def get_essence_mods_for_class(
        self,
        essence: Essence,
        item_class: str,
    ) -> List[Mod]:
        """
        Get the guaranteed mods an essence provides for an item class.
        
        This resolves the mod ID lookup issue where essence data uses
        base mod IDs ("5039") but our mod database uses qualified IDs
        ("5039_Ring_1").
        
        Args:
            essence: The essence to check
            item_class: Item class name ("Ring") or ID ("1")
        
        Returns:
            List of Mod objects the essence can add to this item class
        """
        # Normalize to class ID
        if item_class in ITEM_NAME_TO_CLASS_ID:
            class_id = ITEM_NAME_TO_CLASS_ID[item_class]
        else:
            class_id = item_class
        
        base_mod_ids = essence.guaranteed_mods.get(class_id, [])
        if not base_mod_ids:
            return []
        
        # Look up full mods by base mod ID prefix
        resolved = []
        class_name = ITEM_CLASS_MAP.get(class_id, item_class)
        
        for base_id in base_mod_ids:
            candidates = self._mods_by_base_id.get(base_id, [])
            # Filter to matching item class
            class_mods = [m for m in candidates if class_name in m.item_classes]
            resolved.extend(class_mods)
        
        return resolved
    
    # =========================================================================
    # Mod Lookups
    # =========================================================================
    
    def get_mod(self, mod_id: str) -> Optional[Mod]:
        """Get a mod by its full ID (e.g., '5039_Ring_1')."""
        return self._mods_by_id.get(mod_id)
    
    def get_mods_by_base_id(self, base_mod_id: str) -> List[Mod]:
        """
        Get all mod variants for a base mod ID.
        
        For example, base_mod_id="5039" returns all tiers across all item classes.
        """
        return self._mods_by_base_id.get(base_mod_id, [])
    
    def find_mods(self, query: str, item_class: Optional[str] = None) -> List[ModMatch]:
        """
        Find mods matching a query in their name or group.
        
        Args:
            query: Partial match for mod name or group (case-insensitive)
            item_class: Optional filter by item class
        
        Returns:
            List of matching mods with match context
        """
        query_lower = query.lower()
        matches = []
        
        for mod in self._mods_by_id.values():
            # Filter by item class if specified
            if item_class:
                if item_class not in mod.item_classes:
                    continue
            
            reason = None
            if query_lower in mod.name.lower():
                reason = f"name contains '{query}'"
            elif query_lower in mod.group.lower():
                reason = f"group contains '{query}'"
            
            if reason:
                matches.append(ModMatch(mod=mod, match_reason=reason))
        
        return sorted(matches, key=lambda m: (m.mod.name, m.mod.tier))
    
    def find_mods_by_stat(
        self,
        stat_type: str,
        item_class: Optional[str] = None,
    ) -> List[Mod]:
        """
        Find mods that affect a particular stat type.
        
        Common stat types: "life", "resistance", "mana", "strength", etc.
        
        Args:
            stat_type: Stat keyword to search for
            item_class: Optional filter by item class
        """
        matches = []
        stat_lower = stat_type.lower()
        
        for mod in self._mods_by_id.values():
            if item_class and item_class not in mod.item_classes:
                continue
            
            # Check name/group for stat keywords
            if (stat_lower in mod.name.lower() or 
                stat_lower in mod.group.lower() or
                any(stat_lower in t.lower() for t in mod.tags)):
                matches.append(mod)
        
        return matches
    
    def list_mods_for_class(
        self,
        item_class: str,
        is_prefix: Optional[bool] = None,
    ) -> List[Mod]:
        """
        List all mods available for an item class.
        
        Args:
            item_class: Item class name ("Ring")
            is_prefix: Filter by prefix (True) or suffix (False), or None for all
        """
        mods = []
        for mod in self._mods_by_id.values():
            if item_class not in mod.item_classes:
                continue
            if is_prefix is not None and mod.is_prefix != is_prefix:
                continue
            mods.append(mod)
        return sorted(mods, key=lambda m: (m.name, m.tier))
    
    # =========================================================================
    # Item Base Lookups
    # =========================================================================
    
    def get_base(self, base_id: str) -> Optional[ItemBase]:
        """Get an item base by ID."""
        return self._bases_by_id.get(base_id)
    
    def get_base_by_name(self, name: str) -> Optional[ItemBase]:
        """Get an item base by name (case-insensitive)."""
        return self._bases_by_name.get(name.lower())
    
    def list_bases_for_class(self, item_class: str) -> List[ItemBase]:
        """List all item bases for an item class."""
        return [
            b for b in self._bases_by_id.values()
            if b.item_class == item_class
        ]
    
    # =========================================================================
    # Summary & Reporting
    # =========================================================================
    
    def summary(self) -> Dict[str, int]:
        """Get a summary of loaded data."""
        return {
            "total_mods": len(self._mods_by_id),
            "unique_base_mod_ids": len(self._mods_by_base_id),
            "total_essences": len(self._essences_by_id),
            "lesser_essences": len(self.list_essences_by_tier("lesser")),
            "greater_essences": len(self.list_essences_by_tier("greater")),
            "perfect_essences": len(self.list_essences_by_tier("perfect")),
            "total_bases": len(self._bases_by_id),
        }
    
    def print_essence_info(self, essence_id: str, item_class: str = "Ring") -> None:
        """Print detailed information about an essence."""
        e = self.get_essence(essence_id)
        if not e:
            print(f"Essence not found: {essence_id}")
            return
        
        tier = self.get_essence_tier(essence_id)
        print(f"\n{'='*60}")
        print(f"Essence: {e.name} (ID: {e.essence_id})")
        print(f"Tier: {tier}")
        print(f"{'='*60}")
        
        class_id = self.class_name_to_id(item_class) or item_class
        base_mods = e.guaranteed_mods.get(class_id, [])
        print(f"\nGuaranteed mod IDs for {item_class} (class_id={class_id}): {base_mods}")
        
        resolved = self.get_essence_mods_for_class(e, item_class)
        print(f"\nResolved mods ({len(resolved)} found):")
        for m in resolved:
            print(f"  - {m.mod_id}: {m.name} (T{m.tier}, ilvl {m.ilvl_required})")
            if m.min_value is not None:
                print(f"    Values: {m.min_value} - {m.max_value}")
    
    def print_mod_family(self, base_mod_id: str) -> None:
        """Print all variants of a mod across tiers and item classes."""
        mods = self.get_mods_by_base_id(base_mod_id)
        if not mods:
            print(f"No mods found for base ID: {base_mod_id}")
            return
        
        print(f"\n{'='*60}")
        print(f"Mod Family: {base_mod_id} ({len(mods)} variants)")
        print(f"{'='*60}")
        
        # Group by item class
        by_class: Dict[str, List[Mod]] = defaultdict(list)
        for m in mods:
            for ic in m.item_classes:
                by_class[ic].append(m)
        
        for ic, class_mods in sorted(by_class.items()):
            print(f"\n{ic}:")
            for m in sorted(class_mods, key=lambda x: x.tier):
                print(f"  T{m.tier}: {m.mod_id} (ilvl {m.ilvl_required})")
                if m.min_value is not None:
                    print(f"       Values: {m.min_value} - {m.max_value}")


def create_explorer_from_path(data_path: str) -> DataExplorer:
    """
    Convenience function to create an explorer from a data file path.
    
    Example:
        explorer = create_explorer_from_path("src/poe_rl/data/static/poec_data.json")
    """
    from ..data.loader import load_mods, load_essences, load_item_bases
    
    mods = load_mods(data_path)
    essences = load_essences(data_path)
    bases = load_item_bases(data_path)
    
    return DataExplorer(mods, essences, bases)
