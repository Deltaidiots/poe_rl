# Data Model Reference

This document describes the data structures and ID formats used in the PoE2 Ring Crafting RL project.

## Overview

The data layer loads crafting information from [Craft of Exile](https://www.craftofexile.com/) exports and normalizes them into Python dataclasses. Understanding the ID conventions is crucial for debugging essence/mod resolution issues.

## ID Format Conventions

### Mod IDs

Mods use a **qualified ID format** that includes the item class and tier:

```
{base_mod_id}_{ItemClass}_{tier}
```

**Examples:**
- `5039_Ring_1` - Lightning Resistance Tier 1 for Rings
- `5039_Ring_8` - Lightning Resistance Tier 8 for Rings
- `4823_Amulet_3` - Some mod Tier 3 for Amulets

**Key Points:**
- Tier 1 is the **best** (highest values, highest ilvl requirement)
- Tier 8 is the **worst** (lowest values, available from ilvl 1)
- Same base mod can exist across multiple item classes with different IDs

### Essence IDs

Essences use **simple numeric IDs**:

```
{numeric_id}
```

**Examples:**
- `3187` - Essence of Grounding
- `3188` - Greater Essence of Grounding
- `3189` - Perfect Essence of Grounding

### Item Class IDs vs Names

There are **two representations** for item classes:

| Class ID | Class Name   |
|----------|--------------|
| 1        | Ring         |
| 2        | Amulet       |
| 3        | Belt         |
| 4        | Boots        |
| 5        | Gloves       |
| 6        | Helmet       |
| 7        | Body Armour  |
| 8        | Shield       |
| 9        | Quiver       |

**Important:** Essences reference classes by **numeric ID** (e.g., `"1"` for Ring), but `ItemBase.item_class` uses **names** (e.g., `"Ring"`). The `item_class_id` field bridges this gap.

### Base Item IDs

Item bases use simple numeric IDs:

```
{numeric_id}
```

**Examples:**
- `12` - Pearl Ring
- `13` - Topaz Ring

## Core Data Structures

### Mod

```python
@dataclass
class Mod:
    mod_id: str              # Qualified ID: "5039_Ring_1"
    name: str                # Display name: "+#% to Lightning Resistance"
    group: str               # Mod group for mutual exclusion
    tier: int                # 1 = best, 8 = worst
    generation_type: str     # "prefix" or "suffix"
    is_prefix: bool          # True for prefixes
    ilvl_required: int       # Minimum item level to roll
    item_classes: List[str]  # ["Ring", "Amulet"]
    min_value: float         # Minimum stat roll
    max_value: float         # Maximum stat roll
    weight_tags: Dict        # Spawn weights per tag
    tags: List[str]          # Mod type tags
```

### Essence

```python
@dataclass
class Essence:
    essence_id: str                           # "3187"
    name: str                                 # "Essence of Grounding"
    guaranteed_mods: Dict[str, List[str]]     # {"1": ["5039"]} - class_id -> base_mod_ids
```

**Note:** `guaranteed_mods` uses:
- Keys: Item class **IDs** (e.g., `"1"` not `"Ring"`)
- Values: **Base mod IDs** (e.g., `"5039"` not `"5039_Ring_1"`)

### ItemBase

```python
@dataclass
class ItemBase:
    base_id: str              # "12"
    name: str                 # "Pearl Ring"
    item_class: str           # "Ring" (name)
    item_class_id: str        # "1" (numeric ID)
    drop_level: int           # 1-85
    tags: List[str]           # Item tags
    implicits: List[str]      # Implicit mod IDs
```

### Item

```python
@dataclass
class Item:
    base: ItemBase
    ilvl: int                 # Item level (affects mod pool)
    prefix_ids: List[str]     # Currently applied prefix mod IDs
    suffix_ids: List[str]     # Currently applied suffix mod IDs
    active_omens: Set[str]    # Active omen effects
    rarity: str               # "Normal", "Magic", "Rare"
    fractured_mods: List[str] # Cannot be removed
```

## Data Flow

```
poec_data.json (Craft of Exile export)
         │
         ▼
    load_json_file()        # Strip wrapper prefix (poecd=)
         │
         ├──► load_mods()          → List[Mod]
         │         Creates qualified IDs: base_mod_id + "_" + class + "_" + tier
         │
         ├──► load_essences()      → List[Essence]
         │         Keeps base mod IDs in guaranteed_mods
         │
         └──► load_item_bases()    → List[ItemBase]
                   Sets both item_class (name) and item_class_id (numeric)
         │
         ▼
    CraftingDatabase
         │
         ├── mods_by_id: Dict[str, Mod]
         ├── essences_by_id: Dict[str, Essence]
         └── bases_by_id: Dict[str, ItemBase]
         │
         ▼
    CraftingEngine
         │
         └── Applies actions, resolves mod lookups
```

## Common Gotchas

### 1. Essence Mod ID Mismatch

**Problem:** Essence data contains base mod IDs (`"5039"`), but the mod database uses qualified IDs (`"5039_Ring_1"`).

**Solution:** Use `_resolve_guaranteed_mod()` in `EssenceEffect` which:
1. Takes base mod ID from essence
2. Finds all mods with that prefix
3. Filters by item class
4. Selects tier based on essence type (lesser/greater/perfect)

### 2. Item Class ID vs Name

**Problem:** `Essence.guaranteed_mods` uses class IDs (`"1"`), but `Item.base.item_class` is a name (`"Ring"`).

**Solution:** `ItemBase` now has both:
- `item_class`: The name ("Ring")
- `item_class_id`: The numeric ID ("1")

Use `item_class_id` for essence mod lookups.

### 3. Tier Numbering

**Problem:** Higher tier numbers are worse, not better.

**Convention:**
- Tier 1 = Best (T1 life = +70-79 life)
- Tier 8 = Worst (T8 life = +3-9 life)

When selecting essence mods:
- Perfect Essence → Tier 1-3
- Greater Essence → Tier 3-5
- Lesser Essence → Tier 5-8

## Using the Debug Tools

The `debug` module provides utilities for exploring data relationships:

```bash
# Show data summary
uv run python -m poe_rl.debug.cli summary

# Search for essences
uv run python -m poe_rl.debug.cli essence "grounding"

# Get detailed essence info with mod resolution
uv run python -m poe_rl.debug.cli essence-info 3187 --class Ring

# Explore a mod family across tiers/classes
uv run python -m poe_rl.debug.cli mod-family 5039

# Find mods by name
uv run python -m poe_rl.debug.cli mod "resistance" --class Ring
```

Or programmatically:

```python
from poe_rl.debug import DataExplorer
from poe_rl.debug.explorer import create_explorer_from_path

explorer = create_explorer_from_path("src/poe_rl/data/static/poec_data.json")

# Find all life essences
life_essences = explorer.find_essences("life")

# Get mods an essence can add
essence = explorer.get_essence("3187")
ring_mods = explorer.get_essence_mods_for_class(essence, "Ring")

# Explore a mod family
mods = explorer.get_mods_by_base_id("5039")
```
