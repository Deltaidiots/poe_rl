# PoE 2 Crafting Rules Specification

This document defines the crafting rules, conditions, and mechanics for Path of Exile 2 as enforced in the simulator. These rules are derived from the game's calculation logic (`poe2.js`).

## General Rules

### Item Rarity & Mod Limits
- **Normal (White)**: 0 modifiers.
- **Magic (Blue)**: 1-2 modifiers (Max 1 Prefix, Max 1 Suffix).
- **Rare (Yellow)**: 3-6 modifiers (Max 3 Prefixes, Max 3 Suffixes).
- **Unique (Orange)**: Fixed modifiers (cannot be crafted with standard currency).

### Corruption
- **Corrupted Items**: Cannot be modified by any currency except Tainted Currency (not yet implemented).
- **Vaal Orb**: Corrupts an item, making it unmodifiable. Outcome is unpredictable (can brick to Rare, change implicits, or do nothing).

### Item Levels (ilvl)
- Currency tiers (Greater, Perfect) have minimum item level requirements for the *modifiers* they generate.
- **Greater**: Minimum mod level 55 (Transmute/Augment), 35 (Regal/Exalt/Chaos).
- **Perfect**: Minimum mod level 70 (Transmute/Augment), 50 (Regal/Exalt/Chaos).

---

## Currency Mechanics

### Orb of Transmutation
**Target**: Normal Items.
**Effect**: Upgrades to Magic. Adds exactly **1** random modifier.
- **Orb of Transmutation**: Adds 1 mod.
- **Greater Orb of Transmutation**: Adds 1 mod (min mod level: 55).
- **Perfect Orb of Transmutation**: Adds 1 mod (min mod level: 70).

### Orb of Augmentation
**Target**: Magic Items.
**Condition**: Must have an open affix slot (Prefix or Suffix).
**Effect**: Adds **1** random modifier.
- **Orb of Augmentation**: Adds 1 mod.
- **Greater Orb of Augmentation**: Adds 1 mod (min mod level: 55).
- **Perfect Orb of Augmentation**: Adds 1 mod (min mod level: 70).

### Orb of Alteration
**Target**: Magic Items.
**Effect**: Reforges the item. Removes all existing modifiers and adds **1** new random modifier. Result remains Magic.

### Regal Orb
**Target**: Magic Items.
**Effect**: Upgrades to Rare. Adds **1** random modifier. Retains existing modifiers.
- **Regal Orb**: Adds 1 mod.
- **Greater Regal Orb**: Adds 1 mod (min mod level: 35).
- **Perfect Regal Orb**: Adds 1 mod (min mod level: 50).
**Supported Omens**:
- **Omen of Homogenising Coronation**: Adds a modifier of the same type (tag) as an existing modifier.
- **Omen of Dextral Coronation**: Adds a Suffix modifier.
- **Omen of Sinistral Coronation**: Adds a Prefix modifier.

### Orb of Alchemy
**Target**: Normal or Magic Items.
**Effect**: Upgrades (or Reforges) to Rare. Removes all existing modifiers and adds **4** random modifiers.
**Supported Omens**:
- **Omen of Dextral Alchemy**: Result has maximum number of Suffixes.
- **Omen of Sinistral Alchemy**: Result has maximum number of Prefixes.

### Chaos Orb
**Target**: Rare Items.
**Effect**: Removes **1** random modifier and adds **1** new random modifier.
- **Chaos Orb**: Removes 1, Adds 1.
- **Greater Chaos Orb**: Removes 1, Adds 1 (min mod level: 35).
- **Perfect Chaos Orb**: Removes 1, Adds 1 (min mod level: 50).
**Supported Omens**:
- **Omen of Whittling**: Removes the lowest level modifier.
- **Omen of Dextral Erasure**: Removes a Suffix.
- **Omen of Sinistral Erasure**: Removes a Prefix.

### Exalted Orb
**Target**: Rare Items.
**Condition**: Must have an open affix slot.
**Effect**: Adds **1** random modifier.
- **Exalted Orb**: Adds 1 mod.
- **Greater Exalted Orb**: Adds 1 mod (min mod level: 35).
- **Perfect Exalted Orb**: Adds 1 mod (min mod level: 50).
**Supported Omens**:
- **Omen of Greater Exaltation**: Adds **2** random modifiers (if space permits).
- **Omen of Homogenising Exaltation**: Adds a modifier of the same type (tag) as an existing modifier.
- **Omen of Dextral Exaltation**: Adds a Suffix.
- **Omen of Sinistral Exaltation**: Adds a Prefix.

### Orb of Annulment
**Target**: Magic or Rare Items.
**Condition**: Must have at least 1 modifier.
**Effect**: Removes **1** random modifier.
**Supported Omens**:
- **Omen of Dextral Annulment**: Removes a Suffix.
- **Omen of Sinistral Annulment**: Removes a Prefix.
- **Omen of Light**: Removes a Desecrated modifier.

### Divine Orb
**Target**: Magic or Rare Items.
**Effect**: Rerolls the numeric values of all explicit modifiers.
**Supported Omens**:
- **Omen of the Blessed**: Rerolls Implicit modifiers instead of Explicit ones.

### Orb of Scouring
**Target**: Magic or Rare Items.
**Effect**: Removes all modifiers and downgrades the item to Normal rarity.

### Orb of Chance
**Target**: Normal Items.
**Effect**: Upgrades to Magic (80%), Rare (19%), or Unique (1%).

---

## Essence Mechanics

Essences in PoE 2 function differently based on their tier.

### Lesser / Normal / Greater Essences
**Target**: **Magic Items** only.
**Effect**: Upgrades to Rare and adds **1** guaranteed modifier (defined by the Essence). Retains existing modifiers (Regal-like behavior).

### Perfect / Corrupted Essences
**Target**: **Rare Items** only.
**Effect**: Removes **1** random modifier and adds **1** guaranteed modifier (defined by the Essence).
**Supported Omens**:
- **Omen of Dextral Crystallisation**: Removes a Suffix.
- **Omen of Sinistral Crystallisation**: Removes a Prefix.

---

## Omen Mechanics

Omens are consumable items that activate automatically when their condition is met. An item can have only **one** active Omen at a time.

### Activation Rules
1.  **Apply Omen**: Use the Omen item on the gear piece. The gear piece gains the "Active Omen" property.
2.  **Consume Omen**: Use the corresponding currency (e.g., Chaos Orb). The Omen effect triggers, modifying the currency's behavior, and the Omen is removed from the item.

### List of Omens

| Omen Name | Consumed By | Effect |
| :--- | :--- | :--- |
| **Omen of Dextral Annulment** | Orb of Annulment | Removes a Suffix. |
| **Omen of Sinistral Annulment** | Orb of Annulment | Removes a Prefix. |
| **Omen of Light** | Orb of Annulment | Removes a Desecrated modifier. |
| **Omen of Dextral Exaltation** | Exalted Orb | Adds a Suffix. |
| **Omen of Sinistral Exaltation** | Exalted Orb | Adds a Prefix. |
| **Omen of Greater Exaltation** | Exalted Orb | Adds 2 modifiers. |
| **Omen of Homogenising Exaltation** | Exalted Orb | Adds a mod sharing a tag with an existing mod. |
| **Omen of Whittling** | Chaos Orb | Removes the lowest level modifier. |
| **Omen of Dextral Erasure** | Chaos Orb | Removes a Suffix. |
| **Omen of Sinistral Erasure** | Chaos Orb | Removes a Prefix. |
| **Omen of Homogenising Coronation** | Regal Orb | Adds a mod sharing a tag with an existing mod. |
| **Omen of Dextral Coronation** | Regal Orb | Adds a Suffix. |
| **Omen of Sinistral Coronation** | Regal Orb | Adds a Prefix. |
| **Omen of Dextral Crystallisation** | Perfect/Corrupted Essence | Removes a Suffix. |
| **Omen of Sinistral Crystallisation** | Perfect/Corrupted Essence | Removes a Prefix. |
| **Omen of the Blessed** | Divine Orb | Rerolls Implicit modifiers. |
| **Omen of Dextral Alchemy** | Orb of Alchemy | Result has max Suffixes. |
| **Omen of Sinistral Alchemy** | Orb of Alchemy | Result has max Prefixes. |

---

## Desecration Mechanics
**Target**: Rare Items.
**Effect**: Adds a special "Desecrated" modifier from a specific pool (Bone/Jawbone/etc.).
**Omens**: Specific Omens (`Liege`, `Sovereign`, `Blackblooded`) guarantee specific Desecrated mod groups.
