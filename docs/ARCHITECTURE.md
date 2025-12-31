# Architecture Overview

This document describes the high-level architecture of the PoE2 Ring Crafting RL project.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PoE2 Ring Crafting RL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐ │
│  │   Data      │───▶│     Engine      │───▶│        RL Environment        │ │
│  │   Layer     │    │     Layer       │    │           Layer              │ │
│  └─────────────┘    └─────────────────┘    └──────────────────────────────┘ │
│        │                   │                            │                   │
│        │                   │                            │                   │
│        ▼                   ▼                            ▼                   │
│  ┌───────────┐      ┌───────────────┐          ┌───────────────────┐       │
│  │  Models   │      │ CraftingEngine│          │ RingCraftingEnvV1 │       │
│  │  Parsers  │      │   Actions     │          │     Agents        │       │
│  │  Loaders  │      │   Database    │          │    Training       │       │
│  └───────────┘      │   Price       │          └───────────────────┘       │
│                     └───────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### Data Layer (`src/poe_rl/data/`)

**Purpose:** Load and normalize crafting data from Craft of Exile exports.

| File | Responsibility |
|------|----------------|
| `models.py` | Dataclass definitions (Mod, Essence, ItemBase, Item) |
| `parsers.py` | JSON parsing logic for different data formats |
| `loader.py` | High-level loading functions with CoE wrapper handling |
| `static/` | Raw data files (poec_data.json, poec_prices.json) |

**Key Design Decisions:**
- Mods get qualified IDs (`{base_id}_{class}_{tier}`) for unique lookup
- Essences keep base mod IDs for flexibility in tier selection
- ItemBase includes both `item_class` (name) and `item_class_id` (numeric) for compatibility

### Engine Layer (`src/poe_rl/engine/`)

**Purpose:** Simulate PoE2 crafting mechanics with cost tracking.

| File | Responsibility |
|------|----------------|
| `database.py` | Container for mods/essences/bases with query methods |
| `core.py` | `CraftingEngine` - combines database + price provider |
| `actions.py` | `CraftingAction` definitions with requirements/effects |
| `price.py` | Price lookup for currency costs |

**Key Design Decisions:**
- Actions are composed of `requirements` (predicates) + `effects` (mutations)
- Engine randomness from `random.Random` for reproducibility
- Every action returns (new_item, chaos_cost) for RL reward shaping

### RL Layer (`src/poe_rl/rl/`)

**Purpose:** Gymnasium environments and training infrastructure.

| File | Responsibility |
|------|----------------|
| `envs/ring_crafting_env.py` | Main PPO environment with action masking |
| `envs/ring_crafting.py` | Legacy discrete Q-learning environment |
| `agents/` | Agent implementations (currently Q-learning) |
| `train_ppo.py` | MaskablePPO training script |
| `eval_ppo.py` | Policy evaluation and visualization |

**Key Design Decisions:**
- 99-dimensional observation space (item state + action availability)
- Action masking via `sb3_contrib.MaskablePPO`
- Reward shaping around life/resistance goals with cost penalties

## Component Interactions

### Training Flow

```
1. Load Data
   poec_data.json → load_mods(), load_essences(), load_item_bases()
                          │
                          ▼
2. Build Engine
   CraftingDatabase(mods, essences, bases)
   CraftingEngine(database, price_provider)
                          │
                          ▼
3. Create Environment
   RingCraftingEnvV1(engine, reward_config)
   ActionMasker wrapper for MaskablePPO
                          │
                          ▼
4. Train Agent
   MaskablePPO.learn(total_timesteps)
   Policy saved to artifacts/
```

### Action Execution Flow

```
1. Agent selects action index
                │
                ▼
2. Environment looks up CraftingAction
   actions[action_idx]
                │
                ▼
3. Check requirements (item state predicates)
   action.requirements → [item_is_rare, has_open_prefix, ...]
                │
                ▼
4. Apply effects (item mutations)
   action.effects → [chaos_reroll, add_mod, remove_omen, ...]
                │
                ▼
5. Calculate reward
   goal_progress + tier_bonuses - cost_penalty
                │
                ▼
6. Return (obs, reward, done, truncated, info)
```

## Key Abstractions

### CraftingAction

Composable action definition:

```python
CraftingAction(
    name="Chaos Orb",
    requirements=[item_is_rare],      # All must pass
    effects=[ChaosRerollEffect()],    # Applied in order
    currency_key="currency/chaos_orb"
)
```

### Requirements (Predicates)

```python
def item_is_rare(item: Item, db: CraftingDatabase) -> bool:
    return item.rarity == "Rare"
```

### Effects (Mutations)

```python
class ChaosRerollEffect:
    def apply(self, item: Item, db: CraftingDatabase, rng: random.Random) -> Item:
        item.prefix_ids = []
        item.suffix_ids = []
        # ... reroll logic
        return item
```

### RewardConfig

Configurable reward shaping:

```python
RewardConfig(
    life_bonus_per_tier=0.5,
    res_bonus_per_tier=0.3,
    cost_penalty_scale=0.01,
    essence_guaranteed_mod_bonus=1.0,
    chaos_spam_penalty_ramp=0.1,
    # ...
)
```

## File Structure

```
src/poe_rl/
├── __init__.py
├── main.py                 # CLI entry point
├── data/
│   ├── models.py          # Dataclasses
│   ├── parsers.py         # JSON parsing
│   ├── loader.py          # High-level loaders
│   └── static/            # Data files
├── engine/
│   ├── database.py        # CraftingDatabase
│   ├── core.py            # CraftingEngine
│   ├── actions.py         # Action definitions
│   └── price.py           # Price provider
├── rl/
│   ├── envs/
│   │   ├── ring_crafting_env.py  # Main env
│   │   └── ring_crafting.py      # Legacy env
│   ├── agents/            # Agent implementations
│   ├── train_ppo.py       # Training script
│   └── eval_ppo.py        # Evaluation script
├── debug/
│   ├── explorer.py        # DataExplorer class
│   └── cli.py             # Debug CLI
└── ui/
    └── app.py             # Streamlit simulator
```

## Extension Points

### Adding New Currency/Actions

1. Add effect class in `actions.py`
2. Create `CraftingAction` with requirements/effects
3. Add to `ACTIONS` list or create factory function
4. Update price provider if new currency type

### Adding New Item Types

1. Extend `ItemBase` and `Item` if needed
2. Create environment variant in `rl/envs/`
3. Adjust observation space and reward config

### Adding New RL Algorithms

1. Create agent class in `rl/agents/`
2. Implement standard interface (train, predict, save, load)
3. Add training script variant

## Design Principles

1. **Functional Style:** Where possible, functions over classes, immutability preferred
2. **Composition over Inheritance:** Actions are composed of requirements/effects
3. **Separation of Concerns:** Data loading, engine logic, and RL are independent
4. **Reproducibility:** Random seeds flow through for deterministic testing
5. **Extensibility:** Easy to add new actions, items, or algorithms
