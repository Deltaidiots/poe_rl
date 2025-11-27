PLan to implement RL based strategy to get the best crafting strategy for POE2 item crafting

## 0. High-level goal

**Input**

* White base (e.g. “STR/DEX boots, ilvl 85”)
* Target item spec (e.g. “T1 life, ≥2 T2 res, open suffix, move speed ≥ 25%”)
* Current market prices for currencies (chaos, exalts, essences, omens, abyss, etc.)

**Output**

* 1–3 **crafting strategies** (sequences of actions) with:

  * Expected total cost
  * Success probability
  * Example crafting paths (step-by-step)
* Optionally: a chat-style explanation of “why” each strategy is chosen.

We’ll build:

1. **Data layer** (CoE/PoE2 + mods + currency prices)
2. **Crafting engine** (apply chaos, essence, omen, abyss, etc. to a simulated item)
3. **RL environment** (Gym-like) + trainer
4. **Strategy evaluator** (run policy, compute stats)
5. **CLI/JSON API** (for programmatic use, plus a chat layer if you want later)

---

## 1. Repo structure (tell Codex to create this)

```text
poe2_crafter/
├── poe2_crafter/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── README.md
│   │   ├── poec_lang.us.json         # from CoE (optional)
│   │   ├── poec_data.json            # from CoE (optional)
│   │   ├── poec_common.json          # from CoE (optional)
│   │   ├── mods.json                 # RePoE-style mods for PoE2 (your curated file)
│   ├── domain/
│   │   ├── models.py                 # Mod, ItemBase, Item, CurrencyPrice, etc.
│   │   ├── serialization.py          # From/to JSON ↔ Item
│   ├── data_loader/
│   │   ├── coe_loader.py             # parse CoE JSON
│   │   ├── mods_loader.py            # parse mods.json / RePoE-style
│   │   └── prices_loader.py          # price sources (poe.ninja/trade APIs, mocked in dev)
│   ├── engine/
│   │   ├── actions.py                # definitions of crafting actions
│   │   ├── engine.py                 # CraftingEngine core
│   │   └── strategies.py             # scripted strategy templates (non-RL baseline)
│   ├── rl/
│   │   ├── env.py                    # Gym-style environment
│   │   ├── reward.py                 # reward shaping, termination logic
│   │   ├── train.py                  # training loop
│   │   └── policy_runner.py          # run trained policy for evaluation
│   ├── api/
│   │   ├── spec.py                   # target item specification schema
│   │   ├── optimize.py               # main optimize(target, base, prices) entrypoint
│   ├── cli/
│   │   └── main.py                   # command-line interface
│   └── utils/
│       ├── rng.py
│       └── logging_utils.py
├── experiments/
│   ├── notebooks/                    # optional for manual analysis
├── models/                           # saved RL policies
├── requirements.txt
└── README.md
```

Tell Codex to scaffold this structure and create minimal empty modules first.

---

## 2. Data layer

### 2.1 Domain models (`domain/models.py`)

Ask Codex to implement these dataclasses:

```python
@dataclass
class Mod:
    id: str
    group: str           # group key like "life", "resistance"
    tier: int            # numeric tier (1 = best)
    weight: int          # spawn weight in this pool
    is_prefix: bool
    tags: list[str]      # life, attack, fire, etc.
    min_value: float
    max_value: float

@dataclass
class ItemBase:
    id_base: str         # from poec_data or RePoE
    name: str
    item_class: str      # helmet, boots, ring...
    level_req: int
    max_prefixes: int = 3
    max_suffixes: int = 3
    implicit_mods: list[Mod] = field(default_factory=list)

@dataclass
class Item:
    base: ItemBase
    ilvl: int
    prefixes: list[Mod] = field(default_factory=list)
    suffixes: list[Mod] = field(default_factory=list)
    crafted: bool = False

@dataclass
class CurrencyPrice:
    name: str     # "chaos", "exalt", "essence_of_wrath_t1", ...
    chaos_value: float
```

Also add helpers:

* `Item.copy()`
* `Item.num_open_prefixes/suffixes`
* `Item.has_mod_group(group: str)`

### 2.2 CoE JSON loader (`data_loader/coe_loader.py`)

Use your links:

* `poec_lang.us.json` → base item labels + mod text
* `poec_data.json` → base items (bitems + base ids)
* `poec_common.json` → bench costs, leagues, etc.

Codex tasks:

1. Implement a function `load_lang(path) -> dict` that:

   * Reads the file
   * Strips the `poecl=` prefix
   * Returns `json.loads(rest)`
2. Implement `load_poec_data(path) -> dict`:

   * Strip `poecd=`
   * Load JSON
3. Implement `load_common(path) -> dict`:

   * Strip `poecc=`
   * Load JSON
4. Implement `build_item_bases(lang, poec_data) -> dict[str, ItemBase]`:

   * Interpret `poecd["bitems"]["seq"]` records:

     * map `id_base` → `ItemBase`
     * use `poecl["base"][id_base]` for class names
   * Keep this fairly generic; it’s just for base selection in the RL environment.

### 2.3 Mods loader (`data_loader/mods_loader.py`)

Because CoE’s PoE2 mod weights aren’t exposed cleanly, define your own `mods.json`:

```jsonc
{
  "prefixes": [
    { "id": "life_t1", "group": "life", "tier": 1, "weight": 100, "min_value": 90, "max_value": 110, "tags": ["life"] },
    ...
  ],
  "suffixes": [
    { "id": "res_fire_t1", "group": "resistance", "tier": 1, "weight": 100, "min_value": 40, "max_value": 48, "tags": ["fire"] },
    ...
  ]
}
```

Codex tasks:

* Implement `load_mods(path="data/mods.json") -> list[Mod]`:

  * Parse JSON and instantiate `Mod` objects.
  * Split into prefix/suffix lists.
* (Future) Optionally extend to parse **RePoE** `mods.json` if available.

### 2.4 Price loader (`data_loader/prices_loader.py`)

Design an interface **even if you stub it**; important for RL:

```python
class PriceProvider(Protocol):
    def get_price(self, currency_name: str) -> float:
        ...
```

Codex tasks:

* Implement `StaticPriceProvider` (hardcoded dict or JSON file).
* Implement stub for `PoeNinjaPriceProvider` (HTTP calls commented / TODO so you don’t break ToS during dev).

---

## 3. Crafting engine

### 3.1 Actions definition (`engine/actions.py`)

Define abstract and concrete actions:

```python
class CraftAction(ABC):
    name: str
    currency_name: str  # for cost lookup

    @abstractmethod
    def apply(self, item: Item, rng: Random) -> Item:
        ...
```

Concrete actions to start with:

* `ChaosOrbAction`

  * Clears all prefixes/suffixes, then:

    * Rolls 2–3 random prefixes + 2–3 random suffixes using spawn weights and tag restrictions.
* `ExaltedOrbAction`

  * Adds one random prefix *or* suffix if there is a free slot.
* `EssenceOfLifeAction`

  * Forces a life prefix (best tier available for ilvl) and then random other mods.
* `AbyssCraftAction` (simple version)

  * Adds one extra “special” mod from a subset.
* `BenchCraftAction`

  * Adds chosen bench mod if there is a free slot (cost comes from `poec_common["benchcosts"]` mapping).

Codex tasks:

1. Implement **mod selection** given an `Item`:

   ```python
   def roll_mod(
       item: Item,
       mods: list[Mod],
       allowed_tags: set[str] | None = None,
   ) -> Mod: ...
   ```

   * Filter by not violating `group` uniqueness, open slots, tags, and ilvl (later).
   * Choose by weight.

2. Implement each `CraftAction.apply` using this helper.

### 3.2 CraftingEngine (`engine/engine.py`)

```python
class CraftingEngine:
    def __init__(self, mods: list[Mod], price_provider: PriceProvider, rng: Random | None = None):
        self.mods = mods
        self.price_provider = price_provider
        self.rng = rng or Random()

    def apply_action(self, item: Item, action: CraftAction) -> tuple[Item, float]:
        new_item = deepcopy(item)
        new_item = action.apply(new_item, self.rng)
        cost = self.price_provider.get_price(action.currency_name)
        return new_item, cost
```

Later you can add:

* logging of steps
* seeds for reproducibility.

---

## 4. Target specification & similarity

We need a formal way to say “this item is good enough”.

Create `api/spec.py`:

```python
@dataclass
class TargetConstraint:
    must_have_groups: list[str]       # e.g. ["life"]
    min_res_mods: int                 # e.g. 2
    min_tier_by_group: dict[str, int] # {"life": 2, "resistance": 3}
    required_open_prefixes: int = 0
    required_open_suffixes: int = 1
    custom_score_weights: dict[str, float] = field(default_factory=dict)
```

Codex tasks:

* Implement `evaluate_item(item: Item, spec: TargetConstraint) -> tuple[bool, float]`:

  * Returns `(is_success, score)`:

    * `is_success` = True if all hard constraints satisfied (has life, at least 2 res mods, open suffix, etc.)
    * `score` can be sum of weighted tiers/values (used for RL or for ranking results).

---

## 5. RL environment

### 5.1 Environment design (`rl/env.py`)

State → features:

* Simplest version: just track “progress” like we did in the toy script.
* Better: encode more structure:

```python
@dataclass
class EnvState:
    item: Item
    budget_spent: float
    steps_taken: int
```

For RL you need to map it to a vector:

* Number of mods by group (life/res/attack/etc.)
* Best tier per group.
* Flags: has_open_prefix/suffix, has_desecrated, etc.
* Normalized budget_spent and steps_taken.

Codex tasks:

1. Implement a `gym.Env`-like class:

```python
class PoeCraftEnv(gym.Env):
    def __init__(self, engine: CraftingEngine, base: ItemBase, spec: TargetConstraint, price_provider: PriceProvider):
        ...
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
```

2. Define actions as indices referencing a list:

```python
self.actions = [
    EssenceOfLifeAction(),
    ChaosOrbAction(),
    ExaltedOrbAction(),
    BenchCraftLifeAction(),
    StopAction(),             # meta action to accept current item
]
```

3. Implement `reset()`:

   * Create a fresh white Item from `base`.
   * Zero budget/steps.
   * Return encoded observation.

4. Implement `step(action_idx)`:

   * If action is `Stop`:

     * Evaluate item vs `spec`:

       * If success: reward = +R_success – λ * total_cost
       * Else: reward = −penalty – λ * total_cost
     * `done = True`
   * Else:

     * Apply crafting action via `CraftingEngine`.

     * Update `budget_spent`, `steps_taken`.

     * Compute intermediate reward: e.g.

       ```python
       reward = -cost + shaping_factor * (new_score - old_score)
       ```

     * Terminate if:

       * success reached, or
       * budget > max_budget, or
       * steps > max_steps.

5. Add `render()` that prints the current item mods in a readable way (use lang data for names).

### 5.2 Reward shaping (`rl/reward.py`)

Codex tasks:

* Implement utility functions:

```python
def compute_shaped_reward(
    prev_score: float,
    new_score: float,
    cost: float,
    terminal: bool,
    success: bool
) -> float:
    ...
```

Start simple:

* `reward = -cost` every step
* +big bonus at success (e.g. `+50` chaos equivalent)
* optionally add `(new_score - prev_score)`.

---

## 6. RL training

### 6.1 Trainer (`rl/train.py`)

Codex tasks:

1. Add Stable-Baselines3 to `requirements.txt`.

   ```text
   stable-baselines3
   gymnasium
   numpy
   ```

2. Implement a `train_policy` function:

```python
def train_policy(
    env_config: EnvConfig,
    algo: str = "ppo",
    total_timesteps: int = 1_000_000,
    save_path: str = "models/ppo_poe2_crafter.zip",
) -> None:
    env = PoeCraftEnv(...)
    if algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, verbose=1)
    ...
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
```

3. Make a small `EnvConfig` dataclass for base item, spec, max budget, etc.

4. Add a `__main__` block so you can run:

```bash
python -m poe2_crafter.rl.train --config configs/boots_life_res.json
```

(You can have Codex parse CLI options with `argparse`.)

---

## 7. Strategy evaluation & optimizer

### 7.1 Policy runner (`rl/policy_runner.py`)

Codex tasks:

1. Implement `run_policy_many_episodes(model, env, n_episodes=1000)`:

   * For each episode:

     * `obs = env.reset()`
     * Step until done with `model.predict(obs, deterministic=True)`
     * Record:

       * success / fail
       * total cost
       * sequence of actions
   * Aggregate:

     * success probability
     * mean / median cost
     * example sequences (e.g. three cheapest successes, three median successes).

2. Return a structured result:

```python
@dataclass
class StrategyStats:
    success_prob: float
    expected_cost: float
    cost_distribution: list[float]
    example_paths: list[list[str]]  # sequences of action names
```

### 7.2 Optimizer entrypoint (`api/optimize.py`)

This is the main function you call from CLI/chat:

```python
def optimize_crafting(
    base_name: str,
    target_spec: TargetConstraint,
    price_provider: PriceProvider,
    model_path: str,
    n_eval_episodes: int = 1000,
) -> dict:
    # 1. Load item_base by name
    # 2. Construct env + load RL model
    # 3. Run policy many episodes -> StrategyStats
    # 4. Optionally run a few scripted baseline strategies from `engine/strategies.py` for comparison
    # 5. Return a JSON-serializable dict with results
```

Output example:

```json
{
  "base": "STR/DEX Boots",
  "target": {...},
  "strategy": {
    "success_probability": 0.67,
    "expected_cost_chaos": 134.2,
    "example_paths": [
      ["EssenceOfLife", "Chaos", "Exalt", "Stop"],
      ...
    ]
  },
  "baselines": [
    { "name": "Essence-only", "expected_cost": 200, "success_prob": 0.4 }
  ]
}
```

---

## 8. CLI and chat integration

### 8.1 CLI (`cli/main.py`)

Codex tasks:

* Implement a CLI like:

```bash
poe2-crafter \
  --base "Boots (STR/DEX)" \
  --target-spec configs/target_boots_life_res.json \
  --model-path models/ppo_boots_life_res.zip
```

* Parse `target_spec` JSON into `TargetConstraint`.
* Call `optimize_crafting(...)`.
* Pretty-print summary:

  * “Best strategy: expected cost X, success Y%, exemplar path: Essence → Chaos → Exalt → Stop”.

### 8.2 Chat layer (optional later)

When you’re ready:

* Add a small FastAPI server (`api/server.py`) with endpoints:

  * `POST /optimize` (takes base + natural language spec, uses an LLM to map NL → TargetConstraint).
* Later, plug into a frontend / Discord bot.

---

## 9. Testing & sanity checks

Tell Codex to also generate:

1. **Unit tests** (`tests/`) for:

   * `roll_mod` respecting groups and slots.
   * `CraftAction.apply` not exceeding prefix/suffix caps.
   * `compute_progress/evaluate_item` correctness on simple items.
   * `PoeCraftEnv` termination conditions.

2. **Deterministic runs** with fixed random seed to make debugging reproducible.

3. **Benchmark script** in `experiments/` that:

   * Runs a naive strategy (e.g. “essence spam + stop”) and the RL policy.
   * Compares success prob and cost.

---

## 10. How to phrase this for Codex

When you paste this into Codex, you can give it step-by-step tasks, e.g.:

1. *“Create the repo structure above and stub out empty files.”*
2. *“Implement `domain/models.py` with the Mod, ItemBase, Item, CurrencyPrice dataclasses.”*
3. *“Implement `data_loader/coe_loader.py` with functions to parse poec_lang.us.json, poec_data.json, and poec_common.json, stripping the prefixes.”*
4. *“Implement `data_loader/mods_loader.py` that loads a mods.json like this: …”*
5. *“Implement actions (Chaos, Exalt, Essence) and CraftingEngine as specified.”*
6. *“Implement PoeCraftEnv with observation encoding + reward function as described.”*
7. *“Add Stable-Baselines3 training script using PPO.”*
8. *“Implement policy_runner and optimize_crafting as described.”*
9. *“Add CLI script that wires everything together.”*

If you want, I can next write **concrete Codex prompts** for each module (like copy-pasteable instructions with signatures and docstrings).
