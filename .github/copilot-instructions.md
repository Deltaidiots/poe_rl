# Copilot Instructions
this is a uv project for poe2 ring crafting rl use uv efficiently. 
I am trying to make it functional progreamming style where possible.
Use design prinicples and design patterns where possible
always  try to think about the architecture of the project and how new features will fit in if not sure ask me

## Architecture & Data Flow
- Static PoE2 dumps live under `src/poe_rl/data/static/` (`poec_data.json`, `poec_prices.json`). `data/loader.py` normalizes Craft of Exile exports (prefix stripping, tier expansion) into dataclasses defined in `data/models.py`.
- `engine/database.py` wraps those dataclasses; `CraftingEngine` (`engine/core.py`) couples the DB with a `PriceProvider` so every `CraftingAction` yields both the new `Item` and a chaos-cost. Engine randomness comes from `random.Random`; seed it for reproducibility.
- Currency, omen, and essence behavior is hard-coded in `engine/actions.py`. Actions are composed of `requirements` + `effects`; most effects mutate the passed `Item`. The top-level `ACTIONS` list plus `create_essence_actions()` defines the canonical action order used by RL and UI layers.
- RL code lives in `rl/`: `envs/ring_crafting.py` is the legacy discrete-state environment, while `rl/envs/ring_crafting_env.py` (new PPO target) exposes a 99-d observation, discrete actions with masks, and reward shaping around life/res goals. Agents (currently Q-Learning) are under `rl/agents/`.
- `ui/app.py` is a Streamlit simulator leveraging the same engine; useful to sanity-check crafting flows or visualize omen effects when debugging RL behaviors.

## Key Workflows
- Install deps with `uv sync` (preferred) or `pip install -e .`. Python 3.9+ is required.
- Run the CLI trainer: `uv run python -m poe_rl.main --mods src/poe_rl/data/static/poec_data.json --episodes 5000`. It builds the database, trains Q-learning, then demos the learned policy.
- Generate metadata (life/res groups, essence buckets) with `PYTHONPATH=. python3 -m src.poe_rl.data.ring_inspector --json`. The report guides goal specs for RL environments.
- For PPO/SB3 experiments, instantiate `RingCraftingEnvV1` with a real `CraftingEngine`, then wrap it with `sb3_contrib.common.maskable.MaskableEnv` and feed `env.get_action_mask()` to `ActionMasker`. Costs are already returned via `info['cost']` for custom reward shaping.

## Conventions & Gotchas
- Mod tier logic: tiers are normalized so tier 1 is best. When comparing tiers, use the helpers inside `ring_crafting_env.py` or inspect `Mod.tier`; do not assume weights map to tiers.
- Actions often expect omens or item states; never bypass `CraftingAction.requirements` when calling them (mask invalid actions in RL). Tests assert parity with the Javascript simulator.
- `CraftingDatabase.get_mod_candidates` is intentionally permissive; tests replace it with stubs to force deterministic pools. If you need stricter filtering (item class, tags), extend that method and adjust tests.
- Price lookup uses keys like `currency/method_transmute`. When adding new actions, ensure `price_provider.get_price()` can resolve their currency key or training rewards will skew positive.
- Observation features in `RingCraftingEnvV1` assume base `CraftingEngine.create_item(base_id="12", ilvl=80)`. If you swap bases, update goal specs and normalization constants.

## Testing & Validation
- Currency/omen parity lives in `tests/test_actions_functional.py`; it uses `StubDatabase` pools to assert exact mod rolls. Keep new action logic covered here.
- Fracturing/annulment edge cases are checked in `tests/test_fracturing.py`; omens + essences failure semantics live in `tests/test_omens_failure.py`.
- Run tests with `uv run pytest` (or target a single module for faster feedback). Tests rely on pure Python; no external data downloads required.
