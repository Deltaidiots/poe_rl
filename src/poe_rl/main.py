import argparse
import os
import random
import json
from typing import List

from .data.loader import load_mods, load_json_file, load_essences
from .data.models import ItemBase
from .engine.core import CraftingDatabase, CraftingEngine
from .engine.price import PriceProvider, StaticPriceProvider
from .rl.agents.q_learning import train_q_learning
from .rl.envs.ring_crafting import RingCraftEnv


def demonstrate_policy(env: RingCraftEnv, q_table: List[List[float]], runs: int = 3) -> None:
    """Run demonstration episodes using the learned policy and print the outcomes."""
    for run in range(runs):
        state = env.reset()
        done = False
        total_cost = 0.0
        print(f"\nRun {run + 1}:")
        while not done:
            state_idx = state.to_index()
            # Choose the greedy action
            max_q = max(q_table[state_idx])
            best_actions = [i for i, q in enumerate(q_table[state_idx]) if q == max_q]
            action_idx = random.choice(best_actions)
            action = env.actions[action_idx]
            next_state, reward, done = env.step(action_idx)
            # Print action and state
            print(f"  Action: {action.name}, Reward: {reward:.2f}, State: {next_state}")
            # Costs are negative rewards for currency used
            if reward < 0:
                total_cost += -reward
            state = next_state
        print(f"  Final cost: {total_cost:.2f}")


def build_price_provider() -> PriceProvider:
    """Create a static price provider for all currencies used in this environment."""
    path = "src/poe_rl/data/static/poec_prices.json"
    if not os.path.exists(path):
        print(f"Warning: Price file {path} not found. Using empty prices.")
        return StaticPriceProvider({"none": 0.0})
    
    try:
        # Manually read and strip potential JS assignment prefix
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("poecp="):
                content = content[6:]
            if content.endswith(";"):
                content = content[:-1]
            data = json.loads(content)

        if not isinstance(data, dict):
            return StaticPriceProvider({"none": 0.0})
            
        league_data = data.get("data", {})
        if not isinstance(league_data, dict) or not league_data:
            return StaticPriceProvider({"none": 0.0})
        
        # Pick first league (e.g. "Rise of the Abyssal")
        league_name = next(iter(league_data))
        prices_data = league_data[league_name]
        if not isinstance(prices_data, dict):
            return StaticPriceProvider({"none": 0.0})
        
        prices = {}
        # Flatten categories
        for category in ["currency", "essences", "omens"]:
            cat_prices = prices_data.get(category, {})
            if isinstance(cat_prices, dict):
                for k, v in cat_prices.items():
                    try:
                        prices[k] = float(v)
                    except (ValueError, TypeError):
                        pass
        
        prices["none"] = 0.0
        return StaticPriceProvider(prices)
    except Exception as e:
        print(f"Error loading prices: {e}")
        return StaticPriceProvider({"none": 0.0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent for PoE2 ring crafting")
    parser.add_argument(
        "--mods",
        default="src/poe_rl/data/static/poec_data.json",
        help="Path to the poec_data.json file (default: src/poe_rl/data/static/poec_data.json)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if not os.path.exists(args.mods):
        print(f"Error: Mods file '{args.mods}' not found. Please ensure the data file is present.")
        return

    # Load mods
    all_mods = load_mods(args.mods)
    if not all_mods:
        raise ValueError("No mods loaded; check the mods JSON file")
    
    # Filter for ring mods (Item Class "Ring" or ID "1")
    # In loader.py, we set item_classes=["Ring"] etc.
    mods = [m for m in all_mods if "Ring" in m.item_classes or "1" in m.item_classes]
    
    # Load essences
    essences = load_essences(args.mods)
    essences_by_id = {e.essence_id: e for e in essences}
    
    # Create a generic ring base
    ring_base = ItemBase(
        base_id="generic_ring",
        name="Generic Ring",
        item_class="1", # Use ID "1" for Ring to match Essence data
        drop_level=1,
        tags=["ring", "jewellery"],
        implicits=[],
    )
    # Build crafting database and engine
    db = CraftingDatabase(
        mods_by_id={m.mod_id: m for m in mods}, 
        bases_by_id={ring_base.base_id: ring_base},
        essences_by_id=essences_by_id
    )
    price_provider = build_price_provider()
    engine = CraftingEngine(db=db, price_provider=price_provider)

    # Define groups: we pick one life group and one resistance group from mods heuristically
    life_group = None
    res_group = None
    
    # Common group names in CoE data
    target_life = "Life"
    target_res = "FireResistance" 
    
    for m in mods:
        if life_group is None and m.is_prefix and m.group == target_life:
            life_group = m.group
        if res_group is None and not m.is_prefix and m.group == target_res:
            res_group = m.group
        if life_group and res_group:
            break
            
    # Fallback heuristics if exact matches not found
    if not life_group:
        for m in mods:
            if m.is_prefix and "Life" in m.group:
                life_group = m.group
                break
    if not res_group:
        for m in mods:
            if not m.is_prefix and "Resistance" in m.group:
                res_group = m.group
                break

    if not life_group or not res_group:
        # List some available groups to help debugging
        available_groups = list(set(m.group for m in mods))
        print(f"Available groups: {available_groups[:10]}...")
        raise ValueError(f"Could not find target groups. Found Life='{life_group}', Res='{res_group}'")

    print(f"Training for groups: Life='{life_group}', Resistance='{res_group}'")

    # Create environment
    # Note: We are using 'life_group' as the 'attack_group' parameter for now, 
    # as the environment expects two target groups.
    env = RingCraftEnv(engine, attack_group=life_group, res_group=res_group)
    # Train Q-learning agent
    q_table = train_q_learning(env, episodes=args.episodes, seed=args.seed)
    # Print Q-table summary
    print("\nQ-table sample (first 8 states):")
    for s in range(8):
        print(f"State {s}: {q_table[s]}")
    # Demonstrate policy
    demonstrate_policy(env, q_table, runs=3)


if __name__ == "__main__":
    main()
