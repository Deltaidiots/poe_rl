# PoE2 Ring Crafting RL

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reinforcement learning project that trains agents to craft optimal rings in Path of Exile 2, using real crafting data from [Craft of Exile](https://www.craftofexile.com/).

## 🎯 Project Goals

Train an RL agent to:
- **Maximize item quality:** Target +Life and +Resistance mods at high tiers
- **Minimize currency costs:** Learn efficient crafting strategies
- **Handle uncertainty:** Navigate PoE2's stochastic crafting system

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/poe_rl.git
cd poe_rl

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Train a PPO Agent

```bash
uv run python -m poe_rl.rl.train_ppo \
    --mods src/poe_rl/data/static/poec_data.json \
    --timesteps 100000
```

### Evaluate a Policy

```bash
uv run python -m poe_rl.rl.eval_ppo \
    --policy artifacts/ppo_ring_policy.zip \
    --episodes 100
```

### Explore the Data

```bash
# Summary of available data
uv run python -m poe_rl.debug.cli summary

# Search for essences
uv run python -m poe_rl.debug.cli essence "grounding"

# Detailed essence info
uv run python -m poe_rl.debug.cli essence-info 3187 --class Ring
```

## 📁 Project Structure

```
src/poe_rl/
├── data/           # Data loading and models
│   ├── models.py   # Dataclasses: Mod, Essence, Item, ItemBase
│   ├── loader.py   # Load from Craft of Exile exports
│   └── static/     # Raw data files
├── engine/         # Crafting simulation
│   ├── core.py     # CraftingEngine
│   ├── actions.py  # Currency/Essence/Omen actions
│   └── database.py # Mod/Essence lookup
├── rl/             # Reinforcement learning
│   ├── envs/       # Gymnasium environments
│   ├── agents/     # Agent implementations
│   └── train_ppo.py
├── debug/          # Data exploration tools
│   ├── explorer.py # DataExplorer class
│   └── cli.py      # Command-line interface
└── ui/             # Streamlit visualization
    └── app.py
```

## 🧠 RL Environment

### Observation Space (99 dimensions)

- Item rarity, affix slots, corruption state
- Per-mod-group features (life, resistance groups)
- Goal progress indicators
- Tag presence vectors

### Action Space (110 discrete actions)

| Category | Count | Examples |
|----------|-------|----------|
| Base Currency | 17 | Transmute, Chaos, Exalt, Annul |
| Essences | 81 | All essence types and tiers |
| Omens | 11 | Exaltation, Annulment, Crystallisation |
| Terminal | 1 | Stop action |

### Reward Shaping

The reward function includes:
- **Progress rewards:** +Life/Resistance mod additions
- **Tier bonuses:** Higher tier mods = more reward
- **Cost penalties:** Currency usage is penalized
- **Anti-exploitation:** Penalties for chaos spam, unused omens
- **Essence incentives:** Bonuses for deterministic actions

See [docs/RL_JOURNEY.md](docs/RL_JOURNEY.md) for the full story.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and component interactions |
| [DATA_MODEL.md](docs/DATA_MODEL.md) | ID formats, data structures, parsing |
| [RL_JOURNEY.md](docs/RL_JOURNEY.md) | Reward evolution and lessons learned |
| [CRAFTING_RULES.md](CRAFTING_RULES.md) | PoE2 crafting mechanics reference |

## 🛠️ Development

### Run Tests

```bash
uv run pytest
```

### Run Streamlit UI

```bash
uv run streamlit run src/poe_rl/ui/app.py
```

### View Training Logs

```bash
tensorboard --logdir runs/ppo/
```

## 🎮 Example Training Run

```
Episode 100: reward=12.3, steps=45, goal_achieved=False
Episode 200: reward=18.7, steps=32, goal_achieved=True
Episode 300: reward=21.2, steps=28, goal_achieved=True
...

Action distribution (last 100 episodes):
  Essence of the Body: 23%
  Essence of Grounding: 18%
  Regal Orb: 15%
  Exalted Orb: 12%
  Chaos Orb: 8%  ← Reduced from 45% after anti-spam measures
```

## 🔬 Key Insights

1. **Action masking is essential** for sample efficiency
2. **Reward hacking happens** – design explicit countermeasures
3. **Deterministic actions > RNG** – incentivize essences over chaos
4. **Debug your data layer** – mod ID mismatches waste training time

## 📊 Data Sources

- **Mod/Base Data:** [Craft of Exile](https://www.craftofexile.com/) exports
- **Price Data:** Chaos-equivalent values from trade sites

## 🤝 Contributing

Contributions welcome! See the [Architecture docs](docs/ARCHITECTURE.md) for extension points.

Areas needing work:
- [ ] Curriculum learning for progressive difficulty
- [ ] Model-based RL with world model
- [ ] Support for more item types (amulets, belts)
- [ ] Better visualization of learned policies

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the PoE community*
