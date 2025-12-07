# PoE2 Ring Crafting RL

This project implements a Reinforcement Learning (Q-Learning) agent to optimize crafting strategies for rings in Path of Exile 2.

## Project Structure

The project is organized as a modular Python package:

- `src/poe_rl/data`: Data loading and parsing (JSON mods, item bases).
- `src/poe_rl/engine`: Core crafting logic (applying orbs, calculating outcomes).
- `src/poe_rl/rl`: RL environment and agents (Q-Learning).
- `src/poe_rl/main.py`: Entry point for training and demonstration.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Installation

1.  **Install dependencies:**

    ```bash
    uv sync
    ```

    Or with pip:

    ```bash
    pip install -e .
    ```

## Usage

1.  **Download Data:**
    Ensure you have the `poec_mods.json` file (or similar mod data) available.

2.  **Run Training:**

    Run the main script using `uv run`:

    ```bash
    uv run python -m poe_rl.main --mods poec_mods.json --episodes 5000
    ```

    Arguments:
    - `--mods`: Path to the mods JSON file.
    - `--episodes`: Number of training episodes (default: 1000).
    - `--seed`: Random seed (default: 42).

3.  **Train PPO (Stable-Baselines3):**

    ```bash
    uv run python -m poe_rl.rl.train_ppo --mods src/poe_rl/data/static/poec_data.json --timesteps 100000
    ```

    This spins up `RingCraftingEnvV1`, wraps it with `ActionMasker`, trains `MaskablePPO`,
    and saves the resulting policy under `artifacts/ppo_ring_policy.zip`.

## Development

To add new crafting methods or modify the environment:

- **New Actions:** Add classes to `src/poe_rl/engine/actions.py` and register them in `src/poe_rl/rl/envs/ring_crafting.py`.
- **New Items:** Update `src/poe_rl/data/models.py` or the loading logic in `src/poe_rl/data/loader.py`.
