from __future__ import annotations

import random
from typing import List

from ..envs.ring_crafting import RingCraftEnv


def train_q_learning(
    env: RingCraftEnv,
    episodes: int,
    learning_rate: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> List[List[float]]:
    """Train a Q‑learning agent on the ring crafting environment.

    Returns the Q‑table as a nested list: Q[state][action].
    """
    random.seed(seed)
    q_table = [[0.0 for _ in range(env.num_actions)] for _ in range(env.num_states)]
    epsilon = epsilon_start
    for ep in range(episodes):
        state = env.reset()
        state_idx = state.to_index()
        done = False
        while not done:
            # Choose action using epsilon‑greedy
            if random.random() < epsilon:
                action_idx = random.randrange(env.num_actions)
            else:
                max_q = max(q_table[state_idx])
                best_actions = [i for i, q in enumerate(q_table[state_idx]) if q == max_q]
                action_idx = random.choice(best_actions)
            next_state, reward, done = env.step(action_idx)
            next_state_idx = next_state.to_index()
            # Q‑learning update
            target = reward
            if not done:
                target += gamma * max(q_table[next_state_idx])
            q_table[state_idx][action_idx] += learning_rate * (target - q_table[state_idx][action_idx])
            state_idx = next_state_idx
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return q_table
