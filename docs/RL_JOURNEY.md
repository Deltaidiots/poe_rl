# RL Journey: Training a PPO Agent for PoE2 Crafting

This document chronicles the development of the RL environment and reward function, lessons learned about reward hacking, and mitigations we've implemented.

## Table of Contents

1. [The Problem](#the-problem)
2. [Environment Design](#environment-design)
3. [Reward Function Evolution](#reward-function-evolution)
4. [Reward Hacking Observations](#reward-hacking-observations)
5. [Mitigations](#mitigations)
6. [Training Insights](#training-insights)
7. [Future Directions](#future-directions)

---

## The Problem

**Goal:** Train an RL agent to craft high-quality rings in Path of Exile 2, optimizing for:
- +Maximum Life mods (Tier 1-3)
- +Elemental/Chaos Resistance mods (2+ mods, Tier 1-3)
- Minimizing currency costs

**Challenge:** PoE2 crafting is fundamentally stochastic:
- Chaos Orb: Completely rerolls all mods (pure RNG)
- Essences: Guarantee one specific mod + random others
- Exalted Orb: Adds one random mod
- Annulment: Removes one random mod

The agent must learn when RNG-heavy actions are worth the risk vs. deterministic alternatives.

---

## Environment Design

### State Space (99 dimensions)

```python
# Global features (12)
- Rarity (one-hot: Normal, Magic, Rare)
- Prefix slots (current/max)
- Suffix slots (current/max)
- Corruption flag
- Fracture mod count
- Item level (normalized)
- Quality (normalized)

# Mod group features (56 = 14 groups × 4 features each)
- For each life/resistance group:
  - Count of mods in group
  - Best tier achieved
  - Total value
  - Presence flag

# Aggregate features (7)
- Total affixes
- Prefix ratio
- Total tier score
- Life mod count
- Resistance mod count
- Goal progress (life)
- Goal progress (resistance)

# Tag features (16 = 8 tags × 2)
- Tag presence in prefixes
- Tag presence in suffixes

# Goal features (8)
- Life goal progress
- Resistance goal progress
- Life tier average
- Resistance tier average
- Life count
- Resistance count
- Goal achievement flags
```

### Action Space (110 discrete actions)

```
17 Base Currency Actions:
  - Transmutation (3 tiers)
  - Augmentation (3 tiers)
  - Regal (3 tiers)
  - Alchemy (1)
  - Chaos (3 tiers)
  - Exalted (3 tiers)
  - Annulment (1)

81 Essence Actions (dynamically loaded):
  - All essences that can roll on Rings
  - Lesser/Greater/Perfect tiers for each

11 Omen Actions:
  - Exaltation omens (targeted prefix/suffix)
  - Annulment omens
  - Crystallisation omens (fracturing)
  - Erasure omens

1 Stop Action:
  - End episode and collect final reward
```

### Action Masking

We use `sb3_contrib.MaskablePPO` to prevent invalid actions:

```python
def get_action_mask(self) -> np.ndarray:
    mask = np.zeros(self.action_space.n, dtype=np.int8)
    for idx, action in enumerate(self._actions):
        if self._can_execute(action):
            mask[idx] = 1
    return mask
```

This prevents the agent from:
- Using Chaos on Normal items (requires Rare)
- Using Exalted when all slots are full
- Using Transmutation on Rare items
- etc.

---

## Reward Function Evolution

### Version 1: Simple Goal-Based

```python
reward = 0
if goal_achieved:
    reward = 20.0
reward -= cost * 0.01
```

**Result:** Agent never learned – reward too sparse.

### Version 2: Progress Shaping

```python
reward = 0
# Reward for adding resistance mods
reward += new_res_mods * 1.2
# Bonus for tier improvements
reward += tier_improvement * 0.85
# Milestone bonuses (1, 2, 3 res mods)
reward += milestone_bonus
# Penalty for losing progress
reward -= res_lost * 0.35
# Cost penalty
reward -= cost * 0.005
```

**Result:** Agent learned to make progress but discovered reward hacking...

### Version 3: Anti-Exploitation Additions

```python
# Repeat action penalties
if action in recent_actions:
    reward -= repeat_penalty

# Deterministic action bonuses (essences, omens)
if is_deterministic_action:
    reward += 0.4
    cost *= 0.6  # Discounted cost

# Omen utilization
if omen_consumed_quickly:
    reward += 0.35
if omen_wasted:
    reward -= 0.5
```

### Version 4: Essence & RNG-Specific (Current)

```python
# Essence bonuses
if is_essence_action:
    reward += essence_guaranteed_mod_bonus  # 0.8
    if adds_goal_aligned_mod:
        reward += essence_goal_aligned_bonus  # 1.2
    if step < 10:
        reward += essence_early_use_bonus  # 0.5

# Chaos spam penalty
if chaos_use_count > chaos_spam_threshold:
    extra_uses = chaos_use_count - threshold
    reward -= extra_uses * chaos_spam_penalty_ramp  # 0.05 per use

# RNG action base penalty
if is_rng_action:
    reward -= rng_action_base_penalty  # 0.02
```

---

## Reward Hacking Observations

### Problem 1: Chaos Orb Spam

**Behavior:** Agent learned to repeatedly use Chaos Orb, exploiting the fact that:
- Sometimes it rolls good mods → positive reward
- Cost is low relative to potential positive spikes
- No penalty for regression (we later added one)

**Training logs showed:**
```
Step 100: Chaos Orb → reward 0.3
Step 101: Chaos Orb → reward -0.1  
Step 102: Chaos Orb → reward 0.8  ← Lucky roll
Step 103: Chaos Orb → reward -0.2
...
```

The variance in rewards encouraged risk-seeking behavior.

### Problem 2: Omen Hoarding

**Behavior:** Agent applied omens but never used them, because:
- Applying omen → small positive reward for "setup"
- Actually using omen → risk of losing mods

### Problem 3: Ignoring Essences

**Behavior:** Essences were rarely used despite guaranteeing specific mods:
- Original action space only had 14 essences (now 81)
- Mod ID resolution was broken (essence couldn't find target mod)
- Cost appeared higher than chaos (but chaos needed more uses)

---

## Mitigations

### 1. Repeat Action Penalties

```python
if action_family in recent_actions[-3:]:
    reward -= repeat_action_penalty * streak_length
```

Prevents any single action from being spammed indefinitely.

### 2. Deterministic Action Bonuses

Actions with predictable outcomes get bonuses:

```python
DETERMINISTIC_ACTIONS = {
    "essences": 0.4 bonus + 0.6 cost discount
    "omens + exalt": same
}
```

### 3. Slot Congestion Pressure

When suffix slots stay full for too long without progress:

```python
if steps_since_suffix_change > grace_period:
    reward -= congestion_penalty * (1 + growth * overtime)
```

Forces the agent to make decisions rather than churn.

### 4. Structural Progress Rewards

Reward net gains in affix count, regardless of which mods:

```python
affix_delta = new_affixes - old_affixes
if affix_delta > 0:
    reward += structural_affix_gain_reward
else:
    reward -= structural_affix_loss_penalty * abs(delta)
```

### 5. Omen Utilization Tracking

```python
if omen_consumed_within_N_steps:
    reward += timely_consumption_bonus
if episode_ends_with_unused_omen:
    reward -= unconsumed_omen_penalty
```

### 6. Chaos Spam Escalating Penalty

```python
if chaos_uses > threshold:
    penalty = (chaos_uses - threshold) * ramp_rate
    reward -= penalty
```

### 7. Expanded Essence Action Space

Changed from static 14 essences to dynamic 81:

```python
DYNAMIC_ESSENCE_ACTIONS = True  # Load all from database
```

And fixed the mod resolution bug so essences actually work.

---

## Training Insights

### What Works

1. **Action Masking:** Crucial for sample efficiency – agent doesn't waste time on impossible actions
2. **Progress Shaping:** Milestones for 1, 2, 3 resistance mods provide clear learning signal
3. **Cost Discounts:** Making deterministic actions "cheaper" in reward terms encourages their use
4. **Early Use Bonuses:** Nudging essence use in first 10 steps leads to better starting states

### What Doesn't Work

1. **Too Much Penalty:** Heavy repeat penalties caused agent to cycle through actions randomly
2. **Sparse Rewards:** Goal-only rewards led to no learning
3. **Ignoring Variance:** Must explicitly penalize high-variance strategies

### Hyperparameter Sensitivity

| Parameter | Too Low | Too High | Sweet Spot |
|-----------|---------|----------|------------|
| `cost_coefficient` | Agent ignores cost | Agent too conservative | 0.005 |
| `repeat_action_penalty` | Spam continues | Random cycling | 0.04 |
| `chaos_spam_threshold` | Hits immediately | No effect | 5 |
| `essence_guaranteed_mod_bonus` | Still ignored | Overused | 0.8 |

---

## Future Directions

### 1. Curriculum Learning

Start with simpler goals (any 1 resistance mod) and progressively increase difficulty.

### 2. Multi-Objective Rewards

Separate reward channels with tunable weights:
```python
RewardChannelWeights(
    general=1.0,
    target_specific=1.0,
    omen=1.0,
    cost=1.0,
    guardrail=1.0
)
```

### 3. Intrinsic Motivation

Reward exploration of underused actions:
```python
curiosity_bonus = 1 / sqrt(action_visit_count + 1)
```

### 4. State Abstraction

The 99-dim observation might be simplified:
- PCA on mod features
- Learned embeddings for mod types

### 5. Model-Based RL

Since we have a simulator, could train a world model and plan.

---

## Reproducing Results

### Quick Training Run

```bash
uv run python -m poe_rl.rl.train_ppo \
    --mods src/poe_rl/data/static/poec_data.json \
    --timesteps 100000 \
    --save-path artifacts/my_policy.zip
```

### Evaluate a Policy

```bash
uv run python -m poe_rl.rl.eval_ppo \
    --policy artifacts/ppo_ring_policy.zip \
    --episodes 100
```

### View TensorBoard Logs

```bash
tensorboard --logdir runs/ppo/
```

---

## Key Takeaways

1. **Reward hacking is inevitable** – design for it from the start
2. **Deterministic > Stochastic** – explicitly prefer predictable outcomes
3. **Progressive penalties** – ramp costs for repeated behaviors
4. **Action masking is essential** – reduces exploration burden significantly
5. **Debug your data layer** – a broken mod lookup = wasted training time
