from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Tuple

from ..data.models import Item
from .actions import CraftingAction
from .database import CraftingDatabase
from .price import PriceProvider


@dataclass
class CraftingEngine:
    """
    Core engine for applying crafting actions to items.

    The engine uses a :class:`CraftingDatabase` to look up mods and bases and a
    :class:`PriceProvider` to compute currency costs.  Randomness is supplied
    via a :class:`random.Random` instance to allow reproducible simulations.

    Parameters
    ----------
    db:
        Database containing mods and base items.
    price_provider:
        Object that converts currency names into chaos values.
    rng:
        Random number generator.  If omitted, a new instance is created.
    """

    db: CraftingDatabase
    price_provider: PriceProvider
    rng: random.Random = field(default_factory=random.Random)

    def create_item(self, base_id: str, ilvl: int) -> Item:
        """Instantiate a white item of the given base and item level."""
        base = self.db.bases_by_id.get(base_id)
        if base is None:
            raise KeyError(f"Unknown base id: {base_id}")
        return Item(base=base, ilvl=ilvl)

    def apply(self, item: Item, action: CraftingAction) -> Tuple[Item, float]:
        """
        Apply a crafting action to an item and return the result and cost.

        Parameters
        ----------
        item:
            The item to modify.  It is not mutated; a new instance is
            returned.
        action:
            The crafting action to apply.  Its ``apply`` method will be
            invoked with the same database and random generator.

        Returns
        -------
        (Item, float)
            A tuple of the new item and the chaos‑equivalent cost of the
            currency consumed.
        """
        new_item = action.apply(item, self.db, self.rng)
        # Use currency_key if available (new system), else currency_name (legacy/fallback)
        key = getattr(action, "currency_key", getattr(action, "currency_name", "unknown"))
        cost = self.price_provider.get_price(key)
        return new_item, cost
