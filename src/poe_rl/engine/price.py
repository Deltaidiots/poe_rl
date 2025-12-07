from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


class PriceProvider(Protocol):
    """Interface for converting currency identifiers into chaos‑equivalent costs."""

    def get_price(self, currency_name: str) -> float:
        """Return the chaos value of the given currency identifier."""
        ...


@dataclass
class StaticPriceProvider:
    """
    Simple price provider mapping currency names to static values.

    Use this when simulating crafting in a vacuum.  A more sophisticated
    implementation could query poe.ninja or another pricing API.
    """

    prices: Dict[str, float]

    def get_price(self, currency_name: str) -> float:
        return self.prices.get(currency_name, 0.0)
