import random
import unittest
from unittest.mock import MagicMock

from src.poe_rl.data.models import Item, ItemBase, Mod
from src.poe_rl.engine.actions import (
    FRACTURE_EXISTS_MESSAGE,
    FRACTURE_MINIMUM_MESSAGE,
    FractureEffect,
    RemoveRandomMod,
    ScourEffect,
)


class TestFracturingParity(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(2025)
        self.base = ItemBase(
            base_id="test_base",
            name="Test Base",
            item_class="Ring",
            drop_level=1,
            tags=["ring"],
            implicits=[],
        )
        self.item = Item(base=self.base, ilvl=86)
        self.item.rarity = "Rare"

        self.db = MagicMock()
        self.db.mods_by_id = {}

    def _register_mod(self, mod_id: str, is_prefix: bool) -> None:
        mod = Mod(
            mod_id=mod_id,
            name=mod_id,
            group=f"group_{mod_id}",
            tier=1,
            generation_type="prefix" if is_prefix else "suffix",
            domain="item",
            ilvl_required=1,
            is_prefix=is_prefix,
            tags=["test"],
            weight_tags={"ring": 100},
        )
        self.db.mods_by_id[mod_id] = mod

    def test_fracture_requires_four_affixes(self) -> None:
        """Simulator only allows fracturing on items with at least four mods."""

        for idx in range(3):
            is_prefix = idx % 2 == 0
            mod_id = f"mod_{idx}"
            self._register_mod(mod_id, is_prefix=is_prefix)
            if is_prefix:
                self.item.prefix_ids.append(mod_id)
            else:
                self.item.suffix_ids.append(mod_id)

        effect = FractureEffect()

        with self.assertRaisesRegex(ValueError, FRACTURE_MINIMUM_MESSAGE):
            effect.apply(self.item, self.db, self.rng)

    def test_fracture_marks_mod_and_blocks_reapplication(self) -> None:
        """Once a mod is fractured it stays protected and further orbs fail."""

        for idx in range(4):
            is_prefix = idx < 2
            mod_id = f"mod_{idx}"
            self._register_mod(mod_id, is_prefix=is_prefix)
            if is_prefix:
                self.item.prefix_ids.append(mod_id)
            else:
                self.item.suffix_ids.append(mod_id)

        effect = FractureEffect()
        effect.apply(self.item, self.db, self.rng)

        self.assertEqual(len(self.item.fractured_mods), 1)
        fractured_id = self.item.fractured_mods[0]
        self.assertIn(fractured_id, self.item.prefix_ids + self.item.suffix_ids)

        with self.assertRaisesRegex(ValueError, FRACTURE_EXISTS_MESSAGE):
            effect.apply(self.item, self.db, self.rng)

    def test_remove_random_mod_ignores_fractured_mods(self) -> None:
        """Annulment-style effects must skip fractured modifiers entirely."""

        self._register_mod("pref_one", is_prefix=True)
        self._register_mod("suf_one", is_prefix=False)
        self.item.prefix_ids.append("pref_one")
        self.item.suffix_ids.append("suf_one")
        self.item.fractured_mods.append("pref_one")

        remover = RemoveRandomMod()
        remover.apply(self.item, self.db, self.rng)

        self.assertEqual(self.item.prefix_ids, ["pref_one"])
        self.assertEqual(self.item.suffix_ids, [])
        self.assertEqual(self.item.fractured_mods, ["pref_one"])

    def test_scour_effect_preserves_fractured_rarity(self) -> None:
        """Orb of Scouring should leave fractured affixes and update rarity."""

        self._register_mod("pref_frac", is_prefix=True)
        self._register_mod("pref_extra", is_prefix=True)
        self._register_mod("suf_frac", is_prefix=False)

        self.item.prefix_ids.extend(["pref_frac", "pref_extra"])
        self.item.suffix_ids.append("suf_frac")
        self.item.fractured_mods.extend(["pref_frac", "suf_frac"])

        ScourEffect().apply(self.item, self.db, self.rng)

        self.assertEqual(self.item.prefix_ids, ["pref_frac"])
        self.assertEqual(self.item.suffix_ids, ["suf_frac"])
        self.assertEqual(self.item.rarity, "Rare")


if __name__ == "__main__":
    unittest.main()
