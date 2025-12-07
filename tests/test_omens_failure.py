import random
import unittest
from unittest.mock import MagicMock

from src.poe_rl.data.models import Essence, Item, ItemBase, Mod
from src.poe_rl.engine.actions import ACTIONS, AddRandomMod, DivineEffect, EssenceEffect, RemoveRandomMod


class TestOmenInteractions(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(1337)
        self.base_item = ItemBase(
            base_id="test_ring",
            name="Test Ring",
            item_class="Ring",
            drop_level=1,
            tags=["ring"],
            implicits=[],
        )
        self.item = Item(base=self.base_item, ilvl=86)
        self.item.rarity = "Rare"

        self.mock_db = MagicMock()
        self.mock_db.essences_by_id = {}

        # Existing mods that seed the homogenising tag filter
        self.prefix_existing = self._make_mod(
            mod_id="existing_fire",
            is_prefix=True,
            group="existing_fire",
            tags=["fire"],
        )
        self.suffix_existing = self._make_mod(
            mod_id="existing_lightning",
            is_prefix=False,
            group="existing_lightning",
            tags=["lightning"],
        )

        # Candidates that either match the homogenising tags or fail them
        self.prefix_homogenised = self._make_mod(
            mod_id="new_fire",
            is_prefix=True,
            group="new_fire",
            tags=["fire"],
        )
        self.prefix_homogenised_extra = self._make_mod(
            mod_id="new_fire_extra",
            is_prefix=True,
            group="new_fire_extra",
            tags=["fire"],
        )
        self.prefix_off_tag = self._make_mod(
            mod_id="new_cold",
            is_prefix=True,
            group="new_cold",
            tags=["cold"],
        )
        self.suffix_homogenised = self._make_mod(
            mod_id="new_lightning",
            is_prefix=False,
            group="new_lightning",
            tags=["lightning"],
        )
        self.suffix_homogenised_extra = self._make_mod(
            mod_id="new_lightning_extra",
            is_prefix=False,
            group="new_lightning_extra",
            tags=["lightning"],
        )
        self.suffix_off_tag = self._make_mod(
            mod_id="new_chaos",
            is_prefix=False,
            group="new_chaos",
            tags=["chaos"],
        )

        self.prefix_candidates = [self.prefix_homogenised, self.prefix_off_tag]
        self.suffix_candidates = [self.suffix_homogenised, self.suffix_off_tag]

        self.mock_db.mods_by_id = {
            mod.mod_id: mod
            for mod in [
                self.prefix_existing,
                self.suffix_existing,
                self.prefix_homogenised,
                self.prefix_homogenised_extra,
                self.prefix_off_tag,
                self.suffix_homogenised,
                self.suffix_homogenised_extra,
                self.suffix_off_tag,
            ]
        }

        self.mock_db.get_mod_candidates.side_effect = self._candidate_stub

    def _make_mod(self, mod_id: str, is_prefix: bool, group: str, tags: list[str]) -> Mod:
        return Mod(
            mod_id=mod_id,
            name=mod_id,
            group=group,
            tier=1,
            generation_type="prefix" if is_prefix else "suffix",
            domain="item",
            ilvl_required=1,
            is_prefix=is_prefix,
            tags=tags,
            weight_tags={"ring": 100},
        )

    def _candidate_stub(self, item: Item, want_prefix: bool) -> list[Mod]:
        pool = self.prefix_candidates if want_prefix else self.suffix_candidates
        # Return a shallow copy so the production code can safely filter the list.
        return list(pool)

    @staticmethod
    def _get_action(name: str):
        for action in ACTIONS:
            if action.name == name:
                return action
        raise AssertionError(f"Crafting action '{name}' not found in registry")

    def test_homogenising_exaltation_with_greater_reverts_on_partial(self) -> None:
        """Partial Greater Exaltation attempts should revert and keep both omens."""

        self.item.prefix_ids.append(self.prefix_existing.mod_id)
        self.item.active_omens.update(
            {"Omen of Homogenising Exaltation", "Omen of Greater Exaltation"}
        )

        action = AddRandomMod(
            count_min=1,
            count_max=1,
            force_type="Prefix",
            allowed_omens=[
                "Omen of Homogenising Exaltation",
                "Omen of Greater Exaltation",
            ],
        )

        with self.assertRaisesRegex(ValueError, "Could not generate mod"):
            action.apply(self.item, self.mock_db, self.rng)

        # The action fails entirely, so no new mods are retained.
        self.assertNotIn(self.prefix_homogenised.mod_id, self.item.prefix_ids)
        self.assertEqual(self.item.prefix_ids, [self.prefix_existing.mod_id])

        # Both omens remain for the next attempt.
        self.assertIn("Omen of Greater Exaltation", self.item.active_omens)
        self.assertIn("Omen of Homogenising Exaltation", self.item.active_omens)

    def test_homogenising_coronation_with_greater_reverts_on_partial(self) -> None:
        """Regal-flow homogenising should match the simulator's revert semantics."""

        self.item.suffix_ids.append(self.suffix_existing.mod_id)
        self.item.active_omens.update(
            {"Omen of Homogenising Coronation", "Omen of Greater Exaltation"}
        )

        action = AddRandomMod(
            count_min=1,
            count_max=1,
            force_type="Suffix",
            allowed_omens=[
                "Omen of Homogenising Coronation",
                "Omen of Greater Exaltation",
            ],
        )

        with self.assertRaisesRegex(ValueError, "Could not generate mod"):
            action.apply(self.item, self.mock_db, self.rng)

        self.assertNotIn(self.suffix_homogenised.mod_id, self.item.suffix_ids)
        self.assertEqual(self.item.suffix_ids, [self.suffix_existing.mod_id])
        self.assertIn("Omen of Greater Exaltation", self.item.active_omens)
        self.assertIn("Omen of Homogenising Coronation", self.item.active_omens)

    def test_homogenising_exaltation_consumes_omens_on_full_success(self) -> None:
        """Greater + Homogenising should consume both omens when two mods land."""

        self.item.prefix_ids.append(self.prefix_existing.mod_id)
        self.item.active_omens.update(
            {"Omen of Homogenising Exaltation", "Omen of Greater Exaltation"}
        )

        # Provide two eligible homogenised mods so both rolls can succeed.
        self.prefix_candidates = [
            self.prefix_homogenised,
            self.prefix_homogenised_extra,
        ]

        action = AddRandomMod(
            count_min=1,
            count_max=1,
            force_type="Prefix",
            allowed_omens=[
                "Omen of Homogenising Exaltation",
                "Omen of Greater Exaltation",
            ],
        )

        action.apply(self.item, self.mock_db, self.rng)

        self.assertIn(self.prefix_homogenised.mod_id, self.item.prefix_ids)
        self.assertIn(self.prefix_homogenised_extra.mod_id, self.item.prefix_ids)
        self.assertNotIn("Omen of Greater Exaltation", self.item.active_omens)
        self.assertNotIn("Omen of Homogenising Exaltation", self.item.active_omens)

    def test_homogenising_prefers_matching_side_without_dextral(self) -> None:
        """Homogenising alone should fall back to whatever side has valid tags."""

        self.item.prefix_ids.append(self.prefix_existing.mod_id)
        self.item.active_omens.add("Omen of Homogenising Exaltation")

        # Make suffix candidates fail the homogenising filter.
        self.suffix_candidates = [self.suffix_off_tag]

        action = AddRandomMod(
            count_min=1,
            count_max=1,
            allowed_omens=["Omen of Homogenising Exaltation"],
        )

        action.apply(self.item, self.mock_db, self.rng)

        self.assertIn(self.prefix_homogenised.mod_id, self.item.prefix_ids)
        self.assertNotIn("Omen of Homogenising Exaltation", self.item.active_omens)

    def test_remove_random_mod_failure_preserves_omen(self) -> None:
        """Dextral annulment should report an error when no suffix exists."""

        self.item.prefix_ids.append(self.prefix_existing.mod_id)
        self.item.active_omens.add("Omen of Dextral Annulment")

        remover = RemoveRandomMod()

        with self.assertRaisesRegex(ValueError, "Could not remove a mod"):
            remover.apply(self.item, self.mock_db, self.rng)

        self.assertEqual(self.item.prefix_ids, [self.prefix_existing.mod_id])
        self.assertIn("Omen of Dextral Annulment", self.item.active_omens)

    def test_essence_conflict_raises_error(self) -> None:
        """Essence craft should fail when a conflicting mod group is present."""

        conflicting_mod = self._make_mod(
            mod_id="conflict_mod",
            is_prefix=True,
            group="essence_group",
            tags=["fire"],
        )
        guaranteed_mod = self._make_mod(
            mod_id="essence_mod",
            is_prefix=True,
            group="essence_group",
            tags=["fire"],
        )

        self.mock_db.mods_by_id[conflicting_mod.mod_id] = conflicting_mod
        self.mock_db.mods_by_id[guaranteed_mod.mod_id] = guaranteed_mod

        self.item.prefix_ids.append(conflicting_mod.mod_id)
        self.item.rarity = "Magic"

        essence = Essence(
            essence_id="test_essence",
            name="Lesser Essence",
            guaranteed_mods={self.item.base.item_class: [guaranteed_mod.mod_id]},
        )
        self.mock_db.essences_by_id = {"test_essence": essence}

        effect = EssenceEffect("test_essence")

        with self.assertRaisesRegex(ValueError, "The item already has a mod of this type"):
            effect.apply(self.item, self.mock_db, self.rng)

        self.assertEqual(self.item.prefix_ids, [conflicting_mod.mod_id])
        self.assertEqual(self.item.rarity, "Magic")

    def test_divine_requires_explicit_mod_without_omen(self) -> None:
        """Divine Orb should fail on blank items when no omen is active."""

        self.item.prefix_ids.clear()
        self.item.suffix_ids.clear()

        effect = DivineEffect()

        with self.assertRaisesRegex(ValueError, "No explicit modifiers to reroll"):
            effect.apply(self.item, self.mock_db, self.rng)

    def test_divine_with_blessed_consumes_omen_and_requires_implicits(self) -> None:
        """Blessed omen should only resolve if the base exposes implicits."""

        self.item.base.implicits = ["+5 all attributes"]
        self.item.active_omens.add("Omen of the Blessed")

        effect = DivineEffect()
        effect.apply(self.item, self.mock_db, self.rng)

        self.assertNotIn("Omen of the Blessed", self.item.active_omens)

        # Removing implicits should now cause a failure
        self.item.base.implicits = []
        self.item.active_omens.add("Omen of the Blessed")
        with self.assertRaisesRegex(ValueError, "No implicit modifiers to reroll"):
            effect.apply(self.item, self.mock_db, self.rng)

    def test_duplicate_omen_application_is_blocked(self) -> None:
        """Applying the same omen twice should fail fast via requirements."""

        action = self._get_action("Omen of Greater Exaltation")
        self.item.active_omens.add("Omen of Greater Exaltation")

        with self.assertRaisesRegex(ValueError, "Omen of Greater Exaltation is already active"):
            action.apply(self.item, self.mock_db, self.rng)


if __name__ == "__main__":
    unittest.main()
