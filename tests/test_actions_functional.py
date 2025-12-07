import random
import unittest
from typing import cast

from src.poe_rl.data.models import Essence, Item, ItemBase, Mod
from src.poe_rl.engine.actions import (
    ACTIONS,
    CraftingAction,
    SetOmen,
    create_essence_actions,
    get_action_by_name,
)
from src.poe_rl.engine.database import CraftingDatabase


def _make_mod(
    mod_id: str,
    *,
    is_prefix: bool,
    group: str | None = None,
    ilvl: int = 1,
    tags: list[str] | None = None,
) -> Mod:
    group_name = group or f"grp_{mod_id}"
    return Mod(
        mod_id=mod_id,
        name=mod_id,
        group=group_name,
        tier=1,
        generation_type="prefix" if is_prefix else "suffix",
        domain="item",
        ilvl_required=ilvl,
        is_prefix=is_prefix,
        tags=tags or ["crafted"],
        weight_tags={"ring": 100},
    )


class StubDatabase(CraftingDatabase):
    """Controlled candidate selection to mirror deterministic JS scenarios."""

    def __init__(self, mods: dict[str, Mod], base: ItemBase) -> None:
        super().__init__(mods_by_id=mods, bases_by_id={base.base_id: base}, essences_by_id={})
        self.prefix_pool: list[Mod] = []
        self.suffix_pool: list[Mod] = []

    def set_pools(self, *, prefix: list[Mod] | None = None, suffix: list[Mod] | None = None) -> None:
        self.prefix_pool = prefix or []
        self.suffix_pool = suffix or []

    def get_mod_candidates(self, item: Item, want_prefix: bool) -> list[Mod]:  # type: ignore[override]
        pool = self.prefix_pool if want_prefix else self.suffix_pool
        return list(pool)


class TestActionParity(unittest.TestCase):
    def setUp(self) -> None:
        self.base = ItemBase(
            base_id="ring_base",
            name="Test Ring",
            item_class="Ring",
            drop_level=1,
            tags=["ring"],
            implicits=[],
        )

    def _make_item(self, *, rarity: str = "Normal", ilvl: int = 86) -> Item:
        item = Item(base=self.base, ilvl=ilvl)
        item.rarity = rarity
        return item

    def _make_db(self, mods: list[Mod], base: ItemBase | None = None) -> StubDatabase:
        base_obj = base or self.base
        return StubDatabase({mod.mod_id: mod for mod in mods}, base_obj)

    def _make_full_db(
        self,
        mods: list[Mod],
        *,
        essences: dict[str, Essence] | None = None,
        base: ItemBase | None = None,
    ) -> CraftingDatabase:
        base_obj = base or self.base
        return CraftingDatabase(
            mods_by_id={mod.mod_id: mod for mod in mods},
            bases_by_id={base_obj.base_id: base_obj},
            essences_by_id=essences or {},
        )

    def _get_action(self, name: str) -> CraftingAction:
        action = get_action_by_name(name)
        self.assertIsNotNone(action, f"Action '{name}' is missing from registry")
        assert action is not None
        return action

    def _apply_action(
        self,
        name: str,
        item: Item,
        db: CraftingDatabase,
        *,
        prefix_pool: list[Mod] | None = None,
        suffix_pool: list[Mod] | None = None,
        seed: int = 0,
    ) -> Item:
        action = self._get_action(name)
        if isinstance(db, StubDatabase):
            db.set_pools(prefix=prefix_pool, suffix=suffix_pool)
        rng = random.Random(seed)
        return action.apply(item, db, rng)

    def test_regal_with_dextral_coronation_forces_suffix(self) -> None:
        """Dextral Coronation should mirror JS by forcing a suffix roll."""

        prefix_existing = _make_mod("pref_existing", is_prefix=True)
        regal_suffix = _make_mod("regal_suffix", is_prefix=False)

        db = self._make_db([prefix_existing, regal_suffix])

        item = self._make_item(rarity="Magic")
        item.prefix_ids.append(prefix_existing.mod_id)
        item.active_omens.add("Omen of Dextral Coronation")

        result = self._apply_action(
            "Regal Orb",
            item,
            db,
            prefix_pool=[prefix_existing],
            suffix_pool=[regal_suffix],
            seed=1337,
        )

        self.assertEqual(result.rarity, "Rare")
        self.assertIn(regal_suffix.mod_id, result.suffix_ids)
        self.assertEqual(result.prefix_ids.count(prefix_existing.mod_id), 1)
        self.assertNotIn("Omen of Dextral Coronation", result.active_omens)

    def test_chaos_orb_whittling_removes_lowest_level_mod(self) -> None:
        """Whittling omen should target the lowest level modifier before rerolling."""

        prefix_filler = [
            _make_mod(f"pref_{idx}", is_prefix=True, group=f"pref_grp_{idx}", ilvl=100 + idx)
            for idx in range(3)
        ]
        suffix_low = _make_mod("suffix_low", is_prefix=False, ilvl=10)
        suffix_high = _make_mod("suffix_high", is_prefix=False, ilvl=80)
        chaos_new = _make_mod("suffix_new", is_prefix=False, group="suffix_new_grp")

        db = self._make_db(prefix_filler + [suffix_low, suffix_high, chaos_new])

        item = self._make_item(rarity="Rare")
        item.prefix_ids.extend(mod.mod_id for mod in prefix_filler)
        item.suffix_ids.extend([suffix_low.mod_id, suffix_high.mod_id])
        item.active_omens.add("Omen of Whittling")

        result = self._apply_action(
            "Chaos Orb",
            item,
            db,
            prefix_pool=prefix_filler,
            suffix_pool=[chaos_new],
            seed=9001,
        )

        self.assertNotIn(suffix_low.mod_id, result.suffix_ids)
        self.assertIn(suffix_high.mod_id, result.suffix_ids)
        self.assertIn(chaos_new.mod_id, result.suffix_ids)
        self.assertNotIn("Omen of Whittling", result.active_omens)
        self.assertEqual(result.prefix_ids, item.prefix_ids)

    def test_transmutation_tiers_enforce_level_filters(self) -> None:
        pref_low = _make_mod("transmute_low", is_prefix=True, ilvl=1)
        pref_mid = _make_mod("transmute_mid", is_prefix=True, ilvl=60)
        pref_high = _make_mod("transmute_high", is_prefix=True, ilvl=80)
        db = self._make_db([pref_low, pref_mid, pref_high])

        normal = self._make_item(rarity="Normal", ilvl=20)
        result = self._apply_action(
            "Orb of Transmutation",
            normal,
            db,
            prefix_pool=[pref_low],
            seed=5,
        )
        self.assertEqual(result.rarity, "Magic")
        self.assertEqual(result.prefix_ids, [pref_low.mod_id])

        with self.assertRaisesRegex(ValueError, "Item level must be at least 55"):
            self._apply_action(
                "Greater Orb of Transmutation",
                self._make_item(rarity="Normal", ilvl=40),
                db,
                prefix_pool=[pref_mid],
            )

        greater_target = self._make_item(rarity="Normal", ilvl=60)
        greater = self._apply_action(
            "Greater Orb of Transmutation",
            greater_target,
            db,
            prefix_pool=[pref_low, pref_mid],
            seed=7,
        )
        self.assertEqual(greater.prefix_ids, [pref_mid.mod_id])

        with self.assertRaisesRegex(ValueError, "Item level must be at least 70"):
            self._apply_action(
                "Perfect Orb of Transmutation",
                self._make_item(rarity="Normal", ilvl=65),
                db,
                prefix_pool=[pref_high],
            )

        perfect_target = self._make_item(rarity="Normal", ilvl=75)
        perfect = self._apply_action(
            "Perfect Orb of Transmutation",
            perfect_target,
            db,
            prefix_pool=[pref_high],
            seed=11,
        )
        self.assertEqual(perfect.prefix_ids, [pref_high.mod_id])

    def test_augmentation_variants_require_open_slot_and_level(self) -> None:
        pref_low = _make_mod("aug_low", is_prefix=True, ilvl=10)
        pref_high = _make_mod("aug_high", is_prefix=True, ilvl=80)
        suffix_fill = _make_mod("aug_suffix_fill", is_prefix=False, ilvl=1)
        db = self._make_db([pref_low, pref_high, suffix_fill])

        full_item = self._make_item(rarity="Magic")
        full_item.prefix_ids.append(pref_low.mod_id)
        full_item.suffix_ids.append(suffix_fill.mod_id)
        with self.assertRaisesRegex(ValueError, "Item has no open affix slots"):
            self._apply_action(
                "Orb of Augmentation",
                full_item,
                db,
                prefix_pool=[pref_high],
            )

        greater_target = self._make_item(rarity="Magic", ilvl=80)
        greater = self._apply_action(
            "Greater Orb of Augmentation",
            greater_target,
            db,
            prefix_pool=[pref_low, pref_high],
            seed=13,
        )
        self.assertEqual(greater.prefix_ids, [pref_high.mod_id])

        with self.assertRaisesRegex(ValueError, "Item level must be at least 70"):
            self._apply_action(
                "Perfect Orb of Augmentation",
                self._make_item(rarity="Magic", ilvl=65),
                db,
                prefix_pool=[pref_high],
            )

        perfect_target = self._make_item(rarity="Magic", ilvl=80)
        perfect = self._apply_action(
            "Perfect Orb of Augmentation",
            perfect_target,
            db,
            prefix_pool=[pref_high],
            seed=17,
        )
        self.assertEqual(perfect.prefix_ids, [pref_high.mod_id])

    def test_alteration_rerolls_magic_affixes(self) -> None:
        pref_existing = _make_mod("alter_existing", is_prefix=True)
        suf_existing = _make_mod("alter_suffix", is_prefix=False)
        pref_new = _make_mod("alter_new", is_prefix=True)
        db = self._make_db([pref_existing, suf_existing, pref_new])

        item = self._make_item(rarity="Magic")
        item.prefix_ids.append(pref_existing.mod_id)
        item.suffix_ids.append(suf_existing.mod_id)

        result = self._apply_action(
            "Orb of Alteration",
            item,
            db,
            prefix_pool=[pref_new],
            suffix_pool=[suf_existing],
            seed=19,
        )

        self.assertEqual(result.rarity, "Magic")
        self.assertEqual(result.prefix_ids, [pref_new.mod_id])
        self.assertEqual(result.suffix_ids, [])

    def test_alchemy_with_dextral_omen_fills_suffixes(self) -> None:
        prefix_mods = [
            _make_mod(f"alchemy_prefix_{idx}", is_prefix=True, group=f"ap_{idx}")
            for idx in range(3)
        ]
        suffix_mods = [
            _make_mod(f"alchemy_suffix_{idx}", is_prefix=False, group=f"as_{idx}")
            for idx in range(3)
        ]
        db = self._make_db(prefix_mods + suffix_mods)

        item = self._make_item(rarity="Normal")
        item.active_omens.add("Omen of Dextral Alchemy")

        result = self._apply_action(
            "Orb of Alchemy",
            item,
            db,
            prefix_pool=prefix_mods,
            suffix_pool=suffix_mods,
            seed=23,
        )

        self.assertEqual(result.rarity, "Rare")
        self.assertEqual(len(result.suffix_ids), 3)
        total_mods = len(result.prefix_ids) + len(result.suffix_ids)
        self.assertGreaterEqual(total_mods, 4)
        self.assertLessEqual(total_mods, 6)
        self.assertNotIn("Omen of Dextral Alchemy", result.active_omens)

    def test_chaos_tiers_respect_level_filters(self) -> None:
        prefix_low = _make_mod("chaos_low", is_prefix=True, ilvl=5)
        prefix_high = _make_mod("chaos_high", is_prefix=True, ilvl=60)
        suffix_target = _make_mod("chaos_suffix", is_prefix=False, ilvl=1)
        db = self._make_db([prefix_low, prefix_high, suffix_target])

        item = self._make_item(rarity="Rare", ilvl=80)
        item.suffix_ids.append(suffix_target.mod_id)

        greater = self._apply_action(
            "Greater Chaos Orb",
            item,
            db,
            prefix_pool=[prefix_low, prefix_high],
            suffix_pool=[],
            seed=29,
        )
        self.assertNotIn(suffix_target.mod_id, greater.suffix_ids)
        self.assertEqual(greater.prefix_ids, [prefix_high.mod_id])

        with self.assertRaisesRegex(ValueError, "Item level must be at least 50"):
            self._apply_action(
                "Perfect Chaos Orb",
                self._make_item(rarity="Rare", ilvl=45),
                db,
                prefix_pool=[prefix_high],
            )

        perfect_item = self._make_item(rarity="Rare", ilvl=80)
        perfect_item.suffix_ids.append(suffix_target.mod_id)
        perfect = self._apply_action(
            "Perfect Chaos Orb",
            perfect_item,
            db,
            prefix_pool=[prefix_high],
            suffix_pool=[],
            seed=31,
        )
        self.assertEqual(perfect.prefix_ids, [prefix_high.mod_id])

    def test_exalted_omens_force_types_counts_and_tags(self) -> None:
        pref_existing = _make_mod("exalt_existing", is_prefix=True, tags=["fire"])
        pref_homog = _make_mod("exalt_homog", is_prefix=True, tags=["fire"])
        pref_other = _make_mod("exalt_other", is_prefix=True, tags=["cold"])
        suffix_option = _make_mod("exalt_suffix", is_prefix=False)
        db = self._make_db([pref_existing, pref_homog, pref_other, suffix_option])

        dextral_item = self._make_item(rarity="Rare")
        dextral_item.active_omens.add("Omen of Dextral Exaltation")
        dextral = self._apply_action(
            "Exalted Orb",
            dextral_item,
            db,
            prefix_pool=[pref_homog],
            suffix_pool=[suffix_option],
            seed=37,
        )
        self.assertIn(pref_homog.mod_id, dextral.prefix_ids)
        self.assertNotIn("Omen of Dextral Exaltation", dextral.active_omens)

        greater_item = self._make_item(rarity="Rare")
        greater_item.active_omens.add("Omen of Greater Exaltation")
        greater = self._apply_action(
            "Exalted Orb",
            greater_item,
            db,
            prefix_pool=[pref_homog, pref_other],
            suffix_pool=[suffix_option],
            seed=41,
        )
        self.assertEqual(len(greater.prefix_ids) + len(greater.suffix_ids), 2)

        homog_item = self._make_item(rarity="Rare")
        homog_item.prefix_ids.append(pref_existing.mod_id)
        homog_item.active_omens.add("Omen of Homogenising Exaltation")
        homogenised = self._apply_action(
            "Exalted Orb",
            homog_item,
            db,
            prefix_pool=[pref_homog, pref_other],
            suffix_pool=[suffix_option],
            seed=43,
        )
        self.assertIn(pref_homog.mod_id, homogenised.prefix_ids)
        self.assertNotIn(pref_other.mod_id, homogenised.prefix_ids)

    def test_annulment_with_light_targets_desecrated(self) -> None:
        desecrated = _make_mod("annul_desecrated", is_prefix=False, tags=["desecrated"])
        keeper = _make_mod("annul_keeper", is_prefix=True)
        db = self._make_db([desecrated, keeper])

        item = self._make_item(rarity="Rare")
        item.prefix_ids.append(keeper.mod_id)
        item.suffix_ids.append(desecrated.mod_id)
        item.active_omens.add("Omen of Light")

        result = self._apply_action("Orb of Annulment", item, db, seed=47)
        self.assertIn(keeper.mod_id, result.prefix_ids)
        self.assertNotIn(desecrated.mod_id, result.suffix_ids)
        self.assertNotIn("Omen of Light", result.active_omens)

    def test_scouring_preserves_fractured_mods_and_handles_resurgence(self) -> None:
        pref_keep = _make_mod("fractured_keep", is_prefix=True)
        suffix_drop = _make_mod("scour_drop", is_prefix=False)
        extra_prefix = _make_mod("scour_extra_pref", is_prefix=True)
        extra_suffix = _make_mod("scour_extra_suf", is_prefix=False)
        db = self._make_db([pref_keep, suffix_drop, extra_prefix, extra_suffix])

        item = self._make_item(rarity="Rare")
        item.prefix_ids.append(pref_keep.mod_id)
        item.suffix_ids.append(suffix_drop.mod_id)
        item.fractured_mods.append(pref_keep.mod_id)

        result = self._apply_action("Orb of Scouring", item, db, seed=53)
        self.assertEqual(result.prefix_ids, [pref_keep.mod_id])
        self.assertEqual(result.suffix_ids, [])
        self.assertEqual(result.rarity, "Magic")

        resurgence_item = self._make_item(rarity="Rare")
        resurgence_item.prefix_ids.extend([pref_keep.mod_id, extra_prefix.mod_id])
        resurgence_item.suffix_ids.append(extra_suffix.mod_id)
        resurgence_item.active_omens.add("Omen of Resurgence")
        resurgence = self._apply_action(
            "Orb of Scouring",
            resurgence_item,
            db,
            prefix_pool=[pref_keep, extra_prefix],
            suffix_pool=[suffix_drop, extra_suffix],
            seed=59,
        )
        total = len(resurgence.prefix_ids) + len(resurgence.suffix_ids)
        self.assertGreaterEqual(total, 4)
        self.assertLessEqual(total, 6)
        self.assertEqual(resurgence.rarity, "Rare")

    def test_orb_of_chance_rolls_magic_or_rare(self) -> None:
        prefix_mods = [_make_mod(f"chance_pref_{idx}", is_prefix=True, group=f"cp_{idx}") for idx in range(4)]
        suffix_mods = [_make_mod(f"chance_suf_{idx}", is_prefix=False, group=f"cs_{idx}") for idx in range(4)]
        db = self._make_db(prefix_mods + suffix_mods)

        magic_result = self._apply_action(
            "Orb of Chance",
            self._make_item(rarity="Normal"),
            db,
            prefix_pool=prefix_mods,
            suffix_pool=suffix_mods,
            seed=1,
        )
        self.assertEqual(magic_result.rarity, "Magic")
        total_magic = len(magic_result.prefix_ids) + len(magic_result.suffix_ids)
        self.assertGreaterEqual(total_magic, 1)
        self.assertLessEqual(total_magic, 2)

        rare_result = self._apply_action(
            "Orb of Chance",
            self._make_item(rarity="Normal"),
            db,
            prefix_pool=prefix_mods,
            suffix_pool=suffix_mods,
            seed=0,
        )
        self.assertEqual(rare_result.rarity, "Rare")
        total_rare = len(rare_result.prefix_ids) + len(rare_result.suffix_ids)
        self.assertGreaterEqual(total_rare, 4)
        self.assertLessEqual(total_rare, 6)

    def test_vaal_orb_corrupts_and_can_brick(self) -> None:
        prefix_mods = [_make_mod(f"vaal_pref_{idx}", is_prefix=True, group=f"vp_{idx}") for idx in range(6)]
        suffix_mods = [_make_mod(f"vaal_suf_{idx}", is_prefix=False, group=f"vs_{idx}") for idx in range(6)]
        db = self._make_db(prefix_mods + suffix_mods)

        steady_item = self._make_item(rarity="Magic")
        steady_item.prefix_ids.append(prefix_mods[0].mod_id)
        steady = self._apply_action(
            "Vaal Orb",
            steady_item,
            db,
            prefix_pool=prefix_mods,
            suffix_pool=suffix_mods,
            seed=0,
        )
        self.assertTrue(steady.corrupted)
        self.assertEqual(steady.prefix_ids, steady_item.prefix_ids)

        brick_item = self._make_item(rarity="Rare")
        brick = self._apply_action(
            "Vaal Orb",
            brick_item,
            db,
            prefix_pool=prefix_mods,
            suffix_pool=suffix_mods,
            seed=1,
        )
        self.assertTrue(brick.corrupted)
        total_brick = len(brick.prefix_ids) + len(brick.suffix_ids)
        self.assertGreaterEqual(total_brick, 4)
        self.assertLessEqual(total_brick, 6)
        self.assertEqual(brick.rarity, "Rare")

    def test_divine_orb_requires_valid_targets(self) -> None:
        db = self._make_db([])

        with self.assertRaisesRegex(ValueError, "No explicit modifiers to reroll"):
            self._apply_action("Divine Orb", self._make_item(rarity="Magic"), db)

        omen_item = self._make_item(rarity="Rare")
        omen_item.active_omens.add("Omen of the Blessed")
        with self.assertRaisesRegex(ValueError, "No implicit modifiers to reroll"):
            self._apply_action("Divine Orb", omen_item, db)

        implicit_base = ItemBase(
            base_id="implicit_ring",
            name="Implicit Ring",
            item_class="Ring",
            drop_level=1,
            tags=["ring"],
            implicits=["+1 All Attributes"],
        )
        implicit_db = self._make_db([], base=implicit_base)
        blessed_item = Item(base=implicit_base, ilvl=86)
        blessed_item.rarity = "Rare"
        blessed_item.active_omens.add("Omen of the Blessed")
        blessed_result = self._apply_action("Divine Orb", blessed_item, implicit_db)
        self.assertNotIn("Omen of the Blessed", blessed_result.active_omens)

        explicit_item = self._make_item(rarity="Rare")
        explicit_mod = _make_mod("divine_explicit", is_prefix=True)
        explicit_db = self._make_db([explicit_mod])
        explicit_item.prefix_ids.append(explicit_mod.mod_id)
        explicit_result = self._apply_action("Divine Orb", explicit_item, explicit_db)
        self.assertEqual(explicit_result.prefix_ids, [explicit_mod.mod_id])

    def test_fracturing_rules(self) -> None:
        mods = [_make_mod(f"fracture_mod_{idx}", is_prefix=(idx % 2 == 0)) for idx in range(4)]
        db = self._make_db(mods)

        too_few = self._make_item(rarity="Rare")
        too_few.prefix_ids.append(mods[0].mod_id)
        with self.assertRaisesRegex(ValueError, "Not enough modifiers to fracture"):
            self._apply_action("Fracturing Orb", too_few, db)

        already_fractured = self._make_item(rarity="Rare")
        already_fractured.prefix_ids.extend([mods[0].mod_id, mods[2].mod_id])
        already_fractured.suffix_ids.extend([mods[1].mod_id, mods[3].mod_id])
        already_fractured.fractured_mods.append(mods[0].mod_id)
        with self.assertRaisesRegex(ValueError, "Item already has a fractured modifier"):
            self._apply_action("Fracturing Orb", already_fractured, db)

        target = self._make_item(rarity="Rare")
        target.prefix_ids.extend([mods[0].mod_id, mods[2].mod_id])
        target.suffix_ids.extend([mods[1].mod_id, mods[3].mod_id])
        fractured = self._apply_action("Fracturing Orb", target, db, seed=61)
        self.assertEqual(len(fractured.fractured_mods), 1)
        self.assertIn(fractured.fractured_mods[0], target.prefix_ids + target.suffix_ids)

    def test_catalyst_sets_quality_to_twenty(self) -> None:
        db = self._make_db([])
        item = self._make_item(rarity="Rare")
        item.quality = 5
        result = self._apply_action("Catalyst", item, db)
        self.assertEqual(result.quality, 20)

    def test_all_set_omen_actions_apply_state(self) -> None:
        db = self._make_db([])
        for action in ACTIONS:
            if not action.effects:
                continue
            if not all(isinstance(effect, SetOmen) for effect in action.effects):
                continue
            with self.subTest(action=action.name):
                item = self._make_item()
                result = action.apply(item, db, random.Random(0))
                omen_effect = cast(SetOmen, action.effects[0])
                self.assertIn(omen_effect.omen_name, result.active_omens)

    def test_stop_action_is_noop(self) -> None:
        action = self._get_action("Stop")
        item = self._make_item(rarity="Rare")
        item.prefix_ids.append("existing_mod")
        db = self._make_db([])
        result = action.apply(item, db, random.Random(0))
        self.assertEqual(item, result)
        self.assertIsNot(item, result)

    def test_essence_actions_cover_magic_and_perfect(self) -> None:
        magic_mod = _make_mod("ess_magic_mod", is_prefix=True, group="ess_magic")
        perfect_mod = _make_mod("ess_perfect_mod", is_prefix=True, group="ess_perfect")
        removable_mod = _make_mod("ess_suffix_victim", is_prefix=False)
        ess_magic = Essence(
            essence_id="ess_magic",
            name="Essence of Testing",
            guaranteed_mods={"Ring": [magic_mod.mod_id]},
        )
        ess_perfect = Essence(
            essence_id="ess_perfect",
            name="Perfect Essence of Testing",
            guaranteed_mods={"Ring": [perfect_mod.mod_id]},
        )
        db = self._make_full_db(
            [magic_mod, perfect_mod, removable_mod],
            essences={
                ess_magic.essence_id: ess_magic,
                ess_perfect.essence_id: ess_perfect,
            },
        )

        actions = {action.name: action for action in create_essence_actions(db)}
        self.assertIn(ess_magic.name, actions)
        self.assertIn(ess_perfect.name, actions)

        magic_item = self._make_item(rarity="Magic")
        magic_result = actions[ess_magic.name].apply(magic_item, db, random.Random(0))
        self.assertEqual(magic_result.rarity, "Rare")
        self.assertIn(magic_mod.mod_id, magic_result.prefix_ids)

        rare_item = self._make_item(rarity="Rare")
        rare_item.suffix_ids.append(removable_mod.mod_id)
        perfect_result = actions[ess_perfect.name].apply(rare_item, db, random.Random(0))
        self.assertIn(perfect_mod.mod_id, perfect_result.prefix_ids)
        self.assertNotIn(removable_mod.mod_id, perfect_result.suffix_ids)


if __name__ == "__main__":
    unittest.main()
