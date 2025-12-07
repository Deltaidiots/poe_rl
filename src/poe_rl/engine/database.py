from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..data.models import Item, ItemBase, Mod, Essence


@dataclass
class CraftingDatabase:
    """
    Container for mods and base items used by the crafting engine.

    Parameters
    ----------
    mods_by_id:
        Mapping of modifier identifier to :class:`Mod` objects.  These come
        from :func:`poe2_dataloader.parse_mods`.
    bases_by_id:
        Mapping of base item identifier to :class:`ItemBase` objects.
    essences_by_id:
        Mapping of essence identifier to :class:`Essence` objects.
    """

    mods_by_id: Dict[str, Mod]
    bases_by_id: Dict[str, ItemBase]
    essences_by_id: Dict[str, Essence] = field(default_factory=dict)

    def get_mod_candidates(self, item: Item, want_prefix: bool) -> List[Mod]:
        """
        Return a list of mods that can be applied to ``item`` as either a
        prefix or suffix.

        This implementation is intentionally permissive: it filters mods by
        whether they are prefixes or suffixes and whether the item's item level
        meets the mod's required level.  It does not restrict based on item
        class, domain or tags.  More restrictive filtering can be added here
        later.

        Parameters
        ----------
        item:
            The current item state.
        want_prefix:
            If ``True``, only prefix mods are returned; otherwise suffix mods.

        Returns
        -------
        List[Mod]
            Mods eligible for rolling on this item under the requested
            classification.
        """
        candidates: List[Mod] = []
        for mod in self.mods_by_id.values():
            if mod.is_prefix != want_prefix:
                continue
            if item.ilvl < mod.ilvl_required:
                continue
            candidates.append(mod)
        return candidates
