
import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path so we can import poe_rl
sys.path.append(str(Path(__file__).parent.parent.parent))

from poe_rl.data.loader import load_item_bases, load_mods, load_essences
from poe_rl.data.models import Item
from poe_rl.engine.actions import ACTIONS, CraftingAction, create_essence_actions
from poe_rl.engine.core import CraftingEngine
from poe_rl.engine.database import CraftingDatabase
from poe_rl.engine.price import PriceProvider, StaticPriceProvider

# Constants
DATA_PATH = Path(__file__).parent.parent / "data" / "static" / "poec_data.json"

st.set_page_config(page_title="PoE2 Crafting Simulator", layout="wide")

st.title("Path of Exile 2 Crafting Simulator")

@st.cache_resource
def get_engine():
    try:
        bases = load_item_bases(DATA_PATH)
        mods = load_mods(str(DATA_PATH))
        essences = load_essences(str(DATA_PATH))
        
        bases_by_id = {b.base_id: b for b in bases}
        mods_by_id = {m.mod_id: m for m in mods}
        essences_by_id = {e.essence_id: e for e in essences}
        
        db = CraftingDatabase(mods_by_id=mods_by_id, bases_by_id=bases_by_id, essences_by_id=essences_by_id)
        price_provider = StaticPriceProvider(prices={})
        return CraftingEngine(db=db, price_provider=price_provider)
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")
        return None

engine = get_engine()

if not engine:
    st.stop()

item_bases = list(engine.db.bases_by_id.values())

# Sidebar for Item Selection
st.sidebar.header("Item Selection")

if not item_bases:
    st.warning("No item bases loaded.")
else:
    # Filter by class
    item_classes = sorted(list(set(b.item_class for b in item_bases)))
    selected_class = st.sidebar.selectbox("Item Class", item_classes)


    # Filter items by class
    filtered_items = [b for b in item_bases if b.item_class == selected_class]
    item_names = sorted([b.name for b in filtered_items])
    selected_item_name = st.sidebar.selectbox("Base Item", item_names)

    # Get selected item base
    selected_base = next((b for b in filtered_items if b.name == selected_item_name), None)

    if selected_base:
        st.sidebar.write(f"**Base Level:** {selected_base.drop_level}")
        
        # Item Level Selector
        selected_ilvl = st.sidebar.number_input(
            "Item Level", 
            min_value=1, 
            max_value=100, 
            value=max(1, selected_base.drop_level),
            step=1
        )

        st.sidebar.write(f"**Implicits:**")
        for imp in selected_base.implicits:
            st.sidebar.write(f"- {imp}")

        # Initialize Item State
        # Reset item if base or ilvl changes significantly (though usually we want to keep crafting state if just tweaking ilvl? 
        # No, changing base resets. Changing ilvl should probably reset too or at least re-validate mods, 
        # but for simplicity let's reset if base changes. 
        # If user changes ilvl, we might want to update the current item's ilvl.
        
        if "current_item" not in st.session_state or st.session_state.get("base_name") != selected_item_name:
            st.session_state.current_item = Item(
                base=selected_base,
                ilvl=selected_ilvl
            )
            st.session_state.base_name = selected_item_name
        
        # Update ilvl if it changed in the UI but item exists
        if st.session_state.current_item.ilvl != selected_ilvl:
             st.session_state.current_item.ilvl = selected_ilvl

        item = st.session_state.current_item

        # Helper to get item image path
        def get_item_image_path(item_base):
            # Map item class to folder
            # This is a heuristic based on folder structure
            # game_poe2/amulets, rings, belts, etc.
            
            base_path = Path(__file__).parent / "static" / "items"
            
            # Normalize class name
            cls = item_base.item_class.lower()
            
            folder = None
            if "amulet" in cls: folder = "amulets"
            elif "ring" in cls: folder = "rings"
            elif "belt" in cls: folder = "belts"
            elif "quiver" in cls: folder = "quivers"
            elif "flask" in cls: folder = "flasks"
            elif "jewel" in cls: folder = "jewels"
            elif "helmet" in cls: folder = "armours/helmets"
            elif "gloves" in cls: folder = "armours/gloves"
            elif "boots" in cls: folder = "armours/boots"
            elif "body" in cls or "armour" in cls: folder = "armours/bodyarmours"
            elif "shield" in cls: folder = "armours/shields"
            elif "focus" in cls: folder = "armours/focii"
            elif "bow" in cls: folder = "weapons/twohandweapons/bows"
            elif "wand" in cls: folder = "weapons/onehandweapons/wands"
            elif "staff" in cls: folder = "weapons/twohandweapons/staves"
            elif "claw" in cls: folder = "weapons/onehandweapons/claws"
            elif "dagger" in cls: folder = "weapons/onehandweapons/daggers"
            elif "sword" in cls:
                if "two" in cls: folder = "weapons/twohandweapons/twohandswords"
                else: folder = "weapons/onehandweapons/onehandswords"
            elif "axe" in cls:
                if "two" in cls: folder = "weapons/twohandweapons/twohandaxes"
                else: folder = "weapons/onehandweapons/onehandaxes"
            elif "mace" in cls:
                if "two" in cls: folder = "weapons/twohandweapons/twohandmaces"
                else: folder = "weapons/onehandweapons/onehandmaces"
            elif "spear" in cls: folder = "weapons/onehandweapons/onehandspears"
            elif "flail" in cls: folder = "weapons/onehandweapons/flails"
            elif "crossbow" in cls: folder = "weapons/twohandweapons/crossbows"
            
            if folder:
                # Try to find image by name
                # Names in files seem to be like "fouramulet1.webp" or "fourring1.webp"
                # This is tricky because we don't have a direct map from Base Name to Filename.
                # However, poe2.js might have had some clues, or we can try to fuzzy match or just use a default for the class.
                # Given the file list "fouramulet1.webp", it seems they are generic or indexed.
                # Let's try to find a generic one or just pick one.
                
                # Better approach: Check if we can map specific bases.
                # For now, let's just return a random one from the folder or the first one to show SOMETHING.
                # Or if the user provided specific images, we could use them.
                
                full_folder = base_path / folder
                if full_folder.exists():
                    files = list(full_folder.glob("*.webp")) + list(full_folder.glob("*.png"))
                    if files:
                        # Hash the base name to pick a consistent image
                        idx = hash(item_base.name) % len(files)
                        return str(files[idx])
            
            return None

        # Main Area
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Current Item")
            
            # Center the image
            img_path = get_item_image_path(item.base)
            if img_path:
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:
                    st.image(img_path, width=150)
            
            # Item Card Style
            st.markdown("""
            <style>
            .item-card {
                background-color: #0e0e0e;
                border: 1px solid #555;
                padding: 15px;
                color: #fff;
                font-family: 'Fontin', 'Segoe UI', sans-serif;
                box-shadow: 0 0 15px rgba(0,0,0,0.8);
                margin-top: 10px;
                max-width: 400px;
                margin-left: auto;
                margin-right: auto;
            }
            .item-header {
                text-align: center;
                font-size: 1.4em;
                padding-bottom: 8px;
                border-bottom: 1px solid #555;
                margin-bottom: 10px;
            }
            .rarity-normal { color: #c8c8c8; }
            .rarity-magic { color: #8888ff; }
            .rarity-rare { color: #ffff77; }
            .rarity-unique { color: #af6025; }
            .item-stats {
                font-size: 0.95em;
                color: #7f7f7f;
                line-height: 1.4;
                text-align: center;
            }
            .item-separator {
                margin-top: 10px; 
                border-top: 1px solid #333; 
                padding-top: 5px;
            }
            .mod-text-magic {
                color: #8888ff;
                font-size: 1.0em;
            }
            .mod-text-rare {
                color: #ffff77;
                font-size: 1.0em;
            }
            .mod-group {
                color: #555;
                font-size: 0.8em;
                margin-left: 5px;
            }
            .omen-text {
                color: #ffaa00;
            }
            /* Reduce button text size in the crafting area */
            div[data-testid="stHorizontalBlock"] button p {
                font-size: 12px !important;
            }
            div[data-testid="stHorizontalBlock"] button {
                padding-top: 0px !important;
                padding-bottom: 0px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            rarity_class = f"rarity-{item.rarity.lower()}"
            mod_class = "mod-text-rare" if item.rarity == "Rare" else "mod-text-magic"
            
            # Construct HTML for Item Card (No indentation to avoid Markdown code block issues)
            html_content = f"""
<div class="item-card">
<div class="item-header {rarity_class}">{item.base.name}</div>
<div class="item-stats">
Item Level: {item.ilvl}<br>
{f"Quality: +{item.quality}%<br>" if item.quality > 0 else ""}
{f"Corrupted<br>" if item.corrupted else ""}
</div>
"""
            
            if item.active_omens:
                 omens_str = ", ".join(sorted(item.active_omens))
                 html_content += f"<div class='item-separator omen-text'>Active Omens: {omens_str}</div>"

            # Implicits
            if item.base.implicits:
                html_content += "<div class='item-separator'>"
                for mod in item.base.implicits:
                    html_content += f"<div class='{mod_class}'>{mod}</div>"
                html_content += "</div>"
            
            # Explicit Mods
            if item.prefix_ids or item.suffix_ids:
                html_content += "<div class='item-separator'>"
                
                for mod_id in item.prefix_ids:
                    mod = engine.db.mods_by_id.get(mod_id)
                    name = mod.name if mod else mod_id
                    html_content += f"<div class='{mod_class}'>{name} <span class='mod-group'>(P)</span></div>"
                
                for mod_id in item.suffix_ids:
                    mod = engine.db.mods_by_id.get(mod_id)
                    name = mod.name if mod else mod_id
                    html_content += f"<div class='{mod_class}'>{name} <span class='mod-group'>(S)</span></div>"
                    
                html_content += "</div>"

            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)

        with col2:
            st.subheader("Crafting Actions")
            
            # Create a map of action names to action objects
            essence_actions = create_essence_actions(engine.db)
            all_actions = ACTIONS + essence_actions
            
            # Categorize actions
            categories = {
                "Basic Currency": [],
                "Essences": [],
                "Omens": [],
                "Other": []
            }
            
            for action in all_actions:
                if "Essence" in action.name:
                    categories["Essences"].append(action)
                elif "Omen" in action.name:
                    categories["Omens"].append(action)
                elif action.name in ["Orb of Transmutation", "Orb of Alchemy", "Chaos Orb", "Exalted Orb", "Regal Orb", "Orb of Annulment", "Orb of Augmentation", "Vaal Orb", "Artificer's Orb", "Divine Orb", "Fracturing Orb", "Orb of Chance", "Scouring Orb", "Orb of Scouring"]:
                    categories["Basic Currency"].append(action)
                else:
                    categories["Other"].append(action)

            # Helper to get image path
            def get_image_path(action_name):
                base_path = Path(__file__).parent / "static"
                
                # Basic Currency
                currency_map = {
                    "Orb of Transmutation": "currency/method_transmute.png",
                    "Orb of Alchemy": "currency/method_poe2_alchemy.png",
                    "Chaos Orb": "currency/method_chaos.png",
                    "Exalted Orb": "currency/method_exalted.png",
                    "Regal Orb": "currency/method_regal.png",
                    "Orb of Annulment": "currency/method_annul.png",
                    "Orb of Augmentation": "currency/method_augmentation.png",
                    "Vaal Orb": "currency/method_vaal.png",
                    "Artificer's Orb": "currency/method_artificer.png",
                    "Divine Orb": "currency/method_divine.png",
                    "Fracturing Orb": "currency/method_fracturing.png",
                    "Orb of Scouring": "currency/method_scouring.png" # If exists
                }
                
                if action_name in currency_map:
                    p = base_path / currency_map[action_name]
                    if p.exists(): return str(p)
                
                # Essences
                if "Essence of" in action_name:
                    # Format: [tier]_[name].png
                    # Tier: Lesser, Greater, Perfect, or nothing (Normal)
                    parts = action_name.split(" ")
                    
                    # Helper to extract name after "Essence of"
                    try:
                        idx = parts.index("Essence")
                        # Name is everything after "of" (which is at idx+1)
                        if idx + 2 < len(parts):
                            raw_name_parts = parts[idx+2:]
                            # Remove "the" if present (e.g. Essence of the Body -> Body)
                            if raw_name_parts[0].lower() == "the":
                                raw_name_parts = raw_name_parts[1:]
                            
                            raw_name = "_".join(raw_name_parts).lower()
                            
                            p = None
                            if parts[0] in ["Lesser", "Greater", "Perfect"]:
                                tier = parts[0].lower()
                                p = base_path / f"essences/{tier}_{raw_name}.png"
                            else:
                                p = base_path / f"essences/essence_{raw_name}.png"
                            
                            if p and p.exists():
                                return str(p)
                    except ValueError:
                        pass

                # Omens
                if "Omen of" in action_name:
                    # Format: name_with_underscores.png
                    # "Omen of Dextral Annulment" -> "dextral_annulment.png"
                    raw_name = action_name.replace("Omen of ", "").replace(" ", "_").lower() + ".png"
                    p = base_path / f"omens/{raw_name}"
                    if p.exists(): return str(p)
                
                return None

            # Helper to check if enabled
            def check_enabled(action, item):
                for req in action.requirements:
                    if not req.check(item):
                        return False, req.error_message()
                return True, ""

            # Tabs
            tabs = st.tabs(["Basic Currency", "Essences", "Omens", "Other"])
            
            def render_action_grid(actions_list):
                cols = st.columns(6)
                for i, action in enumerate(actions_list):
                    col = cols[i % 6]
                    with col:
                        img_path = get_image_path(action.name)
                        if img_path:
                            st.image(img_path, width=40)
                        
                        is_enabled, reason = check_enabled(action, item)
                        
                        if st.button(action.name, key=action.name, disabled=not is_enabled, help=reason if not is_enabled else None, use_container_width=True):
                            # Apply Action Logic (Same as before)
                            try:
                                old_item = item
                                new_item, cost = engine.apply(item, action)
                                st.session_state.current_item = new_item
                                
                                # Calculate changes
                                added_prefixes = set(new_item.prefix_ids) - set(old_item.prefix_ids)
                                removed_prefixes = set(old_item.prefix_ids) - set(new_item.prefix_ids)
                                added_suffixes = set(new_item.suffix_ids) - set(old_item.suffix_ids)
                                removed_suffixes = set(old_item.suffix_ids) - set(new_item.suffix_ids)
                                
                                changes = []
                                for mid in added_prefixes:
                                    mod = engine.db.mods_by_id.get(mid)
                                    name = mod.name if mod else mid
                                    changes.append(f"Added Prefix: {name}")
                                for mid in added_suffixes:
                                    mod = engine.db.mods_by_id.get(mid)
                                    name = mod.name if mod else mid
                                    changes.append(f"Added Suffix: {name}")
                                for mid in removed_prefixes:
                                    mod = engine.db.mods_by_id.get(mid)
                                    name = mod.name if mod else mid
                                    changes.append(f"Removed Prefix: {name}")
                                for mid in removed_suffixes:
                                    mod = engine.db.mods_by_id.get(mid)
                                    name = mod.name if mod else mid
                                    changes.append(f"Removed Suffix: {name}")
                                    
                                if not changes and new_item.rarity != old_item.rarity:
                                    changes.append(f"Rarity changed to {new_item.rarity}")
                                
                                if not changes and new_item.quality != old_item.quality:
                                    changes.append(f"Quality changed to {new_item.quality}%")
                                    
                                log_entry = f"**{action.name}** (Cost: {cost:.2f}): " + ", ".join(changes) if changes else f"**{action.name}** (Cost: {cost:.2f}): No visible changes"
                                
                                if "action_log" not in st.session_state:
                                    st.session_state.action_log = []
                                st.session_state.action_log.insert(0, log_entry)
                                
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to apply action: {e}")

            with tabs[0]:
                render_action_grid(categories["Basic Currency"])
            with tabs[1]:
                render_action_grid(categories["Essences"])
            with tabs[2]:
                render_action_grid(categories["Omens"])
            with tabs[3]:
                render_action_grid(categories["Other"])

            st.write("---")
            st.subheader("Action Log")
            if "action_log" in st.session_state:
                for entry in st.session_state.action_log:
                    st.markdown(entry)
            else:
                st.write("No actions taken yet.")

    else:
        st.info("Please select an item base from the sidebar.")
