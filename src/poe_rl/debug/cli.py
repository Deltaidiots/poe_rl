#!/usr/bin/env python3
"""
CLI for exploring PoE2 crafting data.

Usage:
    uv run python -m poe_rl.debug.cli summary
    uv run python -m poe_rl.debug.cli essence "grounding"
    uv run python -m poe_rl.debug.cli mod "life" --class Ring
    uv run python -m poe_rl.debug.cli mod-family 5039
"""

import argparse
import sys
from pathlib import Path

# Default data path
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "static" / "poec_data.json"


def main():
    parser = argparse.ArgumentParser(
        description="Explore PoE2 crafting data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m poe_rl.debug.cli summary
    python -m poe_rl.debug.cli essence "life"
    python -m poe_rl.debug.cli essence-info 3207 --class Ring
    python -m poe_rl.debug.cli mod "resistance" --class Ring
    python -m poe_rl.debug.cli mod-family 5039
    python -m poe_rl.debug.cli base "pearl"
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to poec_data.json",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Summary command
    subparsers.add_parser("summary", help="Show data summary")
    
    # Essence search
    ess_parser = subparsers.add_parser("essence", help="Search for essences by name")
    ess_parser.add_argument("query", type=str, help="Search query")
    
    # Essence info
    ess_info_parser = subparsers.add_parser("essence-info", help="Show detailed essence info")
    ess_info_parser.add_argument("essence_id", type=str, help="Essence ID")
    ess_info_parser.add_argument("--class", dest="item_class", default="Ring", help="Item class")
    
    # Mod search
    mod_parser = subparsers.add_parser("mod", help="Search for mods")
    mod_parser.add_argument("query", type=str, help="Search query")
    mod_parser.add_argument("--class", dest="item_class", default=None, help="Filter by item class")
    
    # Mod family
    family_parser = subparsers.add_parser("mod-family", help="Show all variants of a mod")
    family_parser.add_argument("base_mod_id", type=str, help="Base mod ID (e.g., 5039)")
    
    # Base search
    base_parser = subparsers.add_parser("base", help="Search for item bases")
    base_parser.add_argument("query", type=str, help="Search query")
    
    # List essences by tier
    list_ess_parser = subparsers.add_parser("list-essences", help="List essences by tier")
    list_ess_parser.add_argument(
        "--tier", 
        choices=["lesser", "greater", "perfect", "all"],
        default="all",
        help="Filter by tier",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Import and create explorer
    from .explorer import create_explorer_from_path
    
    explorer = create_explorer_from_path(args.data)
    
    if args.command == "summary":
        print("\n" + "=" * 60)
        print("PoE2 Crafting Data Summary")
        print("=" * 60)
        for key, value in explorer.summary().items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()
    
    elif args.command == "essence":
        matches = explorer.find_essences(args.query)
        if not matches:
            print(f"No essences found matching '{args.query}'")
            return
        
        print(f"\nFound {len(matches)} essences matching '{args.query}':\n")
        for m in matches:
            print(f"  [{m.tier:>7}] {m.essence.name} (ID: {m.essence.essence_id})")
        print()
    
    elif args.command == "essence-info":
        explorer.print_essence_info(args.essence_id, args.item_class)
    
    elif args.command == "mod":
        matches = explorer.find_mods(args.query, args.item_class)
        if not matches:
            print(f"No mods found matching '{args.query}'")
            return
        
        print(f"\nFound {len(matches)} mods matching '{args.query}'")
        if args.item_class:
            print(f"Filtered to: {args.item_class}")
        print()
        
        # Group by name
        by_name = {}
        for m in matches:
            key = m.mod.name
            if key not in by_name:
                by_name[key] = []
            by_name[key].append(m.mod)
        
        for name, mods in sorted(by_name.items())[:10]:
            print(f"  {name}:")
            for mod in sorted(mods, key=lambda x: x.tier)[:3]:
                print(f"    T{mod.tier}: {mod.mod_id} (ilvl {mod.ilvl_required})")
        
        if len(by_name) > 10:
            print(f"\n  ... and {len(by_name) - 10} more mod types")
        print()
    
    elif args.command == "mod-family":
        explorer.print_mod_family(args.base_mod_id)
    
    elif args.command == "base":
        query_lower = args.query.lower()
        matches = [
            b for b in explorer._bases_by_id.values()
            if query_lower in b.name.lower()
        ]
        
        if not matches:
            print(f"No item bases found matching '{args.query}'")
            return
        
        print(f"\nFound {len(matches)} item bases matching '{args.query}':\n")
        for b in sorted(matches, key=lambda x: x.name)[:20]:
            print(f"  {b.name} (ID: {b.base_id}, class: {b.item_class}, drop lvl: {b.drop_level})")
        
        if len(matches) > 20:
            print(f"\n  ... and {len(matches) - 20} more bases")
        print()
    
    elif args.command == "list-essences":
        essences = explorer.list_essences_by_tier(args.tier)
        print(f"\n{args.tier.title() if args.tier != 'all' else 'All'} Essences ({len(essences)}):\n")
        for e in sorted(essences, key=lambda x: x.name):
            tier = explorer.get_essence_tier(e.essence_id)
            print(f"  [{tier:>7}] {e.name} (ID: {e.essence_id})")
        print()


if __name__ == "__main__":
    main()
