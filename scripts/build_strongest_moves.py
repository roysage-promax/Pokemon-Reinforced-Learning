"""
Write two tables from the gen9 random-battle sets into showdown_gym/showdown_environment.py:
  STRONGEST_MOVE_IDS: for each species, the highest base-power move in any of its movepools - the
    environment's guess at an opponent's next move before that Pokemon has attacked
  IMMUNE_ABILITY_IDS: species whose every set has the same type-immunity ability (Levitate, Volt
    Absorb...: the environment's IMMUNE_ABILITIES), so the opponent's is known before it shows
The tables live in the environment file itself (between its BEGIN / END lines) because only that
file is submitted.

    python scripts/build_strongest_moves.py [path/to/pokemon-showdown/data/random-battles/gen9/sets.json]
"""

import json
import os
import re
import sys
import textwrap

from poke_env.battle import Move

DEFAULT_SETS = os.path.expanduser("~/compsys726/pokemon-showdown/data/random-battles/gen9/sets.json")
ENV_FILE = os.path.join(os.path.dirname(__file__), "..", "showdown_gym", "showdown_environment.py")


def table(name: str, entries: dict) -> str:
    # A no-break space inside each entry keeps a key and its value on the same line
    text = " ".join(f'"{key}":\xa0"{value}",' for key, value in sorted(entries.items()))
    lines = textwrap.fill(text, 100, initial_indent="    ", subsequent_indent="    ").replace("\xa0", " ")
    return f"{name} = {{\n" + lines + "\n}\n"


def write_block(source: str, marker: str, text: str, newline: str) -> str:
    """Replace what's between the file's '# BEGIN <marker>' and '# END <marker>' lines."""
    begin, end = f"# BEGIN {marker}{newline}", f"# END {marker}{newline}"
    start, stop = source.index(begin) + len(begin), source.index(end)
    return source[:start] + text.replace("\n", newline) + source[stop:]


def immune_abilities(source: str) -> set[str]:
    """The ability ids in the environment's IMMUNE_ABILITIES dict."""
    block = source[source.index("IMMUNE_ABILITIES = {") :]
    return set(re.findall(r'^\s*"([a-z]+)":', block[: block.index("}")], re.M))


def ability_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def main():
    sets_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SETS
    with open(sets_path, encoding="utf-8") as f:
        sets = json.load(f)

    strongest = {}
    for species, entry in sets.items():
        best = None
        for move_set in entry["sets"]:
            for name in move_set["movepool"]:
                move = Move("".join(c for c in name.lower() if c.isalnum()), gen=9)
                if move.base_power > 0 and (best is None or move.base_power > best.base_power):
                    best = move
        if best is not None:
            strongest[species] = best.id

    # Keep the file's line endings (the assignment's file has Windows ones)
    with open(ENV_FILE, encoding="utf-8", newline="") as f:
        source = f.read()
    newline = "\r\n" if "\r\n" in source else "\n"

    immune = immune_abilities(source)
    certain = {}
    for species, entry in sets.items():
        abilities = {ability_id(a) for move_set in entry["sets"] for a in move_set.get("abilities", [])}
        if len(abilities) == 1 and abilities <= immune:
            certain[species] = abilities.pop()

    source = write_block(source, "IMMUNE ABILITIES", table("IMMUNE_ABILITY_IDS", certain), newline)
    source = write_block(source, "STRONGEST MOVES", table("STRONGEST_MOVE_IDS", strongest), newline)
    with open(ENV_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(source)
    print(f"{len(strongest)} species' strongest moves, {len(certain)} species' immunity abilities -> {os.path.normpath(ENV_FILE)}")


if __name__ == "__main__":
    main()
