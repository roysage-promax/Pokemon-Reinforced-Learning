"""
Write the strongest-move table into showdown_gym/showdown_environment.py: for each species in gen9
random battles, the highest base-power move in any of its movepools. The environment uses it to
guess an opponent's next move before that Pokemon has attacked. The table lives in the environment
file itself (between the BEGIN / END STRONGEST MOVES lines) because only that file is submitted.

    python scripts/build_strongest_moves.py [path/to/pokemon-showdown/data/random-battles/gen9/sets.json]
"""

import json
import os
import sys
import textwrap

from poke_env.battle import Move

DEFAULT_SETS = os.path.expanduser("~/compsys726/pokemon-showdown/data/random-battles/gen9/sets.json")
ENV_FILE = os.path.join(os.path.dirname(__file__), "..", "showdown_gym", "showdown_environment.py")
BEGIN, END = "# BEGIN STRONGEST MOVES\n", "# END STRONGEST MOVES\n"


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

    # A no-break space inside each entry keeps a species and its move on the same line
    entries = " ".join(f'"{species}":\xa0"{move}",' for species, move in sorted(strongest.items()))
    lines = textwrap.fill(entries, 100, initial_indent="    ", subsequent_indent="    ").replace("\xa0", " ")
    table = "STRONGEST_MOVE_IDS = {\n" + lines + "\n}\n"

    # Keep the file's line endings (the assignment's file has Windows ones)
    with open(ENV_FILE, encoding="utf-8", newline="") as f:
        source = f.read()
    newline = "\r\n" if "\r\n" in source else "\n"
    begin, end = BEGIN.replace("\n", newline), END.replace("\n", newline)
    start, stop = source.index(begin) + len(begin), source.index(end)
    with open(ENV_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(source[:start] + table.replace("\n", newline) + source[stop:])
    print(f"{len(strongest)} species -> {os.path.normpath(ENV_FILE)}")


if __name__ == "__main__":
    main()
