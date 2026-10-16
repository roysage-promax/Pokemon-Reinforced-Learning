"""
Plays battles against the max expert with the hand-written policy in check_env.py (probe variant E
with variant I's heal, ~95.7%) or, given a checkpoint, a trained network, and writes down what
happened - to find out why battles are lost, and which losses better play could have won:

  battles.csv       a row per battle: won, turns, Pokemon left on each side, critical hits, misses
                    and "can't move" (paralysis, sleep, flinch...) each way, and our plays
  losses/NNNN.txt   each lost battle turn by turn: both teams as far as they were seen, then per
                    turn what we chose (with the hand policy's choice when a network plays), the
                    observation's view (HP, the predicted move's threat), and the battle's messages
  summary.txt       win rate, and every counter's average in wins and in losses - luck (crits,
                    misses, can't move) that piles up in losses is a loss better play couldn't avoid

    python scripts/battle_log.py --battles 3000 --out diagnostics/hand
    python scripts/battle_log.py --checkpoint <DQN_checkpoint.pth> --battles 3000 --out diagnostics/agent
Needs the Showdown server running (node pokemon-showdown start --no-security).
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_policy import kind, load_network, same  # noqa: E402
from check_env import CONTEXT, FORCED, PREDICTED, TERA_ACTION, policy, resolve, wilson  # noqa: E402

from showdown_gym.showdown_environment import N_MOVES, SingleShowdownWrapper  # noqa: E402

# Protocol messages that say nothing about the fight itself
QUIET = {"", "t:", "j", "l", "n", "c", "raw", "html", "uhtml", "player", "teamsize", "gen", "tier", "rule",
         "gametype", "rated", "title", "init", "start", "request", "upkeep", "poke", "teampreview",
         "clearpoke", "inactive", "inactiveoff", "timer", "turn", "badge", "-hint", "bigerror", "message"}
LUCK = ["crit_on_us", "crit_by_us", "miss_by_us", "miss_by_them", "cant_us", "cant_them"]
PLAYS = ["attack", "attack+tera", "switch", "forced switch", "heal", "setup"]


def luck(events: list[list[str]], us: str) -> Counter:
    """Crits, misses and "can't move" for each side; Pokemon idents start with the side ("p1a: ...")."""
    counts = Counter()
    for event in events:
        if len(event) < 3:
            continue
        ours = event[2].startswith(us)
        if event[1] == "-crit":
            counts["crit_on_us" if ours else "crit_by_us"] += 1
        elif event[1] == "-miss":
            counts["miss_by_us" if ours else "miss_by_them"] += 1
        elif event[1] == "cant":
            counts["cant_us" if ours else "cant_them"] += 1
    return counts


def describe(mon) -> str:
    moves = "/".join(mon.moves) or "-"
    state = "fainted" if mon.fainted else f"{100 * mon.current_hp_fraction:.0f}%"
    tera = f", tera {mon.tera_type.name.lower()}" if mon.is_terastallized and mon.tera_type else ""
    return f"{mon.species} ({state}; item {mon.item or '?'}, ability {mon.ability or '?'}{tera}; moves {moves})"


def name(battle, obs: np.ndarray, action: int) -> str:
    """The play and what it picks, e.g. "switch to garchomp", "attack+tera earthquake" (poke_env's slot order)."""
    action = resolve(obs, action)
    forced = "forced: " if obs[FORCED] else ""
    if N_MOVES <= action < TERA_ACTION:
        team = list(battle.team.values())
        j = action - N_MOVES
        return f"{forced}switch to {team[j].species if j < len(team) else '?'}"
    moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
    slot = action % TERA_ACTION if action >= TERA_ACTION else action
    return f"{forced}{kind(obs, action)} {moves[slot].id if slot < len(moves) else '?'}"


def view(obs: np.ndarray) -> str:
    predicted = "revealed" if obs[PREDICTED] else "guessed" if obs[PREDICTED + 1] else "unknown"
    return (f"our hp {obs[CONTEXT]:.2f}, theirs {obs[CONTEXT + 1]:.2f}, "
            f"predicted move ({predicted}) hits us x{4 * obs[PREDICTED + 3]:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="", help="play this network instead of the hand-written policy")
    parser.add_argument("--battles", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    net = load_network(args.checkpoint) if args.checkpoint else None
    os.makedirs(f"{args.out}/losses", exist_ok=True)
    env = SingleShowdownWrapper(team_type="random", opponent_type="max", evaluation=True)
    showdown = env.env
    time.sleep(3)

    rows = []
    start = time.time()
    for i in range(args.battles):
        obs, _ = env.reset()
        plays = Counter()
        choices = {}  # turn -> what we chose that turn and why
        done = False
        while not done:
            hand = policy(obs, tera=True)
            action = hand
            if net is not None:
                with torch.no_grad():
                    action = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
            if showdown.battle1 is not None and showdown.agent1_to_move:
                battle = showdown.battle1
                plays["forced switch" if obs[FORCED] else kind(obs, action)] += 1
                note = name(battle, obs, action)
                if not same(obs, action, hand):
                    note += f" (hand policy: {name(battle, obs, hand)})"
                choices.setdefault(battle.turn, []).append(f"{note} - {view(obs)}")
            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            done = terminated or truncated

        battle = showdown.battle1
        us = battle.player_role
        events = [event for turn in sorted(battle.observations) for event in battle.observations[turn].events]
        counts = luck(events, us)
        row = {
            "battle": i,
            "won": bool(battle.won),
            "turns": battle.turn,
            "ours_left": sum(not mon.fainted for mon in battle.team.values()),
            "theirs_left": 6 - sum(mon.fainted for mon in battle.opponent_team.values()),
            **{k: counts[k] for k in LUCK},
            **{p: plays[p] for p in PLAYS},
        }
        rows.append(row)

        if not battle.won:
            lines = [f"battle {i} ({battle.battle_tag}): lost in {battle.turn} turns, "
                     f"{row['theirs_left']} of theirs left; we are {us}", "",
                     "ours:", *(f"  {describe(mon)}" for mon in battle.team.values()),
                     "theirs (as seen):", *(f"  {describe(mon)}" for mon in battle.opponent_team.values()), ""]
            for turn in sorted(battle.observations):
                lines.append(f"-- turn {turn}")
                lines += [f"  WE CHOSE: {choice}" for choice in choices.get(turn, [])]
                lines += ["  " + "|".join(e[1:]) for e in battle.observations[turn].events if len(e) > 1 and e[1] not in QUIET]
            with open(f"{args.out}/losses/{i:04d}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    with open(f"{args.out}/battles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    lo, hi = wilson(len(wins), n)
    who = f"network {args.checkpoint}" if net is not None else "hand-written policy (variant E + I's heal)"
    summary = [f"{who}: {n} battles in {time.time() - start:.0f}s, won {len(wins)} = "
               f"{100 * len(wins) / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]", "",
               f"{'per battle':16s} {'wins':>7s} {'losses':>7s}"]

    def mean(rs, k):
        return sum(r[k] for r in rs) / len(rs) if rs else float("nan")

    for k in ["turns", "ours_left", "theirs_left", *LUCK, *PLAYS]:
        summary.append(f"{k:16s} {mean(wins, k):7.2f} {mean(losses, k):7.2f}")
    summary += ["", "losses by how many of theirs were left: " + ", ".join(
        f"{k}: {v}" for k, v in sorted(Counter(r["theirs_left"] for r in losses).items()))]
    with open(f"{args.out}/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
