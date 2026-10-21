"""
How well the environment's stat-based damage estimate (_damage) matches the damage actually dealt,
against the max expert, next to the rough estimate the observation used before v8 (_expected_damage:
base power * accuracy * type * STAB, no stats or levels).

Plays battles with the hand-written policy (check_env.py, through actions 14-19) and, on turns where
both active Pokémon stay in and nothing else moves the HP much, compares:
  incoming: the opponent's predicted move (once revealed - the max expert always uses it) on our
    active, against the HP we lost (our move mustn't heal, drain or have recoil)
  outgoing: the attack we used on the opponent's active, against the HP it lost
Damage is in % of the defender's max HP. A knock-out cuts the actual damage short, so those turns
only count for "predicted a KO / KO happened".

    python scripts/check_damage.py --battles 300
Needs the Showdown server running (node pokemon-showdown start --no-security).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_env import policy  # noqa: E402

from showdown_gym.showdown_environment import HEAL, N_MOVES, TERA, SingleShowdownWrapper  # noqa: E402


def report(name, pairs, rough, kos):
    pred, actual = np.array(pairs).T
    ratio = actual[pred > 0] / pred[pred > 0]
    print(f"\n{name}: {len(pred)} turns without a KO")
    print(f"  actual / predicted damage: median {np.median(ratio):.2f}, middle half {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f}")
    print(f"  mean absolute error {np.mean(np.abs(actual - pred)):.1f}% of max HP (actual damage averages {np.mean(actual):.1f}%)")
    print(f"  correlation with actual damage: stat-based {np.corrcoef(pred, actual)[0, 1]:.2f}, rough {np.corrcoef(rough, actual)[0, 1]:.2f}")
    tp = sum(p and k for p, k in kos)
    print(f"  knock-outs: {sum(k for _, k in kos)} of {len(kos)} turns; predicted {sum(p for p, _ in kos)}, "
          f"right {tp}; accuracy {100 * np.mean([p == k for p, k in kos]):.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", type=int, default=300)
    args = parser.parse_args()

    env = SingleShowdownWrapper(team_type="random", opponent_type="max")
    time.sleep(3)
    showdown = env.env
    incoming, incoming_rough, incoming_ko = [], [], []
    outgoing, outgoing_rough, outgoing_ko = [], [], []
    start = time.time()

    for _ in range(args.battles):
        obs, _ = env.reset()
        done = False
        while not done:
            battle = showdown.battle1
            action = policy(obs, tera=True, macros=True)
            before = None
            active, opp = battle.active_pokemon, battle.opponent_active_pokemon
            if active is not None and opp is not None and not battle.force_switch and battle.available_moves:
                predicted, guessed = showdown._predicted_move(battle)
                target = showdown._macro(battle, action)[0] if action >= HEAL else action
                tera = target >= TERA and battle.can_tera
                move = showdown._move_slots(battle)[target % TERA] if target < N_MOVES or target >= TERA else None
                before = dict(
                    turn=battle.turn, active=active, opp=opp, hp=active.current_hp, max_hp=active.max_hp,
                    opp_hp=opp.current_hp_fraction, predicted=None if guessed else predicted, move=move, tera=tera,
                )
                if before["predicted"] is not None:
                    before["in"] = showdown._damage(predicted, opp, active)
                    before["in_rough"] = showdown._expected_damage(predicted, opp, active, use_boosts=True)
                if move is not None and move.base_power > 0:
                    before["out"] = showdown._damage(move, active, opp, tera=tera)
                    before["out_rough"] = showdown._expected_damage(move, active, opp, use_boosts=True, tera=tera)
                    before["opp_max"] = showdown._max_hp_points(opp)

            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            done = terminated or truncated
            if before is None:
                continue
            fainted = before["active"].fainted
            opp_fainted = before["opp"].fainted
            # A knock-out ends the turn before the turn counter moves on (the replacement comes first)
            if battle.turn == before["turn"] and not (fainted or opp_fainted):
                continue
            same = battle.active_pokemon is before["active"] and battle.opponent_active_pokemon is before["opp"]
            if "in" in before and (same or fainted):
                m = before["move"]
                clean = m is None or not (
                    showdown._is_heal(m) or m.drain or m.recoil or m.id in ("protect", "detect", "substitute")
                )
                if clean:
                    if fainted:
                        incoming_ko.append((before["in"] >= before["hp"], True))
                    elif not opp_fainted:
                        lost = before["hp"] - before["active"].current_hp
                        incoming.append((100 * before["in"] / before["max_hp"], 100 * lost / before["max_hp"]))
                        incoming_rough.append(before["in_rough"])
                        incoming_ko.append((before["in"] >= before["hp"], False))
            if "out" in before and (same or opp_fainted):
                if opp_fainted:
                    outgoing_ko.append((before["out"] >= before["opp_hp"] * before["opp_max"], True))
                elif not fainted:
                    lost = before["opp_hp"] - before["opp"].current_hp_fraction
                    outgoing.append((100 * before["out"] / before["opp_max"], 100 * lost))
                    outgoing_rough.append(before["out_rough"])
                    outgoing_ko.append((before["out"] >= before["opp_hp"] * before["opp_max"], False))

    print(f"{args.battles} battles in {time.time() - start:.0f}s")
    report("incoming (predicted move on our active)", incoming, incoming_rough, incoming_ko)
    report("outgoing (our attack on their active)", outgoing, outgoing_rough, outgoing_ko)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
