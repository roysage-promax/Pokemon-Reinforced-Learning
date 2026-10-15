"""
Scripted probe: how far can hand-written rules on top of "use the best-damage move" go against
the max expert? No training - each variant plays gen9randombattle battles against
MaxBasePowerPlayer on a local server, to check whether an idea is worth building into the
environment before spending cloud time on it.

The max expert always uses its highest base-power move, whatever our type, and never switches
voluntarily. So once its active Pokemon has attacked, we know the move it will use every turn.

Variants (each adds to the one before, except D which builds on B):
  base  best-damage move; forced switch by type matchup (what the v2 environment can express)
  A     forced and voluntary switches use the opponent's predicted move: switch out when it is
        super-effective on us and a teammate resists it
  A2    A, and also switch to a teammate that is immune to it
  B     A2, and use a setup move (Swords Dance, Nasty Plot...) or heal while we resist it
  C     B, and guess the predicted move before it is revealed, from the random-battle sets
  D     B, and inflict sleep / burn / paralysis / poison while it can't hurt us much
  E     C, and terastallize: into a type that resists the predicted move, or for extra STAB
  F     C, and also set up (to +2) on a neutral hit while above 75% HP
  G     C, and finish a low-HP opponent that is probably faster with a priority move
  H     C + E + F + G
  E-heal, E-setup  E without its heal / setup rule: what each rule is worth
  I     E, and also heal at half HP or less when the predicted move hits us neutrally
  J     E, and lay entry hazards (Stealth Rock, Spikes...) while the predicted move is neutral or
        resisted and 3+ opposing Pokemon are left - the max expert never removes them on purpose
  K     E, and also switch out of a neutral hit to a teammate that resists it and hits at least as
        hard

    python scripts/scripted_probe.py                  # every variant, 600 battles each
    python scripts/scripted_probe.py base B -n 200    # only these
Needs the Showdown server running (node pokemon-showdown start --no-security).
"""

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter

from poke_env import AccountConfiguration, MaxBasePowerPlayer
from poke_env.battle import AbstractBattle, Move, MoveCategory, PokemonType, SideCondition, Status, Target
from poke_env.player import Player

VARIANTS = {
    "base": set(),
    "A": {"predict"},
    "A2": {"predict", "immune"},
    "B": {"predict", "immune", "setup", "heal"},
    "C": {"predict", "immune", "setup", "heal", "prior"},
    "D": {"predict", "immune", "setup", "heal", "status"},
    "E": {"predict", "immune", "setup", "heal", "prior", "tera"},
    "F": {"predict", "immune", "setup", "heal", "prior", "setup_neutral"},
    "G": {"predict", "immune", "setup", "heal", "prior", "priority"},
    "H": {"predict", "immune", "setup", "heal", "prior", "tera", "setup_neutral", "priority"},
    "E-heal": {"predict", "immune", "setup", "prior", "tera"},
    "E-setup": {"predict", "immune", "heal", "prior", "tera"},
    "I": {"predict", "immune", "setup", "heal", "prior", "tera", "heal_neutral"},
    "J": {"predict", "immune", "setup", "heal", "prior", "tera", "hazards"},
    "K": {"predict", "immune", "setup", "heal", "prior", "tera", "switch_neutral"},
}

HAZARD_MOVES = {
    "stealthrock": SideCondition.STEALTH_ROCK,
    "spikes": SideCondition.SPIKES,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
    "stickyweb": SideCondition.STICKY_WEB,
}

# Our own abilities and items are known, so these immunities are certain
IMMUNE_ABILITIES = {
    "levitate": "GROUND",
    "eartheater": "GROUND",
    "flashfire": "FIRE",
    "wellbakedbody": "FIRE",
    "waterabsorb": "WATER",
    "stormdrain": "WATER",
    "dryskin": "WATER",
    "voltabsorb": "ELECTRIC",
    "lightningrod": "ELECTRIC",
    "motordrive": "ELECTRIC",
    "sapsipper": "GRASS",
}

# Healing moves whose amount isn't in the move data's "heal" field
EXTRA_HEAL_MOVES = {"moonlight", "morningsun", "synthesis", "shoreup", "strengthsap", "rest"}

STALL_SECONDS = 60

SLEEP_MOVES = {"spore", "sleeppowder", "hypnosis", "lovelykiss", "sing", "darkvoid", "yawn"}
POWDER_MOVES = {"spore", "sleeppowder", "stunspore"}


def boost_multiplier(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def expected_damage(move, attacker, defender, use_boosts=False) -> float:
    """Same estimate as the environment: base power * accuracy * type multiplier * STAB."""
    stab = 1.5 if move.type in attacker.types else 1.0
    damage = move.base_power * move.accuracy * defender.damage_multiplier(move) * stab
    if use_boosts and move.category != MoveCategory.STATUS:
        stat = "atk" if move.category == MoveCategory.PHYSICAL else "spa"
        damage *= boost_multiplier(attacker.boosts.get(stat, 0))
    return damage


def threat(move, mon) -> float:
    """Type multiplier of the opponent's predicted move on one of our Pokemon."""
    if IMMUNE_ABILITIES.get(mon.ability) == move.type.name:
        return 0.0
    if move.type.name == "GROUND" and mon.item == "airballoon":
        return 0.0
    return mon.damage_multiplier(move)


def load_prior(sets_path: str) -> dict:
    """Species -> highest base-power move in its random-battle movepools."""
    if not os.path.exists(sets_path):
        return {}
    with open(sets_path, encoding="utf-8") as f:
        sets = json.load(f)
    prior = {}
    for species, entry in sets.items():
        moves = []
        for s in entry["sets"]:
            for name in s["movepool"]:
                move_id = "".join(c for c in name.lower() if c.isalnum())
                try:
                    moves.append(Move(move_id, gen=9))
                except Exception:
                    pass
        moves = [m for m in moves if m.base_power > 0]
        if moves:
            prior[species] = max(moves, key=lambda m: m.base_power)
    return prior


class ProbePlayer(Player):
    def __init__(self, variant: str, prior: dict, **kwargs):
        super().__init__(**kwargs)
        self.flags = VARIANTS[variant]
        self.prior = prior
        self.counts = Counter()

    def choose_move(self, battle: AbstractBattle):
        try:
            return self._choose(battle)
        except Exception:
            # A probe shouldn't die on an odd battle state; count it and move on
            self.counts["error"] += 1
            return self.choose_random_move(battle)

    # ---------- prediction ----------
    def predicted_move(self, battle):
        if "predict" not in self.flags:
            return None
        opp = battle.opponent_active_pokemon
        known = [m for m in opp.moves.values() if m.base_power > 0]
        if known:
            return max(known, key=lambda m: m.base_power)
        if "prior" in self.flags:
            return self.prior.get(opp.species)
        return None

    @staticmethod
    def offence(mon, opp) -> float:
        return max((expected_damage(m, mon, opp) for m in mon.moves.values()), default=0.0)

    def matchup(self, mon, opp, predicted) -> float:
        offence = self.offence(mon, opp) / 150.0
        if predicted is not None:
            defence = threat(predicted, mon)
        else:
            defence = max(mon.damage_multiplier(t) for t in opp.types)
        return offence - defence

    # ---------- decision ----------
    def _choose(self, battle: AbstractBattle):
        opp = battle.opponent_active_pokemon
        me = battle.active_pokemon
        predicted = self.predicted_move(battle) if opp is not None else None

        if battle.force_switch:
            if not battle.available_switches or opp is None:
                return self.choose_default_move()
            self.counts["forced_switch"] += 1
            return self.create_order(max(battle.available_switches, key=lambda m: self.matchup(m, opp, predicted)))

        if not battle.available_moves:
            return self.choose_default_move()

        use_boosts = "setup" in self.flags
        best = max(battle.available_moves, key=lambda m: expected_damage(m, me, opp, use_boosts))
        if predicted is None:
            return self.create_order(best)

        my_threat = threat(predicted, me)

        # Voluntary switch to a teammate that takes the predicted move well
        if battle.available_switches:
            switch_in = min(
                battle.available_switches,
                key=lambda m: (threat(predicted, m), -self.matchup(m, opp, predicted)),
            )
            their_threat = threat(predicted, switch_in)
            if my_threat >= 2 and their_threat <= 0.5:
                self.counts["switch_resist"] += 1
                return self.create_order(switch_in)
            if "immune" in self.flags and my_threat >= 1 and their_threat == 0:
                self.counts["switch_immune"] += 1
                return self.create_order(switch_in)
            if (
                "switch_neutral" in self.flags
                and my_threat >= 1
                and their_threat <= 0.5
                and self.offence(switch_in, opp) >= self.offence(me, opp)
            ):
                self.counts["switch_neutral"] += 1
                return self.create_order(switch_in)

        moves = {m.id: m for m in battle.available_moves}
        tera = "tera" in self.flags and self.should_tera(battle, me, predicted, my_threat, best)

        # Finish a low-HP opponent that probably moves first
        if (
            "priority" in self.flags
            and opp.current_hp_fraction <= 0.3
            and opp.base_stats["spe"] >= me.base_stats["spe"]
        ):
            quick = [m for m in moves.values() if m.priority > 0 and m.base_power > 0 and opp.damage_multiplier(m) >= 1]
            if quick:
                self.counts["priority"] += 1
                return self.create_order(max(quick, key=lambda m: expected_damage(m, me, opp, use_boosts)))

        # Heal while the opponent can only hit us for little
        heal_threat = 1 if "heal_neutral" in self.flags else 0.5
        if "heal" in self.flags and my_threat <= heal_threat and me.current_hp_fraction <= 0.5:
            for m in moves.values():
                if m.heal > 0 or m.id in EXTRA_HEAL_MOVES:
                    self.counts["heal"] += 1
                    return self.create_order(m)

        # Set up while the opponent can only hit us for little
        neutral_ok = "setup_neutral" in self.flags and my_threat <= 1 and me.current_hp_fraction >= 0.75
        if "setup" in self.flags and (my_threat <= 0.5 and me.current_hp_fraction >= 0.5 or neutral_ok):
            cap = 6 if my_threat == 0 else 2
            for m in moves.values():
                if m.category != MoveCategory.STATUS or not m.boosts or m.target != Target.SELF:
                    continue
                offensive = [s for s in ("atk", "spa") if m.boosts.get(s, 0) > 0]
                if offensive and min(me.boosts.get(s, 0) for s in offensive) < cap:
                    self.counts["setup"] += 1
                    self.counts["tera"] += tera
                    return self.create_order(m, terastallize=tera)

        # Entry hazards hurt every opponent that comes in after a knockout
        if "hazards" in self.flags and my_threat <= 1:
            opp_left = 6 - sum(m.fainted for m in battle.opponent_team.values())
            for move_id, condition in HAZARD_MOVES.items():
                if move_id in moves and opp_left >= 3 and condition not in battle.opponent_side_conditions:
                    self.counts["hazard"] += 1
                    return self.create_order(moves[move_id])

        # Cripple the opponent with a status condition
        if "status" in self.flags and my_threat <= 1 and opp.status is None:
            order = self.status_move(battle, moves, me, opp, predicted)
            if order is not None:
                return order

        self.counts["tera"] += tera
        return self.create_order(best, terastallize=tera)

    def should_tera(self, battle, me, predicted, my_threat, best) -> bool:
        """Tera when it turns the predicted move into a resisted one, or adds STAB to our best move."""
        # poke_env only learns our tera type after we terastallize; the server's request has it
        tera_name = (me._last_request or {}).get("teraType")
        if not battle.can_tera or not tera_name or me.is_terastallized:
            return False
        tera_type = PokemonType.from_name(tera_name)
        tera_threat = predicted.type.damage_multiplier(tera_type, None, type_chart=me._data.type_chart)
        if IMMUNE_ABILITIES.get(me.ability) == predicted.type.name:
            tera_threat = 0.0
        if my_threat >= 1 and tera_threat <= 0.5:
            return True
        return best.type == tera_type and best.base_power > 0 and tera_threat <= my_threat

    def status_move(self, battle, moves, me, opp, predicted):
        opp_types = {t.name for t in opp.types}
        someone_asleep = any(m.status == Status.SLP for m in battle.opponent_team.values())
        for move_id, m in moves.items():
            if move_id in POWDER_MOVES and "GRASS" in opp_types:
                continue
            if move_id in SLEEP_MOVES and not someone_asleep:
                self.counts["status_sleep"] += 1
                return self.create_order(m)
        for move_id, m in moves.items():
            if move_id == "willowisp" and "FIRE" not in opp_types and predicted.category == MoveCategory.PHYSICAL:
                self.counts["status_burn"] += 1
                return self.create_order(m)
        for move_id, m in moves.items():
            if (
                move_id in {"thunderwave", "glare", "stunspore", "nuzzle"}
                and not opp_types & {"ELECTRIC", "GROUND"}
                and opp.base_stats["spe"] >= me.base_stats["spe"]
            ):
                self.counts["status_para"] += 1
                return self.create_order(m)
        for move_id, m in moves.items():
            if move_id == "toxic" and not opp_types & {"POISON", "STEEL"}:
                self.counts["status_toxic"] += 1
                return self.create_order(m)
        return None


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


async def run_variant(variant, n_battles, chunk, concurrency, prior, stamp):
    player = ProbePlayer(
        variant,
        prior,
        account_configuration=AccountConfiguration(f"pr{variant}{stamp}", None),
        battle_format="gen9randombattle",
        max_concurrent_battles=concurrency,
    )
    opponent = MaxBasePowerPlayer(
        account_configuration=AccountConfiguration(f"mx{variant}{stamp}", None),
        battle_format="gen9randombattle",
        max_concurrent_battles=concurrency,
    )
    wins = played = turns = 0
    stuck = set()
    while played < n_battles:
        n = min(chunk, n_battles - played)
        task = asyncio.ensure_future(player.battle_against(opponent, n_battles=n))
        # poke_env can fail to parse a request (e.g. a Dancer copying a move), and that side then
        # never chooses: forfeit battles that haven't moved on a turn in STALL_SECONDS and leave
        # them out of the results
        last_turn = {}
        while not task.done():
            await asyncio.wait({task}, timeout=STALL_SECONDS)
            for tag, battle in list(player._battles.items()):
                if battle.finished or tag in stuck:
                    continue
                if last_turn.get(tag) == battle.turn:
                    stuck.add(tag)
                    player.counts["stuck"] += 1
                    await player.ps_client.send_message("/forfeit", tag)
                last_turn[tag] = battle.turn
        # poke_env keeps every battle (~1 MB each); record and drop the finished ones
        for p in (player, opponent):
            for tag, battle in list(p._battles.items()):
                if battle.finished:
                    if p is player and tag not in stuck:
                        wins += bool(battle.won)
                        turns += battle.turn
                        played += 1
                    del p._battles[tag]
    return variant, wins, played, turns / max(played, 1), player.counts


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"any of {', '.join(VARIANTS)} (default: all)")
    parser.add_argument("-n", "--battles", type=int, default=600)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--sets",
        default=os.path.expanduser("~/compsys726/pokemon-showdown/data/random-battles/gen9/sets.json"),
        help="random-battle sets for variant C",
    )
    args = parser.parse_args()
    args.variants = args.variants or list(VARIANTS)
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {', '.join(sorted(unknown))}")

    needs_prior = any("prior" in VARIANTS[v] for v in args.variants)
    prior = load_prior(args.sets) if needs_prior else {}
    if needs_prior and not prior:
        print(f"warning: no sets at {args.sets}; the move guess (variant C and later) is off")

    stamp = time.strftime("%H%M%S")
    start = time.time()
    results = await asyncio.gather(
        *(run_variant(v, args.battles, args.chunk, args.concurrency, prior, stamp) for v in args.variants)
    )

    print(f"\n{len(args.variants)} variants x {args.battles} battles vs max in {time.time() - start:.0f}s\n")
    print(f"{'variant':8s} {'wins':>9s} {'win %':>6s} {'95% CI':>14s} {'turns':>6s}  decisions per battle")
    for variant, wins, played, avg_turns, counts in results:
        lo, hi = wilson(wins, played)
        per = ", ".join(f"{k} {v / played:.2f}" for k, v in sorted(counts.items()))
        print(f"{variant:8s} {wins:4d}/{played:<4d} {100 * wins / played:6.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}] {avg_turns:6.1f}  {per}")


if __name__ == "__main__":
    asyncio.run(main())
