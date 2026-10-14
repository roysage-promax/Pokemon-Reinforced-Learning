import json
import os
import time
from typing import Any, Dict
import numpy as np

from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle, Move, MoveCategory, PokemonType, Target
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player
from .base_environment import BaseShowdownEnv

N_MOVES = 4  # actions 0-3: use move slot 0-3
N_SWITCHES = 6  # actions 4-9: switch to team slot 0-5
N_TERA = 4  # actions 10-13: terastallize and use move slot 0-3
TERA = N_MOVES + N_SWITCHES

# Penalty for an action that can't be taken this turn; poke_env then sends the server's default choice.
# 0.5 cut illegal actions to ~0.2 a battle in v3 (0.1 left ~2). SHOWDOWN_ILLEGAL_PENALTY overrides
# it for a training run; testing ignores it.
ILLEGAL_ACTION_PENALTY = float(os.environ.get("SHOWDOWN_ILLEGAL_PENALTY", "0.5"))

# One-off reward for a well-timed tera (see _good_tera): the battle reward pays for good tera only
# a little at a time over the following turns, and the first v4 agents learned to hardly ever
# tera. SHOWDOWN_TERA_BONUS overrides it for a training run; testing ignores it.
TERA_BONUS = float(os.environ.get("SHOWDOWN_TERA_BONUS", "0.3"))

# The same kind of bonus for the hand-written policy's other rules (see _rule_bonus): switching out
# of the opponent's predicted move, and setting up or healing while resisting it. The v4r agents
# make those plays 5-25% of the times the hand policy does. Each is paid at most once per opposing
# Pokémon, so switching back and forth or setting up again after a switch can't farm it. Off by
# default; SHOWDOWN_SWITCH_BONUS and SHOWDOWN_SETUP_BONUS (setup and heal) set them for a run.
SWITCH_BONUS = float(os.environ.get("SHOWDOWN_SWITCH_BONUS", "0"))
SETUP_BONUS = float(os.environ.get("SHOWDOWN_SETUP_BONUS", "0"))

# Tera types that don't replace the Pokémon's types (Stellar keeps its original ones)
NO_TYPE_CHANGE_TERA = {PokemonType.STELLAR, PokemonType.THREE_QUESTION_MARKS}

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

# Species -> strongest move in its random-battle movepools (scripts/build_strongest_moves.py)
with open(os.path.join(os.path.dirname(__file__), "data", "strongest_moves.json"), encoding="utf-8") as _f:
    STRONGEST_MOVES = {species: Move(move_id, gen=9) for species, move_id in json.load(_f).items()}


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one
        self._prev_battle_state = {}
        self._illegal_action = False
        self._bonus = 0.0
        self._bonus_paid = set()
        self._counts = self._new_counts()

    # =========================================================
    # Action space
    # =========================================================
    def _get_action_size(self) -> int | None:
        return N_MOVES + N_SWITCHES + N_TERA

    def process_action(self, action: np.int64) -> np.int64:
        """
        Agent action -> Showdown action:
          0-3    use move slot 0-3                 -> 6-9
          4-9    switch to team slot 0-5           -> 0-5
          10-13  terastallize, use move slot 0-3   -> 22-25
        When we can't terastallize (tera used up), 10-13 just use the move: the first v4 agents,
        penalised for those, learned to avoid the tera actions altogether. A good tera (see
        _good_tera) earns TERA_BONUS, a good switch, setup or heal (see _rule_bonus) SWITCH_BONUS
        or SETUP_BONUS.
        The agent also picks the replacement when one of ours faints. An illegal choice (a move
        during a forced switch, a fainted or already active team slot...) is passed on as is:
        poke_env, run non-strict, swaps it for the server's default choice, and calc_reward
        takes off ILLEGAL_ACTION_PENALTY. On steps where only the opponent has to choose (e.g.
        replacing a fainted Pokémon), poke_env ignores our action, so it is neither penalised
        nor counted.
        """
        action = int(action)
        battle = self.battle1
        acting = battle is not None and self.agent1_to_move
        if action >= TERA and not (acting and battle.can_tera):
            action -= TERA
        self._illegal_action = acting and not self._is_legal(battle, action)
        self._bonus = 0.0
        if acting:
            self._count_action(battle, action)
            if not self._illegal_action:
                if action >= TERA and self._good_tera(battle, action - TERA):
                    self._bonus = TERA_BONUS
                    self._counts["good_tera"] += 1
                self._bonus += self._rule_bonus(battle, action)

        if action < N_MOVES:
            return np.int64(action + 6)
        if action >= TERA:
            return np.int64(action - TERA + 22)
        return np.int64(action - N_MOVES)

    def _is_legal(self, battle: AbstractBattle, action: int) -> bool:
        """Mirrors the checks poke_env's action_to_order makes before accepting an action."""
        if self._waiting(battle):
            return False
        if action >= TERA:
            return battle.can_tera and self._is_legal(battle, action - TERA)
        if action < N_MOVES:
            if battle.force_switch or battle.active_pokemon is None:
                return False
            slots = self._move_slots(battle)
            available = {move.id for move in battle.available_moves}
            return action < len(slots) and slots[action].id in available

        team = list(battle.team.values())
        slot = action - N_MOVES
        if battle.trapped or slot >= len(team):
            return False
        return team[slot].base_species in {mon.base_species for mon in battle.available_switches}

    @staticmethod
    def _waiting(battle: AbstractBattle) -> bool:
        """The server is only waiting on the opponent this step, so nothing we choose is used."""
        return bool((battle.last_request or {}).get("wait"))

    @staticmethod
    def _move_slots(battle: AbstractBattle) -> list[Move]:
        """Moves in the order actions 0-3 use them (same rule as poke_env's action_to_order)."""
        moves = battle.available_moves
        if len(moves) == 1 and moves[0].id in ("struggle", "recharge"):
            return list(moves)
        if battle.active_pokemon is None:
            return []
        return list(battle.active_pokemon.moves.values())[:N_MOVES]

    # =========================================================
    # Battle knowledge used by the observation
    # =========================================================
    @classmethod
    def _expected_damage(cls, move, attacker, defender, use_boosts: bool = False, tera: bool = False) -> float:
        """
        Rough damage estimate: base power * accuracy * type multiplier * STAB (* attack boost).
        tera: as if the attacker terastallized first.
        """
        damage = move.base_power * move.accuracy * defender.damage_multiplier(move) * cls._stab(move, attacker, tera)
        if use_boosts and move.category != MoveCategory.STATUS:
            stat = "atk" if move.category == MoveCategory.PHYSICAL else "spa"
            stage = attacker.boosts.get(stat, 0)
            damage *= (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
        return damage

    @classmethod
    def _stab(cls, move, mon, tera: bool = False) -> float:
        """
        Same-type attack bonus. A terastallized Pokémon keeps STAB on its original types and gains
        it on its tera type, doubled if that is one of its original types.
        """
        stab = 1.5 if move.type in mon.original_types else 1.0
        tera_type = cls._tera_type(mon) if tera or mon.is_terastallized else None
        if tera_type is not None and tera_type not in NO_TYPE_CHANGE_TERA and move.type == tera_type:
            stab = 2.0 if stab > 1 else 1.5
        return stab

    @staticmethod
    def _tera_type(mon) -> PokemonType | None:
        """Our Pokémon's tera type: poke_env only learns it once we terastallize; the server's request has it."""
        if mon.is_terastallized:
            return mon.tera_type
        name = (mon._last_request or {}).get("teraType")
        return PokemonType.from_name(name) if name else None

    @staticmethod
    def _predicted_move(battle: AbstractBattle) -> tuple[Move | None, bool]:
        """
        The opponent's most likely next move and whether it is a guess. The max expert always uses
        its highest base-power move, so once its active Pokemon has attacked, its strongest revealed
        move is exactly what comes next. Before that, guess the strongest move in the species'
        random-battle movepools.
        """
        opp = battle.opponent_active_pokemon
        revealed = [move for move in opp.moves.values() if move.base_power > 0]
        if revealed:
            return max(revealed, key=lambda move: move.base_power), False
        guess = STRONGEST_MOVES.get(opp.species)
        return guess, guess is not None

    @classmethod
    def _threat(cls, move: Move, mon, tera: bool = False) -> float:
        """
        Type multiplier of an opponent's move on one of our Pokemon, counting immunities we know of.
        tera: as if it terastallized first (abilities and items still apply).
        """
        if IMMUNE_ABILITIES.get(mon.ability) == move.type.name:
            return 0.0
        if move.type.name == "GROUND" and mon.item == "airballoon":
            return 0.0
        tera_type = cls._tera_type(mon) if tera else None
        if tera_type is not None and tera_type not in NO_TYPE_CHANGE_TERA:
            return move.type.damage_multiplier(tera_type, None, type_chart=mon._data.type_chart)
        return mon.damage_multiplier(move)

    @staticmethod
    def _is_setup(move: Move) -> bool:
        """Status move that raises our own attack or special attack (Swords Dance, Nasty Plot...)."""
        return (
            move.category == MoveCategory.STATUS
            and move.target == Target.SELF
            and bool(move.boosts)
            and (move.boosts.get("atk", 0) > 0 or move.boosts.get("spa", 0) > 0)
        )

    @staticmethod
    def _is_heal(move: Move) -> bool:
        return move.heal > 0 or move.id in EXTRA_HEAL_MOVES

    def _good_tera(self, battle: AbstractBattle, slot: int) -> bool:
        """
        Terastallizing and using move slot `slot` is well-timed if, against the opponent's
        predicted move, it turns a neutral or super-effective hit into a resisted one, or gives the
        move (more) STAB without the predicted move hitting harder - the two cases where the
        hand-written policy in scripts/check_env.py teras (94.8% vs max).
        """
        active = battle.active_pokemon
        if active is None or battle.opponent_active_pokemon is None:
            return False
        predicted, _ = self._predicted_move(battle)
        if predicted is None:
            return False
        threat = self._threat(predicted, active)
        tera_threat = self._threat(predicted, active, tera=True)
        if threat >= 1 and tera_threat <= 0.5:
            return True
        move = self._move_slots(battle)[slot]
        return move.base_power > 0 and self._stab(move, active, tera=True) > self._stab(move, active) and tera_threat <= threat

    def _rule_bonus(self, battle: AbstractBattle, action: int) -> float:
        """
        Bonus for a legal action that follows one of the hand-written policy's other rules
        (scripts/check_env.py) against the opponent's predicted move:
          switch  it hits our active super-effectively and the teammate resists it, or it hits
                  at least neutrally and the teammate is immune
          heal    we resist it and are at half HP or less
          setup   we resist it and are at half HP or more, attack and special attack both
                  below +2 (+6 when we're immune)
        Every match is counted (good_switch, good_heal, good_setup); the bonus is paid once per
        rule and opposing Pokémon.
        """
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if battle.force_switch or active is None or opp is None:
            return 0.0
        predicted, _ = self._predicted_move(battle)
        if predicted is None:
            return 0.0

        threat = self._threat(predicted, active)
        hp = active.current_hp_fraction
        if N_MOVES <= action < TERA:
            target = list(battle.team.values())[action - N_MOVES]
            target_threat = self._threat(predicted, target)
            if not ((threat >= 2 and target_threat <= 0.5) or (threat >= 1 and target_threat == 0)):
                return 0.0
            rule, bonus = "switch", SWITCH_BONUS
        else:
            move = self._move_slots(battle)[action - TERA if action >= TERA else action]
            cap = 6 if threat == 0 else 2
            if self._is_heal(move) and threat <= 0.5 and hp <= 0.5:
                rule = "heal"
            elif (
                self._is_setup(move)
                and threat <= 0.5
                and hp >= 0.5
                and max(active.boosts.get("atk", 0), active.boosts.get("spa", 0)) < cap
            ):
                rule = "setup"
            else:
                return 0.0
            bonus = SETUP_BONUS

        self._counts[f"good_{rule}"] += 1
        if (rule, opp.species) in self._bonus_paid:
            return 0.0
        self._bonus_paid.add((rule, opp.species))
        return bonus

    # =========================================================
    # Reset / Step
    # =========================================================
    def reset(self, seed=None, options=None):
        """
        poke_env players keep every battle they've played (~1.2 MB each), which ran 200k-step
        runs out of memory. Drop the finished ones, except the battle that just ended: it can
        still get a late message (e.g. after /leave) that poke_env would wait on forever.
        """
        current = {battle.battle_tag for battle in (self.battle1, self.battle2) if battle is not None}
        for agent in (self.agent1, self.agent2):
            for tag, battle in list(agent._battles.items()):
                if battle.finished and tag not in current:
                    del agent._battles[tag]

        self._illegal_action = False
        self._bonus = 0.0
        self._bonus_paid = set()
        self._counts = self._new_counts()
        return super().reset(seed, options)

    def step(self, actions: dict[str, np.int64]):
        """
        Same as BaseShowdownEnv.step, but instead of deep-copying both battles every step
        (about 90% of a step's time), it keeps just the numbers calc_reward compares against.
        """
        self.n += 1
        self._prev_battle_state = {
            battle.player_username: self._reward_state(battle)
            for battle in (self.battle1, self.battle2)
            if battle is not None
        }

        actions[self.agents[0]] = self.process_action(actions[self.agents[0]])

        return super(BaseShowdownEnv, self).step(actions)

    @staticmethod
    def _reward_state(battle: AbstractBattle) -> tuple[float, float, int, int]:
        """
        (our missing HP, opponent's missing HP, our fainted, opponent's fainted), with HP
        in fractions of a Pokémon. Unseen opponent Pokémon aren't in the dict, so they
        count as 0 missing - revealing a new full-HP one doesn't count as healing.
        """
        return (
            sum(1.0 - mon.current_hp_fraction for mon in battle.team.values()),
            sum(1.0 - mon.current_hp_fraction for mon in battle.opponent_team.values()),
            sum(mon.fainted for mon in battle.team.values()),
            sum(mon.fainted for mon in battle.opponent_team.values()),
        )

    # =========================================================
    # Reward Function
    # =========================================================
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward this step:
          0.5 * damage dealt - 0.25 * damage taken   (in fractions of a Pokémon's HP)
          + 0.2 per opponent Pokémon knocked out, - 0.2 per one of ours that faints
          + 1 for a win / - 1 for a loss
          - ILLEGAL_ACTION_PENALTY if the agent's action couldn't be taken
          + TERA_BONUS for a well-timed tera (_good_tera)
          + SWITCH_BONUS or SETUP_BONUS for a good switch, setup or heal (_rule_bonus)
        """
        prior = self._prev_battle_state.get(battle.player_username)
        if prior is None:
            return 0.0

        my_missing, opp_missing, my_fainted, opp_fainted = self._reward_state(battle)

        damage_dealt = opp_missing - prior[1]
        damage_taken = my_missing - prior[0]
        knocked_out = opp_fainted - prior[3]
        lost = my_fainted - prior[2]

        reward = 0.5 * damage_dealt - 0.25 * damage_taken + 0.2 * (knocked_out - lost)

        if battle.won:
            reward += 1.0
        elif battle.lost:
            reward -= 1.0

        if battle is self.battle1:
            if self._illegal_action:
                reward -= ILLEGAL_ACTION_PENALTY
            reward += self._bonus

        return float(reward)

    # =========================================================
    # Observation space
    # =========================================================
    def _observation_size(self) -> int:
        """
        Per move slot (x4, same order as actions 0-3 and 10-13):
          legal now
          relative damage  (expected damage incl. our boosts / the best option's, best = 1; the
            options are every legal move, with and without tera)
          expected power   (base_power * accuracy / 150, capped at 1)
          type effectiveness vs opponent active (multiplier / 4)
          raises our attack or special attack (setup move)
          heals us
          relative damage if we terastallize first (0 when we can't tera now)
        Per team slot (x6, same order as actions 4-9):
          legal switch target now
          HP fraction
          how hard the opponent's predicted move hits it (multiplier / 4; without a prediction,
            the opponent's best type multiplier on it)
          its best move's expected damage vs opponent active / 300, capped at 1
          faster than opponent active (base speed)
        Battle context:
          our active HP, opponent active HP
          our Pokémon left / 6, opponent Pokémon left / 6
          1 if our active is faster (base speed)
          how hard the opponent's types hit our active (best multiplier / 4)
        Opponent's predicted move (see _predicted_move):
          revealed, guessed  (both 0 when there is no prediction)
          base_power * accuracy / 150, capped at 1
          how hard it hits our active (multiplier / 4)
          it gets STAB
          how hard it would hit our active after we terastallize (multiplier / 4; 0 when we
            can't tera now)
        Our attack and special attack boosts (stage / 6)
        Tera:
          we can terastallize now (otherwise actions 10-13 just use the move)
          our active is terastallized
          we haven't used tera this battle (only one Pokémon a battle can)
        1 on a forced switch (only switch actions are legal)
        Nothing is legal on steps where only the opponent chooses (our action is ignored).
        = 28 + 30 + 6 + 6 + 2 + 3 + 1 = 76 features
        """
        return 76

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        obs = np.zeros(self._observation_size(), dtype=np.float32)

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if opp is None:
            return obs

        predicted, guessed = self._predicted_move(battle)
        waiting = self._waiting(battle)
        moving = active is not None and not battle.force_switch and not waiting
        can_tera = moving and battle.can_tera

        # Move slots
        i = 0
        if moving:
            slots = self._move_slots(battle)
            available = {move.id for move in battle.available_moves}
            damage = [
                self._expected_damage(move, active, opp, use_boosts=True) if move.id in available else 0.0
                for move in slots
            ]
            tera_damage = [
                self._expected_damage(move, active, opp, use_boosts=True, tera=True)
                if can_tera and move.id in available
                else 0.0
                for move in slots
            ]
            best = max(damage + tera_damage, default=0.0) or 1.0
            for slot, move in enumerate(slots):
                if move.id not in available:
                    continue  # disabled / out of PP -> all zeros
                base = 7 * slot
                obs[base] = 1.0
                obs[base + 1] = damage[slot] / best
                obs[base + 2] = min(move.base_power * move.accuracy / 150.0, 1.0)
                obs[base + 3] = opp.damage_multiplier(move) / 4.0
                obs[base + 4] = float(self._is_setup(move))
                obs[base + 5] = float(self._is_heal(move))
                obs[base + 6] = tera_damage[slot] / best
        i = 7 * N_MOVES

        # Team slots
        switchable = set()
        if not battle.trapped and not waiting:
            switchable = {mon.base_species for mon in battle.available_switches}
        for slot, mon in enumerate(list(battle.team.values())[:N_SWITCHES]):
            base = i + 5 * slot
            if mon.fainted:
                continue
            obs[base] = float(mon.base_species in switchable)
            obs[base + 1] = mon.current_hp_fraction
            if predicted is not None:
                obs[base + 2] = self._threat(predicted, mon) / 4.0
            else:
                obs[base + 2] = max(mon.damage_multiplier(t) for t in opp.types) / 4.0
            best_damage = max((self._expected_damage(m, mon, opp) for m in mon.moves.values()), default=0.0)
            obs[base + 3] = min(best_damage / 300.0, 1.0)
            obs[base + 4] = float(mon.base_stats["spe"] > opp.base_stats["spe"])
        i += 5 * N_SWITCHES

        # Battle context
        if active is not None:
            obs[i] = active.current_hp_fraction
            obs[i + 4] = float(active.base_stats["spe"] > opp.base_stats["spe"])
            obs[i + 5] = max(active.damage_multiplier(t) for t in opp.types) / 4.0
        obs[i + 1] = opp.current_hp_fraction
        obs[i + 2] = sum(not mon.fainted for mon in battle.team.values()) / 6.0
        # Unseen opponent Pokémon aren't in the dict, so count fainted ones instead
        obs[i + 3] = 1.0 - sum(mon.fainted for mon in battle.opponent_team.values()) / 6.0
        i += 6

        # Opponent's predicted move
        if predicted is not None:
            obs[i] = float(not guessed)
            obs[i + 1] = float(guessed)
            obs[i + 2] = min(predicted.base_power * predicted.accuracy / 150.0, 1.0)
            if active is not None:
                obs[i + 3] = self._threat(predicted, active) / 4.0
            obs[i + 4] = float(predicted.type in opp.types)
            if can_tera:
                obs[i + 5] = self._threat(predicted, active, tera=True) / 4.0
        i += 6

        # Our boosts
        if active is not None:
            obs[i] = active.boosts.get("atk", 0) / 6.0
            obs[i + 1] = active.boosts.get("spa", 0) / 6.0
        i += 2

        # Tera
        obs[i] = float(can_tera)
        obs[i + 1] = float(active is not None and active.is_terastallized)
        obs[i + 2] = float(not battle.used_tera)
        i += 3

        obs[i] = float(battle.force_switch)
        return obs

    # =========================================================
    # Additional info (logging)
    # =========================================================
    @staticmethod
    def _new_counts() -> Dict[str, int]:
        return {
            "illegal_actions": 0,
            "illegal_tera": 0,
            "switches": 0,
            "forced_switches": 0,
            "setup_moves": 0,
            "heal_moves": 0,
            "tera": 0,
            "good_tera": 0,
            "good_switch": 0,
            "good_setup": 0,
            "good_heal": 0,
        }

    def _count_action(self, battle: AbstractBattle, action: int):
        """Per-battle tallies of the agent's choices, logged to train.csv by cares-rl."""
        if self._illegal_action:
            self._counts["illegal_actions"] += 1
            self._counts["illegal_tera"] += action >= TERA
        elif N_MOVES <= action < TERA:
            self._counts["forced_switches" if battle.force_switch else "switches"] += 1
        else:
            if action >= TERA:
                self._counts["tera"] += 1
                action -= TERA
            move = self._move_slots(battle)[action]
            self._counts["setup_moves"] += self._is_setup(move)
            self._counts["heal_moves"] += self._is_heal(move)

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["turns"] = self.battle1.turn
            info[agent]["opp_fainted"] = sum(m.fainted for m in self.battle1.opponent_team.values())
            info[agent]["my_fainted"] = sum(m.fainted for m in self.battle1.team.values())
            info[agent].update(self._counts)
        return info



########################################
# DO NOT EDIT BELOW THIS LINE
########################################

class SingleShowdownWrapper(SingleAgentWrapper):
    """
    Wrapper for single-agent training against specified opponents.
    """

    def __init__(self, team_type: str = "random", opponent_type: str = "random", evaluation: bool = False):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"
        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)
        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")
        bot_teams = {}
        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8") as file:
                    bot_teams[team_file[:-4]] = file.read()
        return bot_teams.get(team_type, None)
