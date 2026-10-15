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
HEAL = TERA + N_TERA  # action 14: use our healing move, whichever slot it is in
SETUP = HEAL + 1  # action 15: use our setup move, whichever slot it is in
BEST_ATTACK = SETUP + 1  # action 16: use our most damaging move
BEST_TERA = BEST_ATTACK + 1  # action 17: terastallize and use our most damaging move after tera
SAFE_SWITCH = BEST_TERA + 1  # action 18: switch to the teammate the opponent's predicted move hurts least
MATCHUP_SWITCH = SAFE_SWITCH + 1  # action 19: switch to the teammate with the best matchup
N_ACTIONS = MATCHUP_SWITCH + 1
MACRO_NAMES = ["heal", "setup", "attack", "tera", "safe_switch", "matchup_switch"]  # actions 14-19

# Penalty for an action that can't be taken this turn; poke_env then sends the server's default choice.
# 0.5 cut illegal actions to ~0.2 a battle in v3 (0.1 left ~2). SHOWDOWN_ILLEGAL_PENALTY overrides
# it for a training run; testing ignores it.
ILLEGAL_ACTION_PENALTY = float(os.environ.get("SHOWDOWN_ILLEGAL_PENALTY", "0.5"))

# One-off reward for a well-timed tera (see _good_tera): the battle reward pays for good tera only
# a little at a time over the following turns, and the first v4 agents learned to hardly ever
# tera. SHOWDOWN_TERA_BONUS overrides it for a training run; testing ignores it.
TERA_BONUS = float(os.environ.get("SHOWDOWN_TERA_BONUS", "0.3"))

# The same kind of bonus for the hand-written policy's other rules (see _rule_bonus): switching out
# of the opponent's predicted move, setting up while resisting it, healing when it doesn't hit
# super-effectively. The v4r agents make those plays 5-25% of the times the hand policy does. A
# switch earns it once per opposing Pokémon, a heal once per pair of active Pokémon, a setup once
# per pair and boost stage - so switching back and forth or setting up again after a switch can't
# farm it. Off by default; SHOWDOWN_SWITCH_BONUS and SHOWDOWN_SETUP_BONUS (setup and heal) set
# them for a run.
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
        return N_ACTIONS

    def process_action(self, action: np.int64) -> np.int64:
        """
        Agent action -> Showdown action:
          0-3    use move slot 0-3                 -> 6-9
          4-9    switch to team slot 0-5           -> 0-5
          10-13  terastallize, use move slot 0-3   -> 22-25
          14     use our healing move              -> 6-9, the slot it is in
          15     use our setup move                -> 6-9, the slot it is in
          16     use our most damaging move        -> 6-9
          17     terastallize, use our most damaging move after tera -> 22-25
          18     switch to the teammate the opponent's predicted move hurts least -> 0-5
          19     switch to the teammate with the best matchup (see _switch_target) -> 0-5
        When we can't terastallize (tera used up), 10-13 and 17 just use the move: the first v4
        agents, penalised for those, learned to avoid the tera actions altogether. Likewise 14 and
        15 use our best attack when we have no such move (see _kind_slot). 14-19 exist because a
        move or teammate can sit in any slot and the network learns each slot's output separately:
        the v4s agents healed mostly when the move was in slot 0, and the v5 agents picked a weaker
        attack than the best ~1.4 times a battle and the hand-written policy's switch target about
        half the time - while using 14 / 15's best-attack fallback ~7 times a battle. 14-19 are the
        hand-written policy's choices, so the agent only has to learn when to make which kind of
        play; 0-13 stay, so it can still make any play. A good tera (see _good_tera) earns
        TERA_BONUS, a good switch, setup or heal (see _rule_bonus) SWITCH_BONUS or SETUP_BONUS.
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
        if action >= HEAL:
            kind = MACRO_NAMES[action - HEAL]
            action, found = self._macro(battle, action) if acting else (0, False)
            if acting:
                self._counts[f"{kind}_action"] += 1
                self._counts["action_fallback"] += not found
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
        if action >= HEAL:
            return self._is_legal(battle, self._macro(battle, action)[0])
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

    def _macro(self, battle: AbstractBattle, action: int) -> tuple[int, bool]:
        """
        The action 0-13 that action 14-19 stands for this turn, and whether it found what it
        names (a healing / setup move, tera, a teammate to switch to). Without one, 14, 15 and 17
        use our best attack; 18 and 19 an illegal switch, like 4-9 when nothing can come in.
        """
        if action in (HEAL, SETUP):
            return self._kind_slot(battle, action)
        if action in (BEST_ATTACK, BEST_TERA):
            tera = action == BEST_TERA and battle.can_tera
            slot = self._best_slot(battle, tera)
            return (TERA + slot if tera else slot), tera or action == BEST_ATTACK
        slot = self._switch_target(battle, safest=action == SAFE_SWITCH)
        return N_MOVES + (slot or 0), slot is not None

    def _legal_moves(self, battle: AbstractBattle) -> list[tuple[int, Move]]:
        available = {move.id for move in battle.available_moves}
        return [(slot, move) for slot, move in enumerate(self._move_slots(battle)) if move.id in available]

    def _kind_slot(self, battle: AbstractBattle, action: int) -> tuple[int, bool]:
        """
        The move slot action HEAL or SETUP uses, and whether it found such a move: the first
        usable healing / setup move, else our best attack (_best_slot).
        """
        is_kind = self._is_heal if action == HEAL else self._is_setup
        for slot, move in self._legal_moves(battle):
            if is_kind(move):
                return slot, True
        return self._best_slot(battle), False

    def _best_slot(self, battle: AbstractBattle, tera: bool = False) -> int:
        """
        The usable move slot with the highest relative damage in the observation (expected
        damage with our boosts; tera: after terastallizing). 0 when no move is usable.
        """
        legal = self._legal_moves(battle)
        active, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if not legal or active is None or opp is None:
            return 0
        slot, _ = max(legal, key=lambda pair: self._expected_damage(pair[1], active, opp, use_boosts=True, tera=tera))
        return slot

    def _switch_target(self, battle: AbstractBattle, safest: bool) -> int | None:
        """
        The team slot actions SAFE_SWITCH (safest) and MATCHUP_SWITCH switch to, from the
        teammates that can come in: the one the opponent's predicted move hits least (ties: best
        matchup), or the best matchup = 2 * offence - threat (_team_member, as in the
        observation). These are the hand-written policy's switch out of a threat and its pick
        when one of ours faints. None when no teammate can come in.
        """
        options = self._switch_options(battle)
        if not options:
            return None
        if safest:
            return min(options, key=lambda o: (o[1], -(2 * o[2] - o[1])))[0]
        return max(options, key=lambda o: 2 * o[2] - o[1])[0]

    def _switch_options(self, battle: AbstractBattle) -> list[tuple[int, float, float]]:
        """(team slot, threat, offence) of each teammate that can come in now."""
        opp = battle.opponent_active_pokemon
        if opp is None or battle.trapped or self._waiting(battle):
            return []
        switchable = {mon.base_species for mon in battle.available_switches}
        predicted, _ = self._predicted_move(battle)
        return [
            (slot, *self._team_member(mon, opp, predicted))
            for slot, mon in enumerate(list(battle.team.values())[:N_SWITCHES])
            if not mon.fainted and mon.base_species in switchable
        ]

    @classmethod
    def _team_member(cls, mon, opp, predicted: Move | None) -> tuple[float, float]:
        """
        (threat, offence) of one of our Pokémon against the opponent's active: how hard the
        predicted move hits it (type multiplier; without a prediction, the opponent's best type
        multiplier on it), and its best move's expected damage on the opponent / 300, capped at 1.
        """
        if predicted is not None:
            threat = cls._threat(predicted, mon)
        else:
            threat = max(mon.damage_multiplier(t) for t in opp.types)
        best_damage = max((cls._expected_damage(m, mon, opp) for m in mon.moves.values()), default=0.0)
        # Rounded like the observation, so equal matchups tie-break the same way in both
        return threat, float(np.float32(min(best_damage / 300.0, 1.0)))

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
          heal    it doesn't hit us super-effectively and we are at half HP or less (healing
                  only while resisting it was worth nothing in scripts/scripted_probe.py; on
                  neutral hits too +1.2 points)
          setup   we resist it and are at half HP or more, attack and special attack both
                  below +2 (+6 when we're immune)
        Every match is counted (good_switch, good_heal, good_setup); the bonus is paid once per
        opposing Pokémon for a switch, once per pair of active Pokémon for a heal, and once per
        pair and boost stage for a setup (switching out resets boosts, so that can't farm it).
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
            rule, bonus, key = "switch", SWITCH_BONUS, ("switch", opp.species)
        else:
            move = self._move_slots(battle)[action - TERA if action >= TERA else action]
            cap = 6 if threat == 0 else 2
            boost = max(active.boosts.get("atk", 0), active.boosts.get("spa", 0))
            if self._is_heal(move) and threat <= 1 and hp <= 0.5:
                rule, key = "heal", ("heal", opp.species, active.species)
            elif self._is_setup(move) and threat <= 0.5 and hp >= 0.5 and boost < cap:
                rule, key = "setup", ("setup", opp.species, active.species, boost)
            else:
                return 0.0
            bonus = SETUP_BONUS

        self._counts[f"good_{rule}"] += 1
        if key in self._bonus_paid:
            return 0.0
        self._bonus_paid.add(key)
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
        We have a usable healing move, a usable setup move (actions 14 and 15 use our best attack
          when we don't)
        What actions 17-19 would do:
          relative damage of the best move after terastallizing (0 when we can't tera now)
          a teammate can come in now (18 and 19 are legal)
          the safest switch's threat (/ 4) and offence, the best-matchup switch's threat (/ 4)
            and offence (same scale as the team slots; 0 when nobody can come in)
        Nothing is legal on steps where only the opponent chooses (our action is ignored).
        = 28 + 30 + 6 + 6 + 2 + 3 + 1 + 2 + 6 = 84 features
        """
        return 84

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
            threat, offence = self._team_member(mon, opp, predicted)
            obs[base] = float(mon.base_species in switchable)
            obs[base + 1] = mon.current_hp_fraction
            obs[base + 2] = threat / 4.0
            obs[base + 3] = offence
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
        i += 1

        # What actions 14 and 15 would use (the per-slot flags are 0 for unusable moves)
        obs[i] = float(any(obs[7 * slot + 5] for slot in range(N_MOVES)))
        obs[i + 1] = float(any(obs[7 * slot + 4] for slot in range(N_MOVES)))
        i += 2

        # What actions 17-19 would do
        obs[i] = max(obs[7 * slot + 6] for slot in range(N_MOVES))
        safest, matchup = self._switch_target(battle, safest=True), self._switch_target(battle, safest=False)
        if safest is not None:
            team_obs = 7 * N_MOVES
            obs[i + 1] = 1.0
            obs[i + 2 : i + 4] = obs[[team_obs + 5 * safest + 2, team_obs + 5 * safest + 3]]
            obs[i + 4 : i + 6] = obs[[team_obs + 5 * matchup + 2, team_obs + 5 * matchup + 3]]
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
            **{f"{kind}_action": 0 for kind in MACRO_NAMES},
            "action_fallback": 0,
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


# Species -> strongest move in its random-battle movepools, for guessing an opponent's next move
# before it has attacked (see _predicted_move). Written by scripts/build_strongest_moves.py from
# Showdown's gen9 random-battle sets; it lives here because only this file is submitted.
# BEGIN STRONGEST MOVES
STRONGEST_MOVE_IDS = {
    "abomasnow": "woodhammer", "alcremie": "alluringvoice", "alomomola": "scald",
    "altaria": "bravebird", "ambipom": "doubleedge", "amoonguss": "sludgebomb",
    "ampharos": "focusblast", "annihilape": "gunkshot", "appletun": "dracometeor",
    "araquanid": "hydropump", "arbok": "gunkshot", "arboliva": "earthpower",
    "arcanine": "closecombat", "arcaninehisui": "headsmash", "arceus": "earthquake",
    "arceusbug": "fireblast", "arceusdark": "judgment", "arceusdragon": "flareblitz",
    "arceuselectric": "judgment", "arceusfairy": "judgment", "arceusfighting": "bodypress",
    "arceusfire": "flareblitz", "arceusflying": "judgment", "arceusghost": "focusblast",
    "arceusgrass": "fireblast", "arceusground": "fireblast", "arceusice": "judgment",
    "arceuspoison": "flareblitz", "arceuspsychic": "bodypress", "arceusrock": "fireblast",
    "arceussteel": "earthquake", "arceuswater": "judgment", "archaludon": "dracometeor",
    "ariados": "megahorn", "armarouge": "armorcannon", "articuno": "bravebird",
    "articunogalar": "hurricane", "avalugg": "bodypress", "avalugghisui": "stoneedge",
    "azelf": "explosion", "azumarill": "superpower", "banette": "gunkshot",
    "barraskewda": "closecombat", "basculegion": "wavecrash", "basculegionf": "wavecrash",
    "basculin": "doubleedge", "bastiodon": "foulplay", "baxcalibur": "glaiverush",
    "beartic": "closecombat", "bellibolt": "muddywater", "bellossom": "moonblast",
    "bisharp": "ironhead", "blastoise": "hydropump", "blaziken": "closecombat",
    "bombirdier": "bravebird", "brambleghast": "powerwhip", "braviary": "bravebird",
    "braviaryhisui": "hurricane", "breloom": "rocktomb", "bronzong": "earthquake",
    "brutebonnet": "closecombat", "bruxish": "wavecrash", "cacturne": "leafstorm",
    "calyrex": "psychic", "calyrexice": "closecombat", "calyrexshadow": "astralbarrage",
    "camerupt": "overheat", "carbink": "moonblast", "ceruledge": "closecombat",
    "cetitan": "superpower", "chandelure": "fireblast", "charizard": "focusblast",
    "chesnaught": "woodhammer", "chienpao": "sacredsword", "chimecho": "dazzlinggleam",
    "chiyu": "overheat", "cinccino": "uturn", "cinderace": "highjumpkick", "clawitzer": "icebeam",
    "clefable": "fireblast", "clodsire": "gunkshot", "cloyster": "hydropump",
    "coalossal": "overheat", "cobalion": "closecombat", "comfey": "terablast",
    "conkeldurr": "closecombat", "copperajah": "superpower", "corviknight": "bravebird",
    "crabominable": "closecombat", "cramorant": "bravebird", "crawdaunt": "closecombat",
    "cresselia": "moonblast", "cryogonal": "icebeam", "cyclizar": "dracometeor",
    "dachsbun": "playrough", "darkrai": "focusblast", "decidueye": "leafstorm",
    "decidueyehisui": "leafblade", "dedenne": "thunderbolt", "delibird": "bravebird",
    "delphox": "focusblast", "deoxys": "psychoboost", "deoxysattack": "psychoboost",
    "deoxysdefense": "psychicnoise", "deoxysspeed": "psychoboost", "dewgong": "hydropump",
    "dialga": "dracometeor", "dialgaorigin": "dracometeor", "diancie": "diamondstorm",
    "dipplin": "dragonpulse", "dodrio": "bravebird", "dondozo": "wavecrash",
    "donphan": "earthquake", "dragalge": "dracometeor", "dragapult": "dracometeor",
    "dragonite": "outrage", "drednaw": "stoneedge", "drifblim": "shadowball",
    "dudunsparce": "boomburst", "dugtrio": "earthquake", "dugtrioalola": "earthquake",
    "dunsparce": "earthquake", "duraludon": "dracometeor", "dusknoir": "focuspunch",
    "eelektross": "closecombat", "eiscue": "liquidation", "electivire": "earthquake",
    "electrode": "explosion", "electrodehisui": "leafstorm", "emboar": "headsmash",
    "empoleon": "icebeam", "enamorus": "superpower", "enamorustherian": "superpower",
    "entei": "flareblitz", "espathra": "dazzlinggleam", "espeon": "psychic",
    "eternatus": "meteorbeam", "excadrill": "earthquake", "exeggutor": "psychic",
    "exeggutoralola": "dracometeor", "falinks": "closecombat", "farigiraf": "bodyslam",
    "feraligatr": "liquidation", "fezandipiti": "gunkshot", "flamigo": "bravebird",
    "flapple": "outrage", "flareon": "flareblitz", "floatzel": "wavecrash", "florges": "moonblast",
    "fluttermane": "moonblast", "flygon": "outrage", "forretress": "ironhead",
    "froslass": "poltergeist", "frosmoth": "hurricane", "furret": "doubleedge",
    "gallade": "leafblade", "galvantula": "thunder", "garchomp": "outrage",
    "gardevoir": "focusblast", "garganacl": "earthquake", "gastrodon": "earthquake",
    "gengar": "focusblast", "gholdengo": "focusblast", "girafarig": "psychic",
    "giratina": "dragonpulse", "giratinaorigin": "dracometeor", "glaceon": "freezedry",
    "glalie": "earthquake", "glastrier": "closecombat", "glimmora": "meteorbeam",
    "gliscor": "earthquake", "gogoat": "earthquake", "golduck": "hydropump", "golem": "explosion",
    "golemalola": "explosion", "golurk": "poltergeist", "goodra": "dracometeor",
    "goodrahisui": "dracometeor", "gothitelle": "focusblast", "gougingfire": "outrage",
    "grafaiai": "gunkshot", "granbull": "earthquake", "greattusk": "closecombat",
    "greedent": "doubleedge", "greninja": "gunkshot", "greninjabond": "gunkshot",
    "grimmsnarl": "spiritbreak", "groudon": "precipiceblades", "grumpig": "focusblast",
    "gumshoos": "doubleedge", "gurdurr": "drainpunch", "gyarados": "earthquake",
    "hariyama": "closecombat", "hatterene": "psychic", "hawlucha": "bravebird",
    "haxorus": "closecombat", "heatran": "magmastorm", "heracross": "closecombat",
    "hippowdon": "earthquake", "hitmonchan": "poisonjab", "hitmonlee": "highjumpkick",
    "hitmontop": "closecombat", "honchkrow": "bravebird", "hooh": "bravebird",
    "hoopa": "focusblast", "hoopaunbound": "gunkshot", "houndoom": "fireblast",
    "houndstone": "poltergeist", "hydrapple": "leafstorm", "hydreigon": "dracometeor",
    "hypno": "focusblast", "illumise": "bugbuzz", "incineroar": "overheat",
    "indeedee": "hypervoice", "indeedeef": "hypervoice", "infernape": "overheat",
    "inteleon": "hydropump", "ironboulder": "closecombat", "ironbundle": "hydropump",
    "ironcrown": "focusblast", "ironhands": "closecombat", "ironjugulis": "meteorbeam",
    "ironleaves": "closecombat", "ironmoth": "fireblast", "ironthorns": "earthquake",
    "irontreads": "earthquake", "ironvaliant": "closecombat", "jirachi": "psychic",
    "jolteon": "thunderbolt", "jumpluff": "uturn", "keldeoresolute": "hydropump",
    "kilowattrel": "hurricane", "kingambit": "kowtowcleave", "kingdra": "dracometeor",
    "klawf": "crabhammer", "kleavor": "closecombat", "klefki": "foulplay", "komala": "earthquake",
    "kommoo": "boomburst", "koraidon": "closecombat", "kricketune": "knockoff",
    "krookodile": "gunkshot", "kyogre": "waterspout", "kyurem": "dracometeor",
    "kyuremblack": "fusionbolt", "kyuremwhite": "dracometeor", "landorus": "focusblast",
    "landorustherian": "earthquake", "lanturn": "thunderbolt", "lapras": "hydropump",
    "latias": "dracometeor", "latios": "dracometeor", "leafeon": "doubleedge",
    "leavanny": "leafblade", "lilligant": "petaldance", "lilliganthisui": "closecombat",
    "lokix": "firstimpression", "lucario": "closecombat", "ludicolo": "leafstorm",
    "lugia": "aeroblast", "lumineon": "hydropump", "lunala": "focusblast", "lurantis": "leafstorm",
    "luvdisc": "surf", "luxray": "supercellslam", "lycanroc": "closecombat",
    "lycanrocdusk": "closecombat", "lycanrocmidnight": "closecombat", "mabosstiff": "playrough",
    "magcargo": "fireblast", "magearna": "fleurcannon", "magmortar": "focusblast",
    "magnezone": "thunderbolt", "malamar": "superpower", "mamoswine": "earthquake",
    "manaphy": "hydropump", "mandibuzz": "bravebird", "masquerain": "hurricane", "maushold": "bite",
    "medicham": "closecombat", "meganium": "earthquake", "meloetta": "focusblast",
    "meowscarada": "flowertrick", "meowstic": "alluringvoice", "meowsticf": "psychic",
    "mesprit": "icebeam", "metagross": "earthquake", "mew": "closecombat", "mewtwo": "fireblast",
    "mienshao": "highjumpkick", "mightyena": "playrough", "milotic": "icebeam",
    "mimikyu": "playrough", "minior": "earthquake", "minun": "thunderbolt",
    "miraidon": "dracometeor", "misdreavus": "shadowball", "mismagius": "energyball",
    "moltres": "bravebird", "moltresgalar": "hurricane", "morpeko": "aurawheel",
    "mudsdale": "earthquake", "muk": "gunkshot", "mukalola": "gunkshot", "munkidori": "focusblast",
    "necrozma": "earthquake", "necrozmadawnwings": "moongeistbeam",
    "necrozmaduskmane": "earthquake", "ninetales": "solarbeam", "ninetalesalola": "blizzard",
    "noctowl": "hurricane", "noivern": "boomburst", "ogerpon": "superpower",
    "ogerponcornerstone": "powerwhip", "ogerponhearthflame": "powerwhip",
    "ogerponwellspring": "woodhammer", "oinkologne": "doubleedge", "oinkolognef": "doubleedge",
    "okidogi": "gunkshot", "oranguru": "focusblast", "oricorio": "hurricane",
    "oricoriopau": "hurricane", "oricoriopompom": "hurricane", "oricoriosensu": "hurricane",
    "orthworm": "irontail", "overqwil": "gunkshot", "pachirisu": "thunderbolt",
    "palafin": "closecombat", "palkia": "dracometeor", "palkiaorigin": "dracometeor",
    "palossand": "earthpower", "passimian": "closecombat", "pawmot": "closecombat",
    "pecharunt": "malignantchain", "pelipper": "hurricane", "perrserker": "closecombat",
    "persian": "doubleedge", "persianalola": "foulplay", "phione": "icebeam",
    "pikachu": "volttackle", "piloswine": "earthquake", "pincurchin": "thunderbolt",
    "plusle": "thunderbolt", "politoed": "focusblast", "poliwrath": "closecombat",
    "polteageist": "shadowball", "porygon2": "icebeam", "porygonz": "icebeam",
    "primarina": "hydropump", "probopass": "bodypress", "pyroar": "fireblast",
    "quagsire": "earthquake", "quaquaval": "closecombat", "qwilfish": "gunkshot",
    "qwilfishhisui": "gunkshot", "rabsca": "bugbuzz", "ragingbolt": "dracometeor",
    "raichu": "focusblast", "raichualola": "focusblast", "raikou": "thunderbolt",
    "rampardos": "headsmash", "rayquaza": "dragonascent", "regice": "icebeam",
    "regidrago": "dragonenergy", "regieleki": "explosion", "regigigas": "bodyslam",
    "regirock": "stoneedge", "registeel": "bodypress", "reshiram": "blueflare",
    "reuniclus": "focusblast", "revavroom": "gunkshot", "rhydon": "megahorn",
    "rhyperior": "megahorn", "ribombee": "moonblast", "rillaboom": "woodhammer",
    "roaringmoon": "outrage", "rotom": "thunderbolt", "rotomfan": "thunderbolt",
    "rotomfrost": "blizzard", "rotomheat": "overheat", "rotommow": "leafstorm",
    "rotomwash": "hydropump", "sableye": "knockoff", "salamence": "outrage",
    "salazzle": "fireblast", "samurott": "megahorn", "samurotthisui": "sacredsword",
    "sandaconda": "earthquake", "sandslash": "earthquake", "sandslashalola": "earthquake",
    "sandyshocks": "earthpower", "sawsbuck": "doubleedge", "sceptile": "leafstorm",
    "scizor": "closecombat", "scovillain": "leafstorm", "scrafty": "poisonjab",
    "screamtail": "playrough", "scyther": "closecombat", "serperior": "leafstorm",
    "seviper": "gunkshot", "shaymin": "seedflare", "shayminsky": "seedflare",
    "shiftry": "leafstorm", "sinistcha": "matchagotcha", "skarmory": "bravebird",
    "skeledirge": "shadowball", "skuntank": "gunkshot", "slaking": "gigaimpact",
    "slitherwing": "closecombat", "slowbro": "futuresight", "slowbrogalar": "fireblast",
    "slowking": "futuresight", "slowkinggalar": "futuresight", "smeargle": "ceaselessedge",
    "sneasler": "closecombat", "snorlax": "earthquake", "solgaleo": "closecombat",
    "spectrier": "darkpulse", "spidops": "uturn", "spiritomb": "poltergeist",
    "squawkabilly": "bravebird", "squawkabillyblue": "bravebird", "squawkabillywhite": "bravebird",
    "squawkabillyyellow": "bravebird", "stantler": "earthquake", "staraptor": "bravebird",
    "stonjourner": "earthquake", "sudowoodo": "headsmash", "suicune": "icebeam",
    "sunflora": "leafstorm", "swalot": "gunkshot", "swampert": "earthquake", "swanna": "bravebird",
    "sylveon": "hypervoice", "talonflame": "overheat", "tatsugiri": "dracometeor",
    "tauros": "closecombat", "taurospaldeaaqua": "closecombat", "taurospaldeablaze": "closecombat",
    "taurospaldeacombat": "closecombat", "tentacruel": "sludgebomb", "terapagos": "terastarstorm",
    "terrakion": "closecombat", "thundurus": "focusblast", "thundurustherian": "focusblast",
    "tinglu": "earthquake", "tinkaton": "gigatonhammer", "toedscruel": "leafstorm",
    "torkoal": "solarbeam", "tornadus": "focusblast", "tornadustherian": "focusblast",
    "torterra": "headlongrush", "toucannon": "boomburst", "toxapex": "liquidation",
    "toxicroak": "closecombat", "toxtricity": "boomburst", "trevenant": "woodhammer",
    "tropius": "earthquake", "tsareena": "highjumpkick", "typhlosion": "eruption",
    "typhlosionhisui": "eruption", "tyranitar": "earthquake", "umbreon": "foulplay",
    "ursaluna": "headlongrush", "ursalunabloodmoon": "bloodmoon", "ursaring": "closecombat",
    "urshifu": "closecombat", "urshifurapidstrike": "closecombat", "uxie": "psychicnoise",
    "vaporeon": "icebeam", "veluza": "aquacutter", "venomoth": "sludgewave",
    "venusaur": "sludgebomb", "vespiquen": "hurricane", "victreebel": "powerwhip",
    "vigoroth": "earthquake", "vikavolt": "bugbuzz", "vileplume": "sludgebomb",
    "virizion": "closecombat", "vivillon": "hurricane", "volbeat": "lunge",
    "volcanion": "steameruption", "volcarona": "fireblast", "walkingwake": "dracometeor",
    "weavile": "knockoff", "weezing": "gunkshot", "weezinggalar": "gunkshot",
    "whimsicott": "moonblast", "whiscash": "hydropump", "wigglytuff": "fireblast",
    "wochien": "gigadrain", "wugtrio": "liquidation", "wyrdeer": "megahorn", "yanmega": "bugbuzz",
    "zacian": "closecombat", "zaciancrowned": "closecombat", "zamazenta": "closecombat",
    "zamazentacrowned": "stoneedge", "zangoose": "closecombat", "zapdos": "hurricane",
    "zapdosgalar": "bravebird", "zarude": "closecombat", "zebstrika": "overheat",
    "zekrom": "boltstrike", "zoroark": "focusblast", "zoroarkhisui": "focusblast",
}
# END STRONGEST MOVES
STRONGEST_MOVES = {species: Move(move_id, gen=9) for species, move_id in STRONGEST_MOVE_IDS.items()}



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
