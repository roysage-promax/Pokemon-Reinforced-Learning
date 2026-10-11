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
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player
from .base_environment import BaseShowdownEnv


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

    # =========================================================
    # Action space
    # =========================================================
    def _get_action_size(self) -> int | None:
        return 4  # one action per move slot

    def process_action(self, action: np.int64) -> np.int64:
        """
        Agent action 0-3 -> Showdown move actions 6-9.
        On a forced switch no move is legal, so pick the replacement with the best
        type matchup instead (Showdown switch actions 0-5).
        """
        if self.battle1 is not None and self.battle1.force_switch:
            return self._best_switch(self.battle1)
        return np.int64(action + 6)

    def _best_switch(self, battle: AbstractBattle) -> np.int64:
        """
        Switch action for the available Pokémon with the best matchup against the
        opponent's active Pokémon: how hard its best move hits them, minus how hard
        their types hit it. Falls back to the server default (-2) if anything is missing.
        """
        opp = battle.opponent_active_pokemon
        if opp is None or not battle.available_switches:
            return np.int64(-2)

        def matchup(mon) -> float:
            offence = max(
                (self._expected_damage(move, mon, opp) for move in mon.moves.values()),
                default=0.0,
            ) / 150.0
            defence = max(mon.damage_multiplier(t) for t in opp.types)
            return offence - defence

        try:
            best = max(battle.available_switches, key=matchup)
        except Exception:
            # Never crash a long training run over the heuristic (e.g. odd typings)
            return np.int64(-2)

        return np.int64(list(battle.team.values()).index(best))

    @staticmethod
    def _expected_damage(move, attacker, defender) -> float:
        """Rough damage estimate: base power * accuracy * type multiplier * STAB."""
        stab = 1.5 if move.type in attacker.types else 1.0
        return move.base_power * move.accuracy * defender.damage_multiplier(move) * stab

    # =========================================================
    # Step
    # =========================================================
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

        return float(reward)

    # =========================================================
    # Observation space
    # =========================================================
    def _observation_size(self) -> int:
        """
        Embedding structure, per move slot (x4):
          available        (1 if the move can be used this turn)
          relative damage  (expected damage / best available move's expected damage, best = 1)
          expected power   (base_power * accuracy / 150, capped at 1)
          type effectiveness vs opponent active (multiplier / 4)
        then battle context:
          our active HP, opponent active HP
          our Pokémon left / 6, opponent Pokémon left / 6
          1 if our active is faster (base speed)
          how hard the opponent's types hit our active (best multiplier / 4)
        = 16 + 6 = 22 features
        """
        return 22

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Per-move features for the 4 move slots, in the same order that actions 6-9
        use (active_pokemon.moves), then battle context. Unusable moves are left as all zeros.
        """
        obs = np.zeros(self._observation_size(), dtype=np.float32)

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if active is None or opp is None:
            return obs

        available_ids = {m.id for m in battle.available_moves}
        moves = list(active.moves.values())[:4]

        damage = [
            self._expected_damage(move, active, opp) if move.id in available_ids else 0.0
            for move in moves
        ]
        best = max(damage, default=0.0) or 1.0

        for i, move in enumerate(moves):
            if move.id not in available_ids:
                continue  # disabled / out of PP / forced switch -> all zeros
            obs[4 * i] = 1.0
            obs[4 * i + 1] = damage[i] / best
            obs[4 * i + 2] = min(move.base_power * move.accuracy / 150.0, 1.0)
            obs[4 * i + 3] = opp.damage_multiplier(move) / 4.0

        obs[16] = active.current_hp_fraction
        obs[17] = opp.current_hp_fraction
        obs[18] = sum(not mon.fainted for mon in battle.team.values()) / 6.0
        # Unseen opponent Pokémon aren't in the dict, so count fainted ones instead
        obs[19] = 1.0 - sum(mon.fainted for mon in battle.opponent_team.values()) / 6.0
        obs[20] = 1.0 if active.base_stats["spe"] > opp.base_stats["spe"] else 0.0
        obs[21] = max(active.damage_multiplier(t) for t in opp.types) / 4.0

        return obs

    # =========================================================
    # Additional info (logging)
    # =========================================================
    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["turns"] = self.battle1.turn
            info[agent]["opp_fainted"] = sum(m.fainted for m in self.battle1.opponent_team.values())
            info[agent]["my_fainted"] = sum(m.fainted for m in self.battle1.team.values())
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
