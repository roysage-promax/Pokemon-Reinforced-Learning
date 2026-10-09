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
                (
                    move.base_power
                    * move.accuracy
                    * opp.damage_multiplier(move)
                    * (1.5 if move.type in mon.types else 1.0)
                    for move in mon.moves.values()
                ),
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

    # =========================================================
    # Reward Function
    # =========================================================
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward = 0.1 * (damage dealt - damage taken) this step, plus +1 for a win / -1 for a loss.
        Damage is the change in each team's missing HP, so revealing a new
        full-HP opponent Pokémon doesn't count as healing.
        """
        prior_battle = self._get_prior_battle(battle)
        if prior_battle is None:
            return 0.0

        def missing_hp(team) -> float:
            # Unseen opponent Pokémon aren't in the dict, so they count as 0 missing
            return sum(1.0 - mon.current_hp_fraction for mon in team.values())

        damage_dealt = missing_hp(battle.opponent_team) - missing_hp(prior_battle.opponent_team)
        damage_taken = missing_hp(battle.team) - missing_hp(prior_battle.team)

        reward = 0.1 * (damage_dealt - damage_taken)

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
          expected power   (base_power * accuracy / 150, capped at 1)
          type effectiveness vs opponent active (multiplier / 4)
          STAB             (1 if move type matches our active Pokémon's type)
        = 12 features
        """
        return 12

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Per-move features for the 4 move slots, in the same order that actions 6-9
        use (active_pokemon.moves). Unusable moves are left as all zeros.
        """
        obs = np.zeros(self._observation_size(), dtype=np.float32)

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if active is None or opp is None:
            return obs

        available_ids = {m.id for m in battle.available_moves}

        for i, move in enumerate(list(active.moves.values())[:4]):
            if move.id not in available_ids:
                continue  # disabled / out of PP / forced switch -> all zeros
            obs[3 * i] = min(move.base_power * move.accuracy / 150.0, 1.0)
            obs[3 * i + 1] = opp.damage_multiplier(move) / 4.0
            obs[3 * i + 2] = 1.0 if move.type in active.types else 0.0

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
