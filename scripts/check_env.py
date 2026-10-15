"""
Checks the environment against the max expert without any training:

  1. random actions: battles finish, observations stay in range, and the observation's "legal"
     flags agree with the legality check the illegal-action penalty uses
  2. hand-written policies that read only the observation vector, the probe's variant C rules
     and C + tera (variant E), healing on neutral hits too (variant I) and healing / setting up
     through actions 14 / 15: their win rates show whether the observation and actions carry
     what an agent needs to play that well

    python scripts/check_env.py                  # 30 random battles, 300 battles per policy
    python scripts/check_env.py --random 10 --policy 1000
Needs the Showdown server running (node pokemon-showdown start --no-security).
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np

from showdown_gym.showdown_environment import HEAL, N_MOVES, N_SWITCHES, SETUP, SingleShowdownWrapper

# Observation layout (see ShowdownEnvironment._observation_size)
MOVE = 0  # 4 x [legal, relative damage, power, effectiveness, setup, heal, relative damage with tera]
TEAM = 7 * N_MOVES  # 6 x [legal, hp, predicted-move threat / 4, offence / 300, faster]
CONTEXT = TEAM + 5 * N_SWITCHES  # [our hp, opp hp, ours left, theirs left, faster, type threat]
PREDICTED = CONTEXT + 6  # [revealed, guessed, power, threat on active / 4, stab, threat after tera / 4]
BOOSTS = PREDICTED + 6  # [atk, spa] / 6
TERA = BOOSTS + 2  # [can tera now, active is terastallized, tera unused]
FORCED = TERA + 3
KINDS = FORCED + 1  # [usable healing move, usable setup move]
TERA_ACTION = N_MOVES + N_SWITCHES


def resolve(obs: np.ndarray, action: int) -> int:
    """The move-slot action HEAL / SETUP stands for, picked from the observation as the environment's _kind_slot does."""
    if action < HEAL:
        return action
    col = 5 if action == HEAL else 4
    legal = [s for s in range(N_MOVES) if obs[MOVE + 7 * s]]
    for s in legal:
        if obs[MOVE + 7 * s + col]:
            return s
    return max(legal, key=lambda s: obs[MOVE + 7 * s + 1]) if legal else 0


def policy(obs: np.ndarray, tera: bool) -> int:
    """Probe variant C, or with tera variant E, using nothing but the observation."""
    legal_moves = [s for s in range(N_MOVES) if obs[MOVE + 7 * s]]
    legal_switches = [j for j in range(N_SWITCHES) if obs[TEAM + 5 * j]]

    def threat(j):
        return 4 * obs[TEAM + 5 * j + 2]

    def matchup(j):
        return 2 * obs[TEAM + 5 * j + 3] - threat(j)

    if obs[FORCED] or not legal_moves:
        return N_MOVES + max(legal_switches, key=matchup) if legal_switches else 0

    best_move = max(legal_moves, key=lambda s: obs[MOVE + 7 * s + 1])
    if not (obs[PREDICTED] or obs[PREDICTED + 1]):
        return best_move

    my_threat = 4 * obs[PREDICTED + 3]
    if legal_switches:
        j = min(legal_switches, key=lambda j: (threat(j), -matchup(j)))
        if my_threat >= 2 and threat(j) <= 0.5:
            return N_MOVES + j
        if my_threat >= 1 and threat(j) == 0:
            return N_MOVES + j

    # Tera when it turns the predicted move into a resisted one, or adds STAB to the best move
    tera_offset = 0
    if tera and obs[TERA]:
        tera_threat = 4 * obs[PREDICTED + 5]
        resist = my_threat >= 1 and tera_threat <= 0.5
        stab = obs[MOVE + 7 * best_move + 6] > obs[MOVE + 7 * best_move + 1] and tera_threat <= my_threat
        if resist or stab:
            tera_offset = TERA_ACTION

    hp = obs[CONTEXT]
    if my_threat <= 1 and hp <= 0.5 and obs[KINDS]:
        return HEAL
    if my_threat <= 0.5 and hp >= 0.5 and obs[KINDS + 1]:
        cap = 6 if my_threat == 0 else 2
        if 6 * max(obs[BOOSTS], obs[BOOSTS + 1]) < cap:
            return SETUP
    return best_move + tera_offset


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def play(env, n_battles, choose, label):
    showdown = env.env
    rng = np.random.default_rng(0)
    wins = steps = flag_mismatches = kind_mismatches = 0
    low, high = np.inf, -np.inf
    predictions = Counter()
    totals = Counter()
    start = time.time()

    for _ in range(n_battles):
        obs, _ = env.reset()
        done = False
        while not done:
            assert obs.shape == (showdown._observation_size(),), obs.shape
            assert np.all(np.isfinite(obs)), obs
            low, high = min(low, obs.min()), max(high, obs.max())

            battle = showdown.battle1
            for a in range(SETUP + 1):
                if a < N_MOVES:
                    flag = obs[MOVE + 7 * a]
                elif a < TERA_ACTION:
                    flag = obs[TEAM + 5 * (a - N_MOVES)]
                elif a < HEAL:
                    flag = obs[MOVE + 7 * (a - TERA_ACTION)] and obs[TERA]
                else:
                    flag = any(obs[MOVE + 7 * s] for s in range(N_MOVES))
                    kind_mismatches += bool(flag) and resolve(obs, a) != showdown._kind_slot(battle, a)[0]
                flag_mismatches += bool(flag) != showdown._is_legal(battle, a)
            predictions["revealed" if obs[PREDICTED] else "guessed" if obs[PREDICTED + 1] else "none"] += 1

            obs, _, terminated, truncated, info = env.step(np.int64(choose(obs, rng)))
            steps += 1
            done = terminated or truncated

        battle = showdown.battle1
        wins += bool(battle.won)
        totals.update(showdown._counts)

    n = n_battles
    lo, hi = wilson(wins, n)
    print(f"\n== {label}: {n} battles in {time.time() - start:.0f}s, {steps / n:.1f} steps each")
    print(f"win rate {wins}/{n} = {100 * wins / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    print("per battle: " + ", ".join(f"{k} {v / n:.2f}" for k, v in totals.items()))
    print(f"observation range [{low:.2f}, {high:.2f}]; legal-flag mismatches: {flag_mismatches}; "
          f"heal/setup slot mismatches: {kind_mismatches}")
    print("opponent's next move: " + ", ".join(f"{k} {100 * v / steps:.0f}%" for k, v in predictions.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--random", type=int, default=30, help="battles with random actions")
    parser.add_argument("--policy", type=int, default=300, help="battles with each observation-only policy")
    args = parser.parse_args()

    env = SingleShowdownWrapper(team_type="random", opponent_type="max")
    time.sleep(3)  # cares-rl waits like this before using a new environment

    if args.random:
        play(env, args.random, lambda obs, rng: int(rng.integers(SETUP + 1)), "random actions")
    if args.policy:
        play(env, args.policy, lambda obs, rng: policy(obs, tera=False), "observation-only policy (probe variant C + I's heal)")
        play(env, args.policy, lambda obs, rng: policy(obs, tera=True), "observation-only policy + tera (variant E + I's heal)")

    # poke_env's background threads can keep a clean exit waiting on the server
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
