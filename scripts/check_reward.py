"""
Checks the v7 reward variants without any training. Plays battles against the max expert with
scripted policies and, at every step, works out the reward each variant in train_parallel.sh's
VARIANTS would give for that same step (each variant is the environment module imported again
with its settings, run on a copy of the environment's bookkeeping):

  1. ctrl gives exactly v6's reward (v6's showdown_environment.py from git, same settings) and
     the reward the environment itself returned
  2. the hand-written policy (scripts/check_env.py, through actions 14-19) pays no v7 costs, so
     the costs only fall on plays it wouldn't make
  3. switching out of a Pokémon the predicted move hits x0.5 or less and straight back in
     (A -> B -> A) earns nothing net under the variants with the switch fix, while under ctrl it
     sometimes earns the switch bonus
  4. per-battle reward, bonus and cost totals per variant, to see their sizes

Policies: hand (the hand-written policy), bouncer (the same, but half the time it switches out of
a resisted or immune spot to the teammate the predicted move hits hardest, which the hand policy
then switches back out of - the v6-1m agents' habit) and random actions.

    python scripts/check_reward.py                 # 100 battles per policy
    python scripts/check_reward.py --battles 300
Needs the Showdown server running (node pokemon-showdown start --no-security).
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_env import FORCED, MOVE, PREDICTED, TEAM, policy, threat  # noqa: E402

from showdown_gym.showdown_environment import N_ACTIONS, N_MOVES, N_SWITCHES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(REPO, "showdown_gym", "showdown_environment.py")


def sweep_variants() -> dict[str, dict[str, str]]:
    """VARIANTS from train_parallel.sh (name -> SHOWDOWN_* settings), expanded by bash itself."""
    script = os.path.join(REPO, "scripts", "train_parallel.sh")
    out = subprocess.run(
        ["bash", "-c", 'eval "$(sed -n "/^DQN3=/,/^)/p" "$1")"; printf "%s\\n" "${VARIANTS[@]}"', "_", script],
        capture_output=True, text=True, check=True,
    ).stdout
    variants = {}
    for line in out.splitlines():
        name, settings = line.split("|", 1)
        variants[name] = dict(pair.split("=", 1) for pair in settings.split())
    return variants


def load_module(name: str, path: str, settings: dict[str, str]):
    """The environment module at `path`, imported fresh with only these SHOWDOWN_* settings set."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("SHOWDOWN_")}
    for k in saved:
        del os.environ[k]
    os.environ.update(settings)
    try:
        # Named inside the package so its relative import (.base_environment) works
        spec = importlib.util.spec_from_file_location(f"showdown_gym.check_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for k in settings:
            del os.environ[k]
        os.environ.update(saved)
    return module


def shadow(module):
    """An environment object of that module's class with just the bookkeeping the reward reads."""
    env = object.__new__(module.ShowdownEnvironment)
    env._counts = env._new_counts()
    env._bonus_paid = set()
    env._switched_out = (None, 0)
    env._bonus = 0.0
    env._illegal_action = False
    return env


def bouncer(obs: np.ndarray, rng) -> int:
    legal_switches = [j for j in range(N_SWITCHES) if obs[TEAM + 5 * j]]
    damaging = any(obs[MOVE + 7 * s] and obs[MOVE + 7 * s + 1] > 0 for s in range(N_MOVES))
    known = obs[PREDICTED] or obs[PREDICTED + 1]
    resisted = 4 * obs[PREDICTED + 3] <= 0.5
    if not obs[FORCED] and known and resisted and legal_switches and damaging and rng.random() < 0.5:
        return N_MOVES + max(legal_switches, key=lambda j: threat(obs, j))
    return policy(obs, tera=True, macros=True)


def play(env, n_battles, choose, label, modules):
    showdown = env.env
    rng = np.random.default_rng(0)
    reward = defaultdict(float)
    bonus = defaultdict(float)
    cost = defaultdict(float)
    cycles = defaultdict(list)  # variant -> net shaping of each switch out of a resisted spot and back
    counts = defaultdict(lambda: defaultdict(int))
    mismatch = {"v6": 0.0, "env": 0.0}
    wins = 0
    start = time.time()

    for _ in range(n_battles):
        obs, _ = env.reset()
        shadows = {name: shadow(module) for name, module in modules.items()}
        last_switch = None  # (turn, species switched out, shaping per variant, out of a resisted spot)
        done = False
        while not done:
            battle = showdown.battle1
            action = np.int64(choose(obs, rng))
            active = battle.active_pokemon
            switches_before = shadows["ctrl"]._counts["switches"]
            resisted_before = shadows["ctrl"]._counts["resist_switch_out"]
            for s in shadows.values():
                s.battle1, s.agent1_to_move = battle, showdown.agent1_to_move
                s.process_action(action)
            shaping = {name: s._bonus for name, s in shadows.items()}
            for name, value in shaping.items():
                (bonus if value > 0 else cost)[name] += value

            if shadows["ctrl"]._counts["switches"] > switches_before:  # a legal voluntary switch
                incoming = switch_target(showdown, battle, action)
                if last_switch and last_switch[3] and battle.turn - last_switch[0] <= 2 and incoming == last_switch[1]:
                    for name in shadows:
                        cycles[name].append(last_switch[2][name] + shaping[name])
                resisted = shadows["ctrl"]._counts["resist_switch_out"] > resisted_before
                last_switch = (battle.turn, active.species, shaping, resisted)

            obs, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_rewards = {}
            for name, s in shadows.items():
                s.battle1, s._prev_battle_state = showdown.battle1, showdown._prev_battle_state
                step_rewards[name] = s.calc_reward(showdown.battle1)
                reward[name] += step_rewards[name]
            mismatch["v6"] = max(mismatch["v6"], abs(step_rewards["v6"] - step_rewards["ctrl"]))
            mismatch["env"] = max(mismatch["env"], abs(env_reward - step_rewards["ctrl"]))

        wins += bool(showdown.battle1.won)
        for name, s in shadows.items():
            for k, v in s._counts.items():
                counts[name][k] += v

    n = n_battles
    print(f"\n== {label}: {n} battles in {time.time() - start:.0f}s, won {wins} ({100 * wins / n:.1f}%)")
    print(f"largest per-step difference: ctrl vs v6 {mismatch['v6']:.2g}, ctrl vs the environment's own reward {mismatch['env']:.2g}")
    c = counts["ctrl"]
    print("per battle: " + ", ".join(f"{k} {c[k] / n:.2f}" for k in
          ("switches", "switch_back", "resist_switch_out", "other_switch", "good_switch", "heal_moves", "high_heal", "good_heal", "good_setup", "good_tera")))
    print(f"{'variant':10s} {'reward':>7s} {'bonus':>6s} {'costs':>6s}   out of a resisted spot and back: {'count':>5s} {'mean net':>8s} {'max net':>7s}")
    for name in modules:
        nets = cycles[name]
        cyc = f"{len(nets):5d} {np.mean(nets):8.3f} {max(nets):7.3f}" if nets else f"{0:5d} {'-':>8s} {'-':>7s}"
        print(f"{name:10s} {reward[name] / n:7.3f} {bonus[name] / n:6.3f} {cost[name] / n:6.3f}   {'':31s} {cyc}")


def switch_target(showdown, battle, action) -> str | None:
    """Species of the teammate a switch action (4-9, or 18/19 resolved) brings in."""
    a = int(action)
    if a >= N_MOVES + N_SWITCHES:
        a = showdown._macro(battle, a)[0]
    if not N_MOVES <= a < N_MOVES + N_SWITCHES:
        return None
    return list(battle.team.values())[a - N_MOVES].species


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", type=int, default=100, help="battles per policy")
    parser.add_argument("--v6_ref", default="5e3d59c", help="git commit with v6's showdown_environment.py")
    args = parser.parse_args()

    variants = sweep_variants()
    v6_source = subprocess.run(["git", "-C", REPO, "show", f"{args.v6_ref}:showdown_gym/showdown_environment.py"],
                               capture_output=True, check=True).stdout
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(v6_source)
    modules = {"v6": load_module("v6", f.name, variants["ctrl"])}
    modules.update({name: load_module(name, ENV_FILE, settings) for name, settings in variants.items()})
    os.unlink(f.name)

    # The environment itself runs with ctrl's settings, so its reward should equal ctrl's
    os.environ.update(variants["ctrl"])
    importlib.reload(sys.modules["showdown_gym.showdown_environment"])
    from showdown_gym.showdown_environment import SingleShowdownWrapper as Wrapper

    env = Wrapper(team_type="random", opponent_type="max")
    time.sleep(3)  # cares-rl waits like this before using a new environment

    play(env, args.battles, lambda obs, rng: policy(obs, tera=True, macros=True), "hand-written policy", modules)
    play(env, args.battles, bouncer, "bouncer (hand policy + switching out of resisted spots)", modules)
    play(env, args.battles, lambda obs, rng: int(rng.integers(N_ACTIONS)), "random actions", modules)

    # poke_env's background threads can keep a clean exit waiting on the server
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
