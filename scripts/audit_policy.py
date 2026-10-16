"""
Plays a trained DQN greedily against the max expert and, at every step it chooses, compares its
action with what the hand-written policy in check_env.py (probe variant E with variant I's heal,
95.7% vs max) would do. The disagreements, grouped by the hand policy's reason, show which rule
the agent hasn't learned - where a reward bonus would help most. Needs a checkpoint trained on
this branch's environment (v6: 84 features, 20 actions).

With --hand_plays, the hand-written policy takes over one kind of decision: every step where it or
the agent would make that kind of play. The win rate gained over --hand_plays none is what the
agent's mistakes of that kind cost; "all" is the hand-written policy on its own.

    python scripts/audit_policy.py ~/cares_rl_logs/DQN/DQN-random-max-v6-s10-*/10/models/final/DQN_checkpoint.pth
    python scripts/audit_policy.py <checkpoint> --battles 500
    python scripts/audit_policy.py <checkpoint> --battles 3000 --hand_plays switch
Needs the Showdown server running (node pokemon-showdown start --no-security).
"""

import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_env import FORCED, MOVE, TERA, TERA_ACTION, policy, resolve, wilson  # noqa: E402

from showdown_gym.showdown_environment import HEAL, MACRO_NAMES, N_MOVES, SingleShowdownWrapper  # noqa: E402

ACTION_GROUPS = ["move slot"] * N_MOVES + ["switch slot"] * (TERA_ACTION - N_MOVES) + ["tera slot"] * N_MOVES
# --hand_plays: forced = replacements for a fainted Pokemon; switch, heal, setup, tera = whenever the
# agent or the hand policy would make that play; attack = both attack without tera but pick
# different moves
TAKEOVERS = ["none", "all", "forced", "switch", "heal", "setup", "tera", "attack"]


def load_network(path: str) -> torch.nn.Sequential:
    """cares-rl's DQN network: Linear-ReLU-Linear-ReLU-Linear, weights under network.model.*"""
    weights = torch.load(path, map_location="cpu", weights_only=False)["network"]
    layers = []
    for i in range(0, 1 + max(int(k.split(".")[2]) for k in weights), 2):
        out_features, in_features = weights[f"network.model.{i}.weight"].shape
        layers += [torch.nn.Linear(in_features, out_features), torch.nn.ReLU()]
    net = torch.nn.Sequential(*layers[:-1])
    net.load_state_dict({k.replace("network.model.", ""): v for k, v in weights.items()})
    return net.eval()


def kind(obs: np.ndarray, action: int) -> str:
    """What an action does: switch, heal, setup or attack, with '+tera' when it terastallizes."""
    action = resolve(obs, action)
    if N_MOVES <= action < TERA_ACTION:
        return "switch"
    tera = action >= TERA_ACTION and obs[TERA]
    slot = action % TERA_ACTION if action >= TERA_ACTION else action
    what = "heal" if obs[MOVE + 7 * slot + 5] else "setup" if obs[MOVE + 7 * slot + 4] else "attack"
    return what + ("+tera" if tera else "")


def same(obs: np.ndarray, a: int, b: int) -> bool:
    """Tera actions just use the move when tera isn't available; moves of equal value count as equal."""
    a, b = resolve(obs, a), resolve(obs, b)
    if not obs[TERA]:
        a = a - TERA_ACTION if a >= TERA_ACTION else a
        b = b - TERA_ACTION if b >= TERA_ACTION else b
    if a == b:
        return True
    if N_MOVES <= a < TERA_ACTION or N_MOVES <= b < TERA_ACTION or (a >= TERA_ACTION) != (b >= TERA_ACTION):
        return False
    sa, sb = a % TERA_ACTION, b % TERA_ACTION
    col = 6 if a >= TERA_ACTION else 1
    return kind(obs, a) == kind(obs, b) and math.isclose(obs[MOVE + 7 * sa + col], obs[MOVE + 7 * sb + col])


def takes_over(what: str, obs: np.ndarray, agent: int, hand: int) -> bool:
    """Whether the hand policy plays this step instead of the agent (see TAKEOVERS)."""
    if what in ("none", "all"):
        return what == "all"
    if obs[FORCED]:
        return what == "forced"
    kinds = (kind(obs, agent), kind(obs, hand))
    if what == "tera":
        return any(k.endswith("+tera") for k in kinds)
    if what == "attack":
        return kinds == ("attack", "attack")
    return what in (k.removesuffix("+tera") for k in kinds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--hand_plays", choices=TAKEOVERS, default="none", help="decisions the hand policy makes instead")
    args = parser.parse_args()

    net = load_network(args.checkpoint)
    env = SingleShowdownWrapper(team_type="random", opponent_type="max", evaluation=True)
    showdown = env.env
    time.sleep(3)

    wins = steps = taken = changed = 0
    situations = Counter()  # hand policy's choice -> steps
    agree = Counter()
    instead = defaultdict(Counter)  # hand's choice -> what the agent did instead
    route = Counter()  # (heal / setup, through action 14-15 or a move slot) -> steps
    used = Counter()  # the agent's action, by group -> steps
    start = time.time()
    for _ in range(args.battles):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
            if showdown.battle1 is not None and showdown.agent1_to_move:
                hand = policy(obs, tera=True)
                situation = "forced switch" if obs[FORCED] else kind(obs, hand)
                did = kind(obs, action).replace("+tera", "")
                if not obs[FORCED] and did in ("heal", "setup"):
                    route[did, "action 14-15" if action >= HEAL else "move slot"] += 1
                situations[situation] += 1
                used[ACTION_GROUPS[action] if action < HEAL else MACRO_NAMES[action - HEAL]] += 1
                if same(obs, action, hand):
                    agree[situation] += 1
                else:
                    did = kind(obs, action) if not obs[FORCED] else "other slot or illegal"
                    instead[situation][did if did != kind(obs, hand) else "same kind, other slot"] += 1
                if takes_over(args.hand_plays, obs, action, hand):
                    taken += 1
                    changed += not same(obs, action, hand)
                    action = hand
            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            steps += 1
            done = terminated or truncated
        wins += bool(showdown.battle1.won)

    n = args.battles
    lo, hi = wilson(wins, n)
    print(f"{n} battles in {time.time() - start:.0f}s: agent won {wins} = {100 * wins / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    if args.hand_plays != "none":
        print(f"hand policy plays: {args.hand_plays} - {taken / n:.2f} steps a battle, "
              f"{changed / n:.2f} of them different from the agent's choice")
    print(f"\n{'hand policy would':16s} {'per battle':>10s} {'agent agrees':>12s}  agent instead (per battle)")
    for situation, count in situations.most_common():
        others = ", ".join(f"{k} {v / n:.2f}" for k, v in instead[situation].most_common())
        print(f"{situation:16s} {count / n:10.2f} {100 * agree[situation] / count:11.0f}%  {others}")
    print("\nagent's actions per battle: " + ", ".join(f"{k} {v / n:.2f}" for k, v in used.most_common()))
    print("agent's heals and setups per battle: " + ", ".join(f"{k} via {r} {v / n:.2f}" for (k, r), v in sorted(route.items())))

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
