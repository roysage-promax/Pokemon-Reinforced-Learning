"""
Plays a trained DQN greedily against the max expert and, at every step it chooses, compares its
action with what the hand-written policy in check_env.py (probe variant E with variant I's heal,
95.7% vs max) would do. The disagreements, grouped by the hand policy's reason, show which rule
the agent hasn't learned - where a reward bonus would help most. Needs a checkpoint trained on
this branch's environment (v5: 78 features, 16 actions).

    python scripts/audit_policy.py ~/cares_rl_logs/DQN/DQN-random-max-v5-s10-*/10/models/final/DQN_checkpoint.pth
    python scripts/audit_policy.py <checkpoint> --battles 500
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

from showdown_gym.showdown_environment import HEAL, N_MOVES, SingleShowdownWrapper  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--battles", type=int, default=200)
    args = parser.parse_args()

    net = load_network(args.checkpoint)
    env = SingleShowdownWrapper(team_type="random", opponent_type="max", evaluation=True)
    showdown = env.env
    time.sleep(3)

    wins = steps = 0
    situations = Counter()  # hand policy's choice -> steps
    agree = Counter()
    instead = defaultdict(Counter)  # hand's choice -> what the agent did instead
    route = Counter()  # (heal / setup, through action 14-15 or a move slot) -> steps
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
                if same(obs, action, hand):
                    agree[situation] += 1
                else:
                    did = kind(obs, action) if not obs[FORCED] else "other slot or illegal"
                    instead[situation][did if did != kind(obs, hand) else "same kind, other slot"] += 1
            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            steps += 1
            done = terminated or truncated
        wins += bool(showdown.battle1.won)

    n = args.battles
    lo, hi = wilson(wins, n)
    print(f"{n} battles in {time.time() - start:.0f}s: agent won {wins} = {100 * wins / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"\n{'hand policy would':16s} {'per battle':>10s} {'agent agrees':>12s}  agent instead (per battle)")
    for situation, count in situations.most_common():
        others = ", ".join(f"{k} {v / n:.2f}" for k, v in instead[situation].most_common())
        print(f"{situation:16s} {count / n:10.2f} {100 * agree[situation] / count:11.0f}%  {others}")
    print("\nagent's heals and setups per battle: " + ", ".join(f"{k} via {r} {v / n:.2f}" for (k, r), v in sorted(route.items())))

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
