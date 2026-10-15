"""
Turns a v4 DQN checkpoint (76 features, 14 actions) into the starting weights of a v5 run (78
features, 16 actions) for `cares-rl transfer`. The two new inputs (usable healing / setup move)
start with zero weights, so the network plays exactly as before until it learns to use them; the
outputs for actions 14 and 15 start as the average of the move-slot outputs 0-3.

    python scripts/expand_checkpoint.py runs/DQN-random-max-v4s-rules-s10-*/10/models/final/DQN_checkpoint.pth out_dir
    cares-rl transfer --transfer_path out_dir --run-name ... cli ...
scripts/train_parallel.sh does both for runs that name a checkpoint to start from.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_env import KINDS  # noqa: E402

from showdown_gym.showdown_environment import HEAL, N_MOVES, SETUP  # noqa: E402

OLD_FEATURES = KINDS  # v4's observation ended where v5's new features begin
NEW_FEATURES = KINDS + 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a v4 DQN_checkpoint.pth")
    parser.add_argument("out_dir", help="folder to write DQN_checkpoint.pth to")
    args = parser.parse_args()

    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["network"]
    # cares-rl's DQN network: Linear-ReLU-Linear-ReLU-Linear, weights under network.model.<index>
    layers = sorted({int(key.split(".")[2]) for key in weights})
    first, last = f"network.model.{layers[0]}", f"network.model.{layers[-1]}"

    w = weights[f"{first}.weight"]
    if w.shape[1] != OLD_FEATURES:
        sys.exit(f"expected a v4 network with {OLD_FEATURES} inputs, got {w.shape[1]}")
    weights[f"{first}.weight"] = torch.cat([w, w.new_zeros(w.shape[0], NEW_FEATURES - OLD_FEATURES)], dim=1)

    w, b = weights[f"{last}.weight"], weights[f"{last}.bias"]
    if w.shape[0] != HEAL:
        sys.exit(f"expected a v4 network with {HEAL} outputs, got {w.shape[0]}")
    new = SETUP + 1 - HEAL
    weights[f"{last}.weight"] = torch.cat([w, w[:N_MOVES].mean(0, keepdim=True).repeat(new, 1)])
    weights[f"{last}.bias"] = torch.cat([b, b[:N_MOVES].mean().repeat(new)])

    os.makedirs(args.out_dir, exist_ok=True)
    # A transfer only loads "network" (cares-rl's DQN.load_models)
    torch.save({"network": weights}, os.path.join(args.out_dir, "DQN_checkpoint.pth"))
    print(f"{args.checkpoint} -> {args.out_dir}/DQN_checkpoint.pth")


if __name__ == "__main__":
    main()
