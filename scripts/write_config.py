"""
Write the config files `cares-rl train cli ARGS` would save (alg_config.json, env_config.json,
train_config.json) with a different hidden-layer width, for `cares-rl train config --data_path DIR`.
The network can't be sized from the command line; the run saves these files with its results, so
`cares-rl test` rebuilds the same network.

    python scripts/write_config.py DIR WIDTH --gym showdown --domain random --task max DQN --lr 0.0003 ...
"""

import json
import logging
import os
import sys

from cares_reinforcement_learning.util.rl_parser import RLParser


def main():
    out_dir, width, cli_args = sys.argv[1], int(sys.argv[2]), sys.argv[3:]

    # Parse exactly as `cares-rl train cli ...` does, so defaults and checks match. Its parser warns
    # that "--gym" is left over on every run; that's harmless, so hide warnings.
    logging.disable(logging.WARNING)
    sys.argv = ["cares-rl", "train", "cli", *cli_args]
    configs = RLParser().parse_args()

    os.makedirs(out_dir, exist_ok=True)
    for name in ("alg_config", "env_config", "train_config"):
        data = configs[name].model_dump(exclude_none=False)
        if name == "alg_config":
            # Hidden sizes are the in/out features that are set; None marks the observation and
            # action sizes, which cares-rl fills in from the environment
            for layer in data["network_config"]["layers"]:
                for key in ("in_features", "out_features"):
                    if layer.get(key) is not None:
                        layer[key] = width
        with open(f"{out_dir}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
