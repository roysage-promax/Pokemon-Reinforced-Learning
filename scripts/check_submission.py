"""
Checks that showdown_environment.py works as the assignment submission, where it is the only file
of ours that is used: copied into a clean copy of the staff's showdown_gym package (STAFF_COMMIT),
with nothing else from this repo, it must import using only the standard library and the packages
in requirements.txt, and play battles against the max expert. Given a checkpoint, the trained
network's input and output sizes must match the observation and action sizes, and it plays the
battles greedily.

    python scripts/check_submission.py
    python scripts/check_submission.py ~/cares_rl_logs/DQN/DQN-random-max-v6-s10-*/10/models/final/DQN_checkpoint.pth
Needs the Showdown server running. Exits 1 if anything fails.
"""

import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile

STAFF_COMMIT = "8d0a390"  # the last commit of the assignment repo before our changes
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENV_FILE = os.path.join(REPO, "showdown_gym", "showdown_environment.py")

# Runs inside the clean copy: PYTHONPATH puts it ahead of the installed (this repo's) package
PLAY = r"""
import os, sys, time
import numpy as np
import showdown_gym.showdown_environment as m

root, checkpoint, battles = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert os.path.realpath(m.__file__).startswith(os.path.realpath(root)), f"imported {m.__file__}, not the clean copy"
env = m.SingleShowdownWrapper(team_type="random", opponent_type="max", evaluation=True)
showdown = env.env
obs_size, n_actions = showdown._observation_size(), showdown._get_action_size()
print(f"observation size {obs_size}, actions {n_actions}")

net = None
if checkpoint:
    import torch
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["network"]
    layers = sorted({int(k.split(".")[2]) for k in weights})
    n_in = weights[f"network.model.{layers[0]}.weight"].shape[1]
    n_out = weights[f"network.model.{layers[-1]}.weight"].shape[0]
    print(f"network: {n_in} inputs, {n_out} outputs")
    assert (n_in, n_out) == (obs_size, n_actions), "the network doesn't fit this environment"
    modules = []
    for i in layers:
        out_features, in_features = weights[f"network.model.{i}.weight"].shape
        modules += [torch.nn.Linear(in_features, out_features), torch.nn.ReLU()]
    net = torch.nn.Sequential(*modules[:-1])
    net.load_state_dict({k.replace("network.model.", ""): v for k, v in weights.items()})
    net.eval()

time.sleep(3)
rng = np.random.default_rng(0)
wins = 0
for _ in range(battles):
    obs, _ = env.reset()
    done = False
    while not done:
        assert obs.shape == (obs_size,) and np.all(np.isfinite(obs)), obs
        if net is not None:
            with torch.no_grad():
                action = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
        else:
            action = int(rng.integers(n_actions))
        obs, _, terminated, truncated, _ = env.step(np.int64(action))
        done = terminated or truncated
    wins += bool(showdown.battle1.won)
print(f"{battles} battles, {wins} won ({'greedy network' if net is not None else 'random actions'})")
sys.stdout.flush()
os._exit(0)
"""


def disallowed_imports(env_file: str) -> list[str]:
    """Top-level modules the environment imports that are neither standard library nor in requirements.txt."""
    with open(os.path.join(REPO, "requirements.txt"), encoding="utf-8") as f:
        required = {re.split(r"[<>=!~\[ ;]", line.strip())[0].lower().replace("-", "_") for line in f if line.strip()}
    with open(env_file, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add(node.module.split(".")[0])
    return sorted(m for m in modules if m not in sys.stdlib_module_names and m.lower() not in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default="")
    parser.add_argument("--battles", type=int, default=3)
    parser.add_argument("--env_file", default=ENV_FILE, help="the environment file to check (default: this repo's)")
    args = parser.parse_args()

    extra = disallowed_imports(args.env_file)
    if extra:
        print(f"FAIL: showdown_environment.py imports {extra}, which requirements.txt doesn't list")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as root:
        archive = subprocess.run(["git", "-C", REPO, "archive", STAFF_COMMIT, "showdown_gym"], capture_output=True, check=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(root, filter="data")
        with open(args.env_file, "rb") as src, open(os.path.join(root, "showdown_gym", "showdown_environment.py"), "wb") as dst:
            dst.write(src.read())

        env = dict(os.environ, PYTHONPATH=root)
        command = [sys.executable, "-c", PLAY, root, os.path.abspath(args.checkpoint) if args.checkpoint else "", str(args.battles)]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=600)
    lines = [line for line in (result.stdout + result.stderr).splitlines() if "warn" not in line.lower()]
    print("\n".join(lines[-15:]))
    if result.returncode != 0 or not any(" battles, " in line for line in lines):
        print("FAIL: the environment doesn't work on its own in a clean copy of the package")
        sys.exit(1)
    print("PASS: showdown_environment.py works on its own" + (" with this network" if args.checkpoint else ""))


if __name__ == "__main__":
    main()
