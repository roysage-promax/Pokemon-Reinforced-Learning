"""
Set up a finished run to train on: a folder for `cares-rl resume --data_path DEST` that carries on
from SRC's last checkpoint, replay buffer and logs up to STEPS steps in all, optionally with a new
learning rate. SRC itself is only read, so several continuations can start from one run.

    python scripts/continue_run.py SRC DEST STEPS [LR]

cares-rl resume takes every setting from the folder's config files, so DEST gets SRC's with
max_steps_training (and lr) changed. The optimiser's state comes from the checkpoint, and loading it
brings back the learning rate it was saved with, so LR is also written into the checkpoint's
optimiser state. The replay buffer (up to ~1 GB) is linked, not copied.

Exits with 2 when SRC has no checkpoint or replay buffer to continue from.
"""

import glob
import json
import os
import shutil
import sys


def main():
    src, dest, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lr = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None

    seed_dirs = sorted(d for d in glob.glob(f"{src}/*") if os.path.basename(d).isdigit())
    for seed_dir in seed_dirs:
        for need in ("models/checkpoint/*_checkpoint.pth", "memory/memory.pkl", "data/train.csv"):
            if not glob.glob(f"{seed_dir}/{need}"):
                print(f"{seed_dir} has no {need} - nothing to continue from")
                sys.exit(2)
    if not seed_dirs:
        print(f"{src} has no seed folders - nothing to continue from")
        sys.exit(2)

    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    with open(f"{src}/alg_config.json") as f:
        alg = json.load(f)
    old_lr = alg["lr"]
    alg["max_steps_training"] = steps
    if lr is not None:
        alg["lr"] = lr
    with open(f"{dest}/alg_config.json", "w") as f:
        json.dump(alg, f)
    for name in ("env_config.json", "train_config.json"):
        shutil.copy(f"{src}/{name}", dest)

    for seed_dir in seed_dirs:
        out = f"{dest}/{os.path.basename(seed_dir)}"
        os.makedirs(f"{out}/data")
        for csv in glob.glob(f"{seed_dir}/data/*.csv"):
            shutil.copy(csv, f"{out}/data")
        os.makedirs(f"{out}/memory")
        os.symlink(os.path.abspath(f"{seed_dir}/memory/memory.pkl"), f"{out}/memory/memory.pkl")
        os.makedirs(f"{out}/models/checkpoint")
        for checkpoint in glob.glob(f"{seed_dir}/models/checkpoint/*_checkpoint.pth"):
            target = f"{out}/models/checkpoint/{os.path.basename(checkpoint)}"
            if lr is None:
                shutil.copy(checkpoint, target)
                continue
            import torch

            state = torch.load(checkpoint, map_location="cpu")
            for group in state["optimizer"]["param_groups"]:
                group["lr"] = lr
            torch.save(state, target)
        with open(f"{seed_dir}/data/train.csv") as f:
            last = f.readlines()[-1].split(",")[0]
        print(f"{out}: from step {last} to {steps}, lr {old_lr} -> {alg['lr']}")


if __name__ == "__main__":
    main()
