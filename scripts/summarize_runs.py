"""
Compare training runs by how they play once exploration is over. After epsilon reaches its
floor (decay_steps), every training battle is effectively a greedy test battle against the max
expert, so the logs hold thousands of test-like battles per run - far tighter than a 100-battle
test, which swings about +-7 points by luck.

For each run: win rate after exploration (with a 95% interval), win rate in the last 20k steps
and the 20k before that (still rising? then more steps may help), and per-battle habits from the
environment's counters (switches, illegal actions, setup and healing moves).

    python scripts/summarize_runs.py                      # ~/cares_rl_logs, v4r runs
    python scripts/summarize_runs.py runs --prefix v3     # the runs committed to the repo
"""

import argparse
import csv
import glob
import json
import math
import os
import re

HABITS = ["switches", "forced_switches", "illegal_actions", "illegal_tera", "setup_moves", "heal_moves", "tera", "good_tera"]
DATE = r"-\d{2}_\d{2}_\d{2}_\d{2}-\d{2}-\d{2}$"


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def win_rate(rows):
    return 100 * sum(r["win"] == "True" for r in rows) / len(rows) if rows else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", nargs="?", default=os.environ.get("CARES_LOG_BASE_DIR", "~/cares_rl_logs"))
    parser.add_argument("--prefix", default="v4r", help="environment version in the run names")
    args = parser.parse_args()
    log_dir = os.path.expanduser(args.log_dir)

    # ~/cares_rl_logs has a folder per algorithm; the repo's runs/ folder is flat
    pattern = f"*-random-max-{args.prefix}-*"
    dirs = sorted(set(glob.glob(f"{log_dir}/*/{pattern}") + glob.glob(f"{log_dir}/{pattern}")))
    train_dirs = [d for d in dirs if "-random-max-test-" not in d]
    if not train_dirs:
        print(f"no runs matching {pattern} under {log_dir}")
        return

    header = f"{'run':16s} {'steps':>7s} {'after exploration':>27s} {'prev 20k':>8s} {'last 20k':>8s}  per battle"
    print(header)
    for d in train_dirs:
        name = re.sub(DATE, "", re.sub(r"^[A-Za-z0-9]+-random-max-", "", os.path.basename(d)))
        seed_dirs = [s for s in glob.glob(f"{d}/*") if os.path.basename(s).isdigit()]
        train_csv = f"{seed_dirs[0]}/data/train.csv" if seed_dirs else ""
        if not os.path.exists(train_csv):
            print(f"{name:16s} no train.csv yet")
            continue
        with open(train_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"{name:16s} no episodes yet")
            continue
        with open(f"{d}/alg_config.json") as f:
            decay = json.load(f).get("decay_steps", 100000)

        steps = int(rows[-1]["total_steps"])
        greedy = [r for r in rows if int(r["total_steps"]) > decay]
        prev = [r for r in rows if steps - 40000 < int(r["total_steps"]) <= steps - 20000]
        last = [r for r in rows if int(r["total_steps"]) > steps - 20000]

        if greedy:
            w = sum(r["win"] == "True" for r in greedy)
            lo, hi = wilson(w, len(greedy))
            after = f"{100 * w / len(greedy):5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}] n={len(greedy):<5d}"
        else:
            after = f"(exploring until {decay // 1000}k)"
        window = greedy or last
        habits = ", ".join(
            f"{h} {sum(float(r[h] or 0) for r in window) / len(window):.2f}" for h in HABITS if h in rows[0]
        )
        print(f"{name:16s} {steps:7d} {after:>27s} {win_rate(prev):7.1f}% {win_rate(last):7.1f}%  {habits}")

    test_pattern = f"*-random-max-test-{args.prefix}-*"
    tests = sorted(set(glob.glob(f"{log_dir}/*/{test_pattern}") + glob.glob(f"{log_dir}/{test_pattern}")))
    if tests:
        print("\ntests (greedy; a 100-battle test swings +-7 points by luck):")
    for d in tests:
        for csv_path in glob.glob(f"{d}/*/data/eval.csv"):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                name = re.sub(DATE, "", re.sub(r"^[A-Za-z0-9]+-random-max-", "", os.path.basename(d)))
                w = sum(r["win"] == "True" for r in rows)
                lo, hi = wilson(w, len(rows))
                print(f"  {name:20s} model {rows[0]['total_steps']:>7s}: {w}/{len(rows)} = {100 * w / len(rows):.1f}% [{100 * lo:.1f}-{100 * hi:.1f}]")


if __name__ == "__main__":
    main()
