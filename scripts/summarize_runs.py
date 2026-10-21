"""
Compare training runs by how they play once exploration is over. After epsilon reaches its
floor (decay_steps), every training battle is effectively a greedy test battle against the max
expert, so the logs hold thousands of test-like battles per run - far tighter than a 100-battle
test, which swings about +-7 points by luck.

For each run: win rate after exploration (with a 95% interval), win rate in the last 20k steps
and the 20k before that (still rising? then more steps may help), and per-battle habits from the
environment's counters (switches, illegal actions, setup and healing moves).

Runs that differ only in their seed (name ending -s10, -s20...) are also pooled into one row per
variant, ranked by win rate over each variant's last 100k steps (or --window), with their tests
pooled too - the table for comparing a sweep such as v7's reward variants.

    python scripts/summarize_runs.py                      # ~/cares_rl_logs, v7 runs
    python scripts/summarize_runs.py runs --prefix v3     # the runs committed to the repo
    python scripts/summarize_runs.py --window 300000 400000
"""

import argparse
import csv
import glob
import gzip
import json
import math
import os
import re

HABITS = ["switches", "forced_switches", "illegal_actions", "illegal_tera", "setup_moves", "heal_moves", "tera", "good_tera", "good_switch", "good_setup", "good_heal", "switch_back", "resist_switch_out", "other_switch", "high_heal", "heal_action", "setup_action", "attack_action", "tera_action", "safe_switch_action", "matchup_switch_action", "action_fallback"]
DATE = r"-\d{2}_\d{2}_\d{2}_\d{2}-\d{2}-\d{2}$"
# Per-battle habits in the variant table: what the v7 reward variants try to change
VARIANT_HABITS = ["switches", "switch_back", "resist_switch_out", "other_switch", "heal_moves", "high_heal", "good_heal", "setup_moves"]


def open_csv(path):
    """A run's CSV log; logs too big for GitHub are stored gzipped (save_runs.sh)"""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", newline="")
    return open(path, newline="")


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
    parser.add_argument("--prefix", default="v7", help="environment version in the run names")
    parser.add_argument("--window", type=int, nargs=2, metavar=("FROM", "TO"),
                        help="steps the variant table pools (default: each variant's last 100k)")
    args = parser.parse_args()
    log_dir = os.path.expanduser(args.log_dir)

    # ~/cares_rl_logs has a folder per algorithm; the repo's runs/ folder is flat
    pattern = f"*-random-max-{args.prefix}-*"
    dirs = sorted(set(glob.glob(f"{log_dir}/*/{pattern}") + glob.glob(f"{log_dir}/{pattern}")))
    train_dirs = [d for d in dirs if "-random-max-test-" not in d]
    if not train_dirs:
        print(f"no runs matching {pattern} under {log_dir}")
        return

    runs, test_wins = {}, {}
    header = f"{'run':16s} {'steps':>7s} {'after exploration':>27s} {'prev 20k':>8s} {'last 20k':>8s}  per battle"
    print(header)
    for d in train_dirs:
        name = re.sub(DATE, "", re.sub(r"^[A-Za-z0-9]+-random-max-", "", os.path.basename(d)))
        seed_dirs = [s for s in glob.glob(f"{d}/*") if os.path.basename(s).isdigit()]
        train_csv = f"{seed_dirs[0]}/data/train.csv" if seed_dirs else ""
        if not (os.path.exists(train_csv) or os.path.exists(train_csv + ".gz")):
            print(f"{name:16s} no train.csv yet")
            continue
        with open_csv(train_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"{name:16s} no episodes yet")
            continue
        runs[name] = rows
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
                test_wins[name.removeprefix("test-")] = (w, len(rows))
                print(f"  {name:20s} model {rows[0]['total_steps']:>7s}: {w}/{len(rows)} = {100 * w / len(rows):.1f}% [{100 * lo:.1f}-{100 * hi:.1f}]")

    variant_table(runs, test_wins, args.window)


def variant_table(runs, tests, window):
    """Runs pooled per variant (the run name without its -sNN), ranked by win rate over a window."""
    groups = {}
    for name in runs:
        groups.setdefault(re.sub(r"-s\d+$", "", name), []).append(name)
    if all(len(names) < 2 for names in groups.values()):
        return
    table = []
    for group, names in groups.items():
        if window:
            start, end = window
        else:
            end = min(int(runs[name][-1]["total_steps"]) for name in names)
            start = max(end - 100000, 0)
        rows = [r for name in names for r in runs[name] if start < int(r["total_steps"]) <= end]
        if rows:
            tested = [tests[name] for name in names if name in tests]
            table.append((group, len(names), start, end, rows, sum(w for w, _ in tested), sum(n for _, n in tested)))
    table.sort(key=lambda t: -win_rate(t[4]))

    print("\nvariants, seeds pooled, best first (overlapping intervals can't be told apart - confirm the top few with new seeds):")
    print(f"{'variant':12s} {'runs':>4s} {'steps':>9s} {'win rate':>26s} {'tests':>15s}  " + " ".join(f"{h[:9]:>9s}" for h in VARIANT_HABITS))
    for group, k, start, end, rows, tw, tn in table:
        w, n = sum(r["win"] == "True" for r in rows), len(rows)
        lo, hi = wilson(w, n)
        rate = f"{100 * w / n:5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}] n={n:<5d}"
        test = f"{100 * tw / tn:5.1f}% n={tn:<5d}" if tn else "-"
        habits = " ".join(
            f"{sum(float(r[h] or 0) for r in rows) / n:9.2f}" if h in rows[0] else f"{'-':>9s}" for h in VARIANT_HABITS
        )
        print(f"{group:12s} {k:4d} {start // 1000:>3d}-{end // 1000:<4d}k {rate:>26s} {test:>15s}  {habits}")


if __name__ == "__main__":
    main()
