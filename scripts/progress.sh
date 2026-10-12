#!/usr/bin/env bash
# Progress of the runs started by train_parallel.sh: steps done, estimated time left and the
# eval win rates at the latest checkpoints (10 battles each, so expect noise), plus each run's
# memory and the machine's. A run's memory should level off (about 1 GB); steady growth means a
# leak. For win rates you can compare, use scripts/summarize_runs.py.
#
#   scripts/progress.sh
#   RUN_PREFIX=v2 scripts/progress.sh   # an older environment version's runs
set -uo pipefail

LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"
RUN_PREFIX="${RUN_PREFIX:-v3}"
now=$(date +%s)

printf "%-10s %-8s %7s %5s %9s %7s  %s\n" run status steps done "time left" memory "eval wins @ latest checkpoints"
# Runs sit in a folder per algorithm (DQN/, DuelingDQN/, ...)
for dir in "$LOG_BASE"/*/*-random-max-"$RUN_PREFIX"-*; do
  [[ -d "$dir" ]] || continue
  base=$(basename "$dir")
  # e.g. DQN-random-max-v3-s10-26_10_11_04-21-03 -> run v3-s10, started 2026-10-11 04:21:03
  name=$(sed -E 's/^[A-Za-z0-9]+-random-max-(.*)-[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$/\1/' <<< "$base")
  started=$(sed -E 's/.*-([0-9]{2})_([0-9]{2})_([0-9]{2})_([0-9]{2})-([0-9]{2})-([0-9]{2})$/20\1-\2-\3 \4:\5:\6/' <<< "$base")
  elapsed=$(( now - $(date -d "$started" +%s) ))
  # Results sit in a folder named after the run's seed (10, 20, ...)
  seed_dir=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | head -n 1)
  # Each run's own length (v3-long trains for longer)
  STEPS=$(grep -o '"max_steps_training": [0-9]*' "$dir/alg_config.json" 2>/dev/null | grep -o '[0-9]*$')
  STEPS=${STEPS:-200000}

  steps=$(tail -n 1 "$seed_dir/data/train.csv" 2>/dev/null | cut -d, -f1)
  [[ "$steps" =~ ^[0-9]+$ ]] || steps=0

  mem="-"
  pids=$(pgrep -d, -f -- "--run-name $name ")
  if [[ -n "$pids" ]]; then
    status=running
    # The training process is the biggest one whose command line names the run
    mem="$(( $(ps -o rss= -p "$pids" 2>/dev/null | sort -n | tail -n 1) / 1024 ))MB"
    if (( steps > 0 )); then
      left=$(( (STEPS - steps) * elapsed / steps ))
      left=$(printf "%dh%02dm" $((left / 3600)) $((left % 3600 / 60)))
    else
      left="?"
    fi
  else
    status=$([[ -d "$seed_dir/models/final" ]] && echo finished || echo stopped)
    left="-"
  fi

  evals=$(awk -F, 'NR > 1 { n[$1]++; w[$1] += ($3 == "True") }
                   END { for (k in n) print k, w[k] "/" n[k] }' "$seed_dir/data/eval.csv" 2>/dev/null \
          | sort -n | tail -n 4 | awk '{ printf "%s@%dk  ", $2, $1 / 1000 }')

  printf "%-10s %-8s %7d %4d%% %9s %7s  %s\n" "$name" "$status" "$steps" $(( steps * 100 / STEPS )) "$left" "$mem" "${evals:--}"
done

free -m | awk '/^Mem:/ { printf "\nmachine memory: %d of %d MB used, %d MB available\n", $3, $2, $7 }'
