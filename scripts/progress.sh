#!/usr/bin/env bash
# Progress of the runs started by train_parallel.sh: steps done, estimated time left and the
# eval win rates at the latest checkpoints (10 battles each, so expect noise).
#
#   scripts/progress.sh
set -uo pipefail

LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"
STEPS="${STEPS:-200000}"
now=$(date +%s)

printf "%-10s %-8s %7s %5s %9s  %s\n" run status steps done "time left" "eval wins @ latest checkpoints"
for dir in "$LOG_BASE"/DQN/DQN-random-max-v2-*; do
  [[ -d "$dir" ]] || continue
  base=$(basename "$dir")
  # e.g. DQN-random-max-v2-s10-26_10_11_04-21-03 -> run v2-s10, started 2026-10-11 04:21:03
  name=$(sed -E 's/^DQN-random-max-(.*)-[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$/\1/' <<< "$base")
  started=$(sed -E 's/.*-([0-9]{2})_([0-9]{2})_([0-9]{2})_([0-9]{2})-([0-9]{2})-([0-9]{2})$/20\1-\2-\3 \4:\5:\6/' <<< "$base")
  elapsed=$(( now - $(date -d "$started" +%s) ))

  steps=$(tail -n 1 "$dir/10/data/train.csv" 2>/dev/null | cut -d, -f1)
  [[ "$steps" =~ ^[0-9]+$ ]] || steps=0

  if pgrep -f -- "--run-name $name " > /dev/null; then
    status=running
    if (( steps > 0 )); then
      left=$(( (STEPS - steps) * elapsed / steps ))
      left=$(printf "%dh%02dm" $((left / 3600)) $((left % 3600 / 60)))
    else
      left="?"
    fi
  else
    status=$([[ -d "$dir/10/models/final" ]] && echo finished || echo stopped)
    left="-"
  fi

  evals=$(awk -F, 'NR > 1 { n[$1]++; w[$1] += ($3 == "True") }
                   END { for (k in n) print k, w[k] "/" n[k] }' "$dir/10/data/eval.csv" 2>/dev/null \
          | sort -n | tail -n 4 | awk '{ printf "%s@%dk  ", $2, $1 / 1000 }')

  printf "%-10s %-8s %7d %4d%% %9s  %s\n" "$name" "$status" "$steps" $(( steps * 100 / STEPS )) "$left" "${evals:--}"
done
