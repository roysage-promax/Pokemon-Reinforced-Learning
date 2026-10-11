#!/usr/bin/env bash
# The grading test - 100 battles against the max expert with --eval_seed 10 - for every v2 run,
# in parallel, then a win-rate summary. cares-rl tests a run's models/final, or its highest
# numbered checkpoint if it has no final one. Needs the Showdown server running.
#
#   scripts/test_parallel.sh                        # every v2 run
#   scripts/test_parallel.sh v2-s10 v2-s20          # only these runs
#   EPISODES=300 scripts/test_parallel.sh v2-s20    # more battles for a tighter estimate
set -euo pipefail

LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"
CARES_RL="${CARES_RL:-cares-rl}"
OUT_DIR="${OUT_DIR:-rl_logs}"
EPISODES="${EPISODES:-100}"

export OMP_NUM_THREADS=1

waited=0
until (exec 3<>/dev/tcp/localhost/8000) 2>/dev/null; do
  if (( waited >= 180 )); then
    echo "Showdown server isn't listening on localhost:8000 - start it first" >&2
    exit 1
  fi
  if (( waited == 0 )); then echo "waiting for the Showdown server on localhost:8000..."; fi
  sleep 2
  waited=$((waited + 2))
done

mkdir -p "$OUT_DIR"
pids=()
names=()
for dir in "$LOG_BASE"/DQN/DQN-random-max-v2-*; do
  [[ -d "$dir" ]] || continue
  name=$(sed -E 's/^DQN-random-max-(.*)-[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$/\1/' <<< "$(basename "$dir")")
  if (( $# > 0 )) && [[ " $* " != *" $name "* ]]; then continue; fi
  nohup "$CARES_RL" test --run-name "test-$name" --skip-prompts \
    --data_path "$dir" --eval_seed 10 --episodes "$EPISODES" \
    > "$OUT_DIR/test-$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  echo "testing $name (pid $!) -> $OUT_DIR/test-$name.log"
  # Showdown account names are stamped with the start time in seconds, so tests must not start together
  sleep 5
done

if (( ${#pids[@]} == 0 )); then
  echo "no runs to test${*:+ matching: $*}" >&2
  exit 1
fi

echo "waiting for the tests to finish..."
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || echo "${names[$i]}: test exited with an error - see $OUT_DIR/test-${names[$i]}.log"
done

echo
printf "%-34s %-8s %s\n" "test" "model" "wins"
for csv in "$LOG_BASE"/DQN/DQN-random-max-test-v2-*/*/data/eval.csv; do
  [[ -f "$csv" ]] || continue
  test_dir=$(basename "$(dirname "$(dirname "$(dirname "$csv")")")")
  awk -F, -v name="${test_dir#DQN-random-max-}" \
    'NR > 1 { n++; w += ($3 == "True"); model = $1 }
     END { if (n) printf "%-34s %-8s %d/%d = %d%%\n", name, model, w, n, 100 * w / n }' "$csv"
done
