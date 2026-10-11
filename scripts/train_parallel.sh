#!/usr/bin/env bash
# Launch several DQN training runs against the max expert in parallel, one process per run.
# Needs the Showdown server running first (node pokemon-showdown start --no-security).
#
#   scripts/train_parallel.sh                        # every run below, 200k steps each
#   STEPS=50000 scripts/train_parallel.sh            # shorter check
#   SHUTDOWN_WHEN_DONE=1 scripts/train_parallel.sh   # power the machine off once every run has ended
#
# Results go to $CARES_LOG_BASE_DIR (default ~/cares_rl_logs); console output to $OUT_DIR/<run>.log.
set -euo pipefail

STEPS="${STEPS:-200000}"
CARES_RL="${CARES_RL:-cares-rl}"
OUT_DIR="${OUT_DIR:-rl_logs}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-0}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"

# The networks are tiny, so one torch thread per run; otherwise every run grabs every core.
export OMP_NUM_THREADS=1

# name | seed | DQN settings
RUNS=(
  "v2-s10|10|--use_double_dqn 1 --n_step 3 --lr 0.0003"
  "v2-s20|20|--use_double_dqn 1 --n_step 3 --lr 0.0003"
  "v2-s30|30|--use_double_dqn 1 --n_step 3 --lr 0.0003"
  "v2-lr1e-4|10|--use_double_dqn 1 --n_step 3 --lr 0.0001"
  "v2-nstep1|10|--use_double_dqn 1 --n_step 1 --lr 0.0003"
  "v2-oldhp|10|--use_double_dqn 0 --n_step 1 --lr 0.001"
)

# The server compiles itself on its first start, so give it time before connecting
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
for run in "${RUNS[@]}"; do
  IFS='|' read -r name seed dqn_args <<< "$run"
  # By default cares-rl pickles the whole replay buffer and redraws the training plot after every
  # episode, which leaves runs waiting on the disk; checkpoint every ~10k steps instead.
  # dqn_args is word-split on purpose
  # shellcheck disable=SC2086
  nohup "$CARES_RL" train --run-name "$name" --skip-prompts cli \
    --gym showdown --domain random --task max \
    --seeds "$seed" --save_train_checkpoints 1 --checkpoint_interval 300 \
    --plot_interval 100 --record_eval_video 0 \
    DQN $dqn_args --max_steps_training "$STEPS" \
    > "$OUT_DIR/$name.log" 2>&1 &
  pids+=("$!")
  echo "started $name (pid $!) -> $OUT_DIR/$name.log"
  # Showdown account names are stamped with the start time in seconds, so runs must not start together
  sleep 5
done

if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then
  # Checks once a minute and keeps going after you log out
  (
    trap '' HUP
    while true; do
      alive=0
      for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
      (( alive )) || break
      sleep 60
    done
    echo "all runs ended $(date) - shutting down"
    $SHUTDOWN_CMD
  ) > "$OUT_DIR/shutdown.log" 2>&1 &
  disown
  echo "will shut down once all runs end (log: $OUT_DIR/shutdown.log)"
fi
