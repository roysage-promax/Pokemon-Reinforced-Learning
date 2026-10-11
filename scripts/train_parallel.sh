#!/usr/bin/env bash
# Launch several DQN training runs against the max expert in parallel, one process per run.
# Needs the Showdown server running first (node pokemon-showdown start --no-security).
#
#   scripts/train_parallel.sh                 # every run below, 200k steps each
#   STEPS=50000 scripts/train_parallel.sh     # shorter check
#
# Results go to $CARES_LOG_BASE_DIR (default ~/cares_rl_logs); console output to $OUT_DIR/<run>.log.
set -euo pipefail

STEPS="${STEPS:-200000}"
CARES_RL="${CARES_RL:-cares-rl}"
OUT_DIR="${OUT_DIR:-rl_logs}"

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

mkdir -p "$OUT_DIR"
for run in "${RUNS[@]}"; do
  IFS='|' read -r name seed dqn_args <<< "$run"
  # dqn_args is word-split on purpose
  # shellcheck disable=SC2086
  nohup "$CARES_RL" train --run-name "$name" --skip-prompts cli \
    --gym showdown --domain random --task max \
    --seeds "$seed" --save_train_checkpoints 1 --record_eval_video 0 \
    DQN $dqn_args --max_steps_training "$STEPS" \
    > "$OUT_DIR/$name.log" 2>&1 &
  echo "started $name (pid $!) -> $OUT_DIR/$name.log"
  # Showdown account names are stamped with the start time in seconds, so runs must not start together
  sleep 5
done
