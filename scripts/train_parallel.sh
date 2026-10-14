#!/usr/bin/env bash
# Launch several training runs against the max expert in parallel, one process per run. Runs are
# started one after another - each once the previous has played its first battle (see
# start_run.sh) - so this takes ~20 s per run before it returns.
# Needs the Showdown server running first (node pokemon-showdown start --no-security).
#
#   scripts/train_parallel.sh                        # every run below, 400k steps unless it sets its own
#   STEPS=50000 scripts/train_parallel.sh            # shorter check (runs with their own step count keep it)
#   SHUTDOWN_WHEN_DONE=1 scripts/train_parallel.sh   # power the machine off once every run has ended
#   scripts/train_parallel.sh v4r-alias v4r-b0.3     # only these runs
#
# Results go to $CARES_LOG_BASE_DIR (default ~/cares_rl_logs); console output to $OUT_DIR/<run>.log.
set -euo pipefail
# shellcheck source=scripts/start_run.sh
source "$(dirname "$0")/start_run.sh"

STEPS="${STEPS:-400000}"
CARES_RL="${CARES_RL:-cares-rl}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-rl_logs}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-0}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"

# The networks are tiny, so one torch thread per run; otherwise every run grabs every core.
export OMP_NUM_THREADS=1

# name | seed | steps (empty = $STEPS) | hidden layer width (empty = cares-rl's 64) |
#   environment settings | algorithm and its settings
# v4 environment: v3 (4 moves + 6 switches, opponent's predicted move in the state) plus 4
# move-and-terastallize actions and tera features. Earlier batches are in git history: v2 (4 move
# actions only) at e5ed864, v3 at b5e07bf and a8cb256.
# Environment settings: SHOWDOWN_ILLEGAL_PENALTY (default 0.5) - reward taken off an illegal action
#   SHOWDOWN_TERA_BONUS (default 0.3) - reward for a well-timed tera
# v4r: the first v4 batch (fe97617) learned to hardly ever tera (0.1 a battle; the hand-written
# policy teras 0.8). Tera actions now just use the move once tera is used up instead of being
# illegal, and a good tera earns a bonus. This batch screens the bonus size and whether some
# exploration should continue (end epsilon 0.02 instead of 0.001) at 400k steps; the v4 runs'
# first 400k steps (same settings otherwise) are the comparison. Compare with 1000-battle tests:
# end epsilon 0.02 plays ~0.5 random actions a battle, which lowers the training win rate.
DQN3="DQN --use_double_dqn 1 --n_step 3 --lr 0.0003 --decay_steps 200000"
EPS="--end_epsilon 0.02"
RUNS=(
  "v4r-alias|10|||SHOWDOWN_TERA_BONUS=0|$DQN3"
  "v4r-alias-eps|10|||SHOWDOWN_TERA_BONUS=0|$DQN3 $EPS"
  "v4r-b0.3|10|||SHOWDOWN_TERA_BONUS=0.3|$DQN3"
  "v4r-b0.3-eps|10|||SHOWDOWN_TERA_BONUS=0.3|$DQN3 $EPS"
  "v4r-b0.3-eps-s20|20|||SHOWDOWN_TERA_BONUS=0.3|$DQN3 $EPS"
  "v4r-b0.6-eps|10|||SHOWDOWN_TERA_BONUS=0.6|$DQN3 $EPS"
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
failed=()
for run in "${RUNS[@]}"; do
  IFS='|' read -r name seed steps width run_env alg_args <<< "$run"
  if (( $# > 0 )) && [[ " $* " != *" $name "* ]]; then continue; fi
  # By default cares-rl pickles the whole replay buffer and redraws the training plot after every
  # episode, which leaves runs waiting on the disk; checkpoint every ~10k steps instead.
  # alg_args (algorithm name, then its settings) is word-split on purpose
  # shellcheck disable=SC2206
  cli_args=(--gym showdown --domain random --task max
    --seeds "$seed" --save_train_checkpoints 1 --checkpoint_interval 300
    --plot_interval 100 --record_eval_video 0
    $alg_args --max_steps_training "${steps:-$STEPS}")
  if [[ -n "$width" ]]; then
    # The network size can't be set from the command line, so write the config files instead
    config_dir="$OUT_DIR/configs/$name"
    "$PYTHON" "$(dirname "$0")/write_config.py" "$config_dir" "$width" "${cli_args[@]}"
    load=(config --data_path "$config_dir")
  else
    load=(cli "${cli_args[@]}")
  fi
  # run_env (VAR=value ...) is word-split on purpose
  # shellcheck disable=SC2086
  if start_run "$name" "$OUT_DIR/$name.log" env $run_env "$CARES_RL" train --run-name "$name" --skip-prompts \
    "${load[@]}"; then
    pids+=("$STARTED_PID")
  else
    failed+=("$name")
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "these runs failed to start - see $OUT_DIR/<run>.log.failed-*: ${failed[*]}" >&2
fi
if (( ${#pids[@]} == 0 )); then
  echo "no runs started${*:+ - none match: $*}" >&2
  exit 1
fi

if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then
  # Checks once a minute and keeps going after you log out. Waits for every cares-rl run on the
  # machine, not just these, so a second call (e.g. restarting a run) can't power off under the
  # first call's runs.
  (
    trap '' HUP
    while pgrep -f -- "$CARES_RL (train|test) --run-name" > /dev/null; do
      sleep 60
    done
    echo "all runs ended $(date) - shutting down"
    $SHUTDOWN_CMD
  ) >> "$OUT_DIR/shutdown.log" 2>&1 &
  disown
  echo "will shut down once every cares-rl run on this machine has ended (log: $OUT_DIR/shutdown.log)"
fi
