#!/usr/bin/env bash
# Launch several training runs against the max expert in parallel, one process per run. Runs are
# started one after another - each once the previous has played its first battle (see
# start_run.sh) - so this takes ~20 s per run before it returns.
# Needs the Showdown server running first (node pokemon-showdown start --no-security).
#
#   scripts/train_parallel.sh                        # every run below, 400k steps unless it sets its own
#   STEPS=50000 scripts/train_parallel.sh            # shorter check (runs with their own step count keep it)
#   SHUTDOWN_WHEN_DONE=1 scripts/train_parallel.sh   # power the machine off once every run has ended
#   TEST_AND_PUSH=1 SHUTDOWN_WHEN_DONE=1 scripts/train_parallel.sh   # and before that run
#                                                    # scripts/finish_batch.sh: test, commit, push
#   scripts/train_parallel.sh v5-s10 v5-warm-s10     # only these runs
#
# Results go to $CARES_LOG_BASE_DIR (default ~/cares_rl_logs); console output to $OUT_DIR/<run>.log.
set -euo pipefail
# shellcheck source=scripts/start_run.sh
source "$(dirname "$0")/start_run.sh"

STEPS="${STEPS:-400000}"
CARES_RL="${CARES_RL:-cares-rl}"
PYTHON="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-rl_logs}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-0}"
TEST_AND_PUSH="${TEST_AND_PUSH:-0}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"

# The networks are tiny, so one torch thread per run; otherwise every run grabs every core.
export OMP_NUM_THREADS=1

# name | seed | steps (empty = $STEPS) | hidden layer width (empty = cares-rl's 64) |
#   environment settings | algorithm and its settings |
#   v4 checkpoint to start from (empty = from scratch; see scripts/expand_checkpoint.py)
# v5 environment: v4 (4 moves, 6 switches, 4 move-and-terastallize actions) plus action 14 "use our
# healing move" and 15 "use our setup move", whichever slot it is in. Earlier batches are in git
# history: v2 (4 move actions only) at e5ed864, v3 at b5e07bf and a8cb256, v4 at fe97617, v4r at
# dbe63d2, v4s at cb5708e.
# Environment settings: SHOWDOWN_ILLEGAL_PENALTY (default 0.5) - reward taken off an illegal action
#   SHOWDOWN_TERA_BONUS (default 0.3) - reward for a well-timed tera
#   SHOWDOWN_SWITCH_BONUS, SHOWDOWN_SETUP_BONUS (default 0) - reward for switching out of the
#   predicted move, setting up while resisting it, and healing when it isn't super-effective
# v4s (1M steps): tera bonus 0.6 92.3%, + switch/setup bonuses 92.8% over the last 200k steps;
# 1000-battle tests 92.0 / 92.7%. The switch bonus worked, setup and heal didn't: the agent sets up
# 5% of the times the hand-written policy does, and heals mostly when the move is in slot 0.
# v5: 400k-step screen against v4s-rules' first 400k (90.3-90.5% at 300-400k). v5 = from scratch;
# v5-warm starts from v4s-rules-s10's final network (new inputs and outputs added) and explores
# again from epsilon 0.2.
DQN3="DQN --use_double_dqn 1 --n_step 3 --lr 0.0003 --decay_steps 200000"
WARM="DQN --use_double_dqn 1 --n_step 3 --lr 0.0003 --start_epsilon 0.2 --decay_steps 100000"
FROM_V4S="runs/DQN-random-max-v4s-rules-s10-*/10/models/final/DQN_checkpoint.pth"
RULES="SHOWDOWN_TERA_BONUS=0.6 SHOWDOWN_SWITCH_BONUS=0.3 SHOWDOWN_SETUP_BONUS=0.3"
RUNS=(
  "v5-s10|10|||$RULES|$DQN3|"
  "v5-s20|20|||$RULES|$DQN3|"
  "v5-s30|30|||$RULES|$DQN3|"
  "v5-warm-s10|10|||$RULES|$WARM|$FROM_V4S"
  "v5-warm-s20|20|||$RULES|$WARM|$FROM_V4S"
  "v5-warm-s30|30|||$RULES|$WARM|$FROM_V4S"
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
  IFS='|' read -r name seed steps width run_env alg_args transfer <<< "$run"
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
  command=(train)
  if [[ -n "$transfer" ]]; then
    # The pattern names one checkpoint in the repo; give it v5's extra inputs and outputs
    src=$(compgen -G "$REPO/$transfer" | head -n 1) || true
    if [[ -z "$src" ]]; then
      echo "$name: no checkpoint matches $transfer" >&2
      failed+=("$name")
      continue
    fi
    "$PYTHON" "$(dirname "$0")/expand_checkpoint.py" "$src" "$OUT_DIR/transfer/$name"
    command=(transfer --transfer_path "$OUT_DIR/transfer/$name")
  fi
  # run_env (VAR=value ...) is word-split on purpose
  # shellcheck disable=SC2086
  if start_run "$name" "$OUT_DIR/$name.log" env $run_env "$CARES_RL" "${command[@]}" --run-name "$name" --skip-prompts \
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

if [[ "$SHUTDOWN_WHEN_DONE" == 1 || "$TEST_AND_PUSH" == 1 ]]; then
  # Checks once a minute and keeps going after you log out. Waits for every cares-rl run on the
  # machine, not just these, so a second call (e.g. restarting a run) can't power off under the
  # first call's runs. The batch's environment version is the first run name's prefix (v5-s10 -> v5).
  prefix="${RUNS[0]%%-*}"
  (
    # Nothing here may stop the shutdown: a failed test or push just gets logged
    set +e
    trap '' HUP
    while pgrep -f -- "$CARES_RL (train|transfer|test) .*--run-name" > /dev/null; do
      sleep 60
    done
    echo "all runs ended $(date)"
    if [[ "$TEST_AND_PUSH" == 1 ]]; then
      RUN_PREFIX="$prefix" "$(dirname "$0")/finish_batch.sh" || echo "finish_batch.sh failed - see above"
    fi
    if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then
      echo "shutting down $(date)"
      $SHUTDOWN_CMD
    fi
  ) >> "$OUT_DIR/after_training.log" 2>&1 &
  disown
  todo=""
  if [[ "$TEST_AND_PUSH" == 1 ]]; then todo+=" test the $prefix runs, commit and push;"; fi
  if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then todo+=" shut down"; fi
  echo "once every cares-rl run on this machine has ended:${todo%;} (log: $OUT_DIR/after_training.log)"
fi
