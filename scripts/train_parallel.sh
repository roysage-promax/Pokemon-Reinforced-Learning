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
#   TEST_AND_PUSH=1 scripts/train_parallel.sh        # test, commit and push as soon as these runs
#                                                    # end, without waiting for other batches
#   TEST_AND_PUSH=1 DIAGNOSE=1 SHUTDOWN_WHEN_DONE=1 scripts/train_parallel.sh   # after the tests,
#                                                    # also scripts/diagnose_batch.sh on these runs
#   scripts/train_parallel.sh v6-s10 v6-s20          # only these runs
#   RESUME=1 TEST_AND_PUSH=1 DIAGNOSE=1 SHUTDOWN_WHEN_DONE=1 scripts/train_parallel.sh
#                                                    # carry on with these runs from their last
#                                                    # checkpoints (finished runs are just tested)
#
# A watcher then follows the runs (log: $OUT_DIR/after_training.log). It restarts the Showdown
# server if it dies (SERVER_START) and resumes a run that has logged no battle for STALL_MINUTES
# (default 15) or ended without a final model, up to MAX_RESUMES (5) times; WATCHDOG=0 turns
# that off. A stalled run's old folder goes to STALLED_DIR (~/cares_rl_stalled).
#
# The runs use this checkout's environment (PYTHONPATH below), so a second checkout - e.g. a git
# worktree of another branch - can train another environment version next to the first one's runs.
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
DIAGNOSE="${DIAGNOSE:-0}"
RESUME="${RESUME:-0}"
WATCHDOG="${WATCHDOG:-1}"
STALL_MINUTES="${STALL_MINUTES:-15}"
STALL_SECONDS="${STALL_SECONDS:-$(( STALL_MINUTES * 60 ))}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"  # seconds between the watcher's checks
MAX_RESUMES="${MAX_RESUMES:-5}"
STALLED_DIR="${STALLED_DIR:-$HOME/cares_rl_stalled}"
# How the watchdog restarts a dead Showdown server - as scripts/setup_gcp_vm.sh starts it
SERVER_START="${SERVER_START:-tmux kill-session -t showdown 2> /dev/null; tmux new -d -s showdown 'cd $REPO/../pokemon-showdown && node pokemon-showdown start --no-security'}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"

# Python finds this checkout's showdown_gym before the installed one
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# The networks are tiny, so one torch thread per run; otherwise every run grabs every core.
export OMP_NUM_THREADS=1

# name | seed | steps (empty = $STEPS) | hidden layer width (empty = cares-rl's 64) |
#   environment settings | algorithm and its settings |
#   v4 checkpoint to start from (empty = from scratch; see scripts/expand_checkpoint.py)
# v6 environment: v5 (4 moves, 6 switches, 4 move-and-terastallize actions, 14 "use our healing
# move", 15 "use our setup move") plus 16 "use our most damaging move", 17 "terastallize and use
# the most damaging move after tera", 18 "switch to the teammate the predicted move hurts least",
# 19 "switch to the best matchup" - the hand-written policy's choices, whichever slot they are in.
# Earlier batches are in git history: v2 (4 move actions only) at e5ed864, v3 at b5e07bf and
# a8cb256, v4 at fe97617, v4r at dbe63d2, v4s at cb5708e, v5 at 5a3b1a4 and a16a2d0.
# Environment settings: SHOWDOWN_ILLEGAL_PENALTY (default 0.5) - reward taken off an illegal action
#   SHOWDOWN_TERA_BONUS (default 0.3) - reward for a well-timed tera
#   SHOWDOWN_SWITCH_BONUS, SHOWDOWN_SETUP_BONUS (default 0) - reward for switching out of the
#   predicted move, setting up while resisting it, and healing when it isn't super-effective
# v4s (1M steps): tera bonus 0.6 92.3%, + switch/setup bonuses 92.8% over the last 200k steps;
# 1000-battle tests 92.0 / 92.7%. The switch bonus worked, setup and heal didn't: the agent sets up
# 5% of the times the hand-written policy does, and heals mostly when the move is in slot 0.
# v5 400k screen (5a3b1a4): from scratch 90.9% at 300-400k (v4s-rules 90.4) and still rising, tests
# 92.0 / 91.3 / 92.5%; it now sets up (good_setup 1.8 a battle, v4s 0.14). Starting from v4s-rules'
# network was worse (89.1%, tests 90.0% pooled), so that's dropped. v5-1m: v5 from scratch for 1M
# steps (a16a2d0). But v5 still picks a weaker attack than its best ~1.4 times a battle and the
# hand-written policy's switch target about half the time.
# v5-1m (ff54bdc): 92.8% over 800k-1M, flat from ~700k; 1000-battle tests 92.0 / 94.3 / 93.1%.
# v6: 400k-step screen with v5-1m's settings (8413f52). The hand-written policy through actions
# 14-19 alone wins 95.2% (3000 battles; through move and team slots 95.9%). v6 learns faster: 92.5%
# at 300-400k against v5-1m's 90.5% over the same steps (same seeds and machine, same time), about
# what v5 reached at 800k-1M, and still rising; tests 92.5 / 93.0 / 92.5%.
# v6-1m: the same for 1M steps. b128: batch 128 instead of 32 - late in training the win rate
# swings +-1.5 points between 50k-step windows, so try steadier updates.
DQN3="DQN --use_double_dqn 1 --n_step 3 --lr 0.0003 --decay_steps 200000"
RULES="SHOWDOWN_TERA_BONUS=0.6 SHOWDOWN_SWITCH_BONUS=0.3 SHOWDOWN_SETUP_BONUS=0.3"
RUNS=(
  "v6-1m-s10|10|1000000||$RULES|$DQN3|"
  "v6-1m-s20|20|1000000||$RULES|$DQN3|"
  "v6-1m-s30|30|1000000||$RULES|$DQN3|"
  "v6-1m-b128-s10|10|1000000||$RULES|$DQN3 --batch_size 128|"
  "v6-1m-b128-s20|20|1000000||$RULES|$DQN3 --batch_size 128|"
)

server_up() { (exec 3<>/dev/tcp/localhost/8000) 2> /dev/null; }

# The server compiles itself on its first start, so give it time before connecting
wait_for_server() {
  local waited=0
  until server_up; do
    if (( waited >= 180 )); then
      echo "Showdown server isn't listening on localhost:8000" >&2
      return 1
    fi
    if (( waited == 0 )); then echo "waiting for the Showdown server on localhost:8000..."; fi
    sleep 2
    waited=$((waited + 2))
  done
}

# A run's latest results folder (a resumed run gets a new one), or nothing
run_dir() {
  compgen -G "$LOG_BASE/*/*-random-max-$1-[0-9][0-9]_*" | sort | tail -n 1 || true
}

finished() { compgen -G "$1/*/models/final/*" > /dev/null; }

# Seconds since the run last logged a battle (train.csv gets a row after every battle)
idle_seconds() {
  local newest
  newest=$(stat -c %Y "$(run_dir "$1")"/*/data/train.csv 2> /dev/null | sort -n | tail -n 1)
  echo $(( $(date +%s) - ${newest:-0} ))
}

stop_run() {
  local i
  kill "$1" 2> /dev/null || return 0
  for i in 1 2 3 4 5; do
    kill -0 "$1" 2> /dev/null || return 0
    sleep 2
  done
  kill -9 "$1" 2> /dev/null
  sleep 1
}

declare -A spec=() run_pid=() resumes=()
for run in "${RUNS[@]}"; do spec[${run%%|*}]=$run; done

# launch NAME [DIR]: start one of RUNS, or with DIR resume it from that folder's last checkpoint,
# replay buffer and logs (cares-rl writes the resumed run to a new folder). Sets STARTED_PID.
launch() {
  local name=$1 from=${2:-} seed steps width run_env alg_args transfer config_dir src
  local -a cli_args load command
  IFS='|' read -r _ seed steps width run_env alg_args transfer <<< "${spec[$name]}"
  # By default cares-rl pickles the whole replay buffer and redraws the training plot after every
  # episode, which leaves runs waiting on the disk; checkpoint every ~10k steps instead.
  # alg_args (algorithm name, then its settings) is word-split on purpose
  # shellcheck disable=SC2206
  cli_args=(--gym showdown --domain random --task max
    --seeds "$seed" --save_train_checkpoints 1 --checkpoint_interval 300
    --plot_interval 100 --record_eval_video 0
    $alg_args --max_steps_training "${steps:-$STEPS}")
  if [[ -n "$from" ]]; then
    # The run's settings come from its folder; its reward settings (run_env) don't, so pass them again.
    # Loading a 1M-step replay buffer (1-3 GB) comes before the first battle, so allow longer
    local STARTUP_TIMEOUT=$(( STARTUP_TIMEOUT > 600 ? STARTUP_TIMEOUT : 600 ))
    command=(resume)
    load=(--data_path "$from")
  else
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
        return 1
      fi
      "$PYTHON" "$(dirname "$0")/expand_checkpoint.py" "$src" "$OUT_DIR/transfer/$name"
      command=(transfer --transfer_path "$OUT_DIR/transfer/$name")
    fi
  fi
  # run_env (VAR=value ...) is word-split on purpose
  # shellcheck disable=SC2086
  start_run "$name" "$OUT_DIR/$name.log" env $run_env "$CARES_RL" "${command[@]}" --run-name "$name" --skip-prompts \
    "${load[@]}"
}

# resume_run NAME: restart a run that stalled or died from its last checkpoint (from scratch if it
# never saved one), then move its old folder to STALLED_DIR so that tests and summaries see one
# folder per run. The old console log is kept as <run>.log.stalled-<time>: it shows why it stopped.
resume_run() {
  local name=$1 dir
  dir=$(run_dir "$name")
  if [[ -f "$OUT_DIR/$name.log" ]]; then
    mv "$OUT_DIR/$name.log" "$OUT_DIR/$name.log.stalled-$(date +%H%M%S)"
  fi
  if [[ -n "$dir" ]] && compgen -G "$dir/*/models/checkpoint/*" > /dev/null; then
    launch "$name" "$dir" || return 1
  else
    echo "$name: no checkpoint to resume from - starting it again"
    launch "$name" || return 1
  fi
  run_pid[$name]=$STARTED_PID
  if [[ -n "$dir" ]]; then
    mkdir -p "$STALLED_DIR"
    mv "$dir" "$STALLED_DIR/"
  fi
}

# Wait for the started runs to finish. With WATCHDOG=1, once a minute: restart the Showdown server
# if it's gone, and resume a run that has logged no battle for STALL_MINUTES or ended without a
# final model, up to MAX_RESUMES times per run. (The server crashed, or dropped one run's
# connection, three times in v6-1m's first 4 hours; poke_env then waits for a message forever.)
watch_runs() {
  local name pid idle active
  local -A over=()
  while :; do
    if [[ "$WATCHDOG" == 1 ]] && ! server_up; then
      echo "$(date) the Showdown server is down - restarting it"
      bash -c "$SERVER_START"
      wait_for_server || echo "$(date) the server didn't come back"
    fi
    active=0
    for name in "${started[@]}"; do
      [[ -n "${over[$name]:-}" ]] && continue
      pid=${run_pid[$name]:-}
      if [[ -n "$pid" ]] && kill -0 "$pid" 2> /dev/null; then
        idle=$(idle_seconds "$name")
        if [[ "$WATCHDOG" != 1 ]] || (( idle < STALL_SECONDS )) || finished "$(run_dir "$name")"; then
          active=1
          continue
        fi
        echo "$(date) $name: no battle logged for $((idle / 60)) min - stopping it (pid $pid)"
        stop_run "$pid"
      elif finished "$(run_dir "$name")"; then
        echo "$(date) $name finished"
        over[$name]=finished
        continue
      else
        echo "$(date) $name ended without a final model"
        if [[ "$WATCHDOG" != 1 ]]; then
          over[$name]=stopped
          continue
        fi
      fi
      if (( ${resumes[$name]:-0} >= MAX_RESUMES )); then
        echo "$(date) $name: resumed $MAX_RESUMES times already - giving up on it"
        over[$name]=given-up
        continue
      fi
      resumes[$name]=$(( ${resumes[$name]:-0} + 1 ))
      echo "$(date) resuming $name (restart ${resumes[$name]} of at most $MAX_RESUMES)"
      if resume_run "$name"; then
        active=1
      else
        echo "$(date) $name didn't restart - giving up on it"
        over[$name]=given-up
      fi
    done
    (( active )) || return 0
    sleep "$WATCH_INTERVAL"
  done
}

wait_for_server || exit 1
mkdir -p "$OUT_DIR"
started=()
failed=()

if [[ "$RESUME" == 1 ]]; then
  # An earlier call's watcher would test, push and shut down as soon as its stalled runs are stopped
  for pid in $(pgrep -f -- '^[^ ]*bash [^ ]*train_parallel\.sh' || true); do
    [[ "$pid" == "$$" ]] && continue
    kill "$pid" 2> /dev/null && echo "stopped an earlier train_parallel.sh watcher (pid $pid)"
  done
fi

for run in "${RUNS[@]}"; do
  name=${run%%|*}
  if (( $# > 0 )) && [[ " $* " != *" $name "* ]]; then continue; fi
  if [[ "$RESUME" == 1 ]]; then
    dir=$(run_dir "$name")
    if [[ -n "$dir" ]] && finished "$dir"; then
      echo "$name has already finished: $dir"
      started+=("$name")
      continue
    fi
    for pid in $(pgrep -f -- "--run-name $name " || true); do
      echo "stopping $name's old process (pid $pid)"
      stop_run "$pid"
    done
    if resume_run "$name"; then started+=("$name"); else failed+=("$name"); fi
  elif launch "$name"; then
    run_pid[$name]=$STARTED_PID
    started+=("$name")
  else
    failed+=("$name")
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "these runs failed to start - see $OUT_DIR/<run>.log.failed-*: ${failed[*]}" >&2
fi
if (( ${#started[@]} == 0 )); then
  echo "no runs started${*:+ - none match: $*}" >&2
  exit 1
fi

# The watcher checks once a minute and keeps going after you log out (see watch_runs). Before a
# shutdown it also waits for every other cares-rl run on the machine, so a second call (another
# batch) can't power off under the first call's runs. Only the runs started here are tested and
# pushed: earlier runs of the same environment version may still be on the machine. Their version
# is the first run name's prefix (v6-s10 -> v6).
prefix="${RUNS[0]%%-*}"
(
  # Nothing here may stop the shutdown: a failed test or push just gets logged
  set +e
  trap '' HUP
  watch_runs
  echo "${started[*]} ended $(date)"
  if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then
    while pgrep -f -- "$CARES_RL (train|transfer|resume|test) .*--run-name" > /dev/null; do
      sleep "$WATCH_INTERVAL"
    done
    echo "all runs ended $(date)"
  fi
  if [[ "$TEST_AND_PUSH" == 1 ]]; then
    RUN_PREFIX="$prefix" "$(dirname "$0")/finish_batch.sh" "${started[@]}" || echo "finish_batch.sh failed - see above"
  fi
  if [[ "$DIAGNOSE" == 1 ]]; then
    "$(dirname "$0")/diagnose_batch.sh" "${started[@]}" || echo "diagnose_batch.sh failed - see above"
  fi
  if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then
    echo "shutting down $(date)"
    $SHUTDOWN_CMD
  fi
) >> "$OUT_DIR/after_training.log" 2>&1 &
disown
todo=""
if [[ "$WATCHDOG" == 1 ]]; then todo+=" (restarting the server or a stalled run if needed)"; fi
if [[ "$TEST_AND_PUSH" == 1 ]]; then todo+=" then test ${started[*]}, commit and push;"; fi
if [[ "$DIAGNOSE" == 1 ]]; then todo+=" diagnose, commit and push;"; fi
if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then todo+=" shut down once every cares-rl run on this machine has ended"; fi
echo "watching ${started[*]}${todo%;} (log: $OUT_DIR/after_training.log)"
