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
#   MAX_PARALLEL=6 scripts/train_parallel.sh         # at most 6 runs at once, the rest queued
#                                                    # (default: one per vCPU, less 2)
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
# Runs training at once; the rest wait in a queue and start as others finish. A run keeps about one
# vCPU busy, and the Showdown server needs a couple: 30 on a c2d-standard-32, 6 on a c2d-standard-8.
MAX_PARALLEL="${MAX_PARALLEL:-$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))}"
# How the watchdog restarts a dead Showdown server - as scripts/setup_gcp_vm.sh starts it
SERVER_START="${SERVER_START:-tmux kill-session -t showdown 2> /dev/null; tmux new -d -s showdown 'cd $REPO/../pokemon-showdown && node pokemon-showdown start --no-security'}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"
SHUTDOWN_WAIT_DIR="${SHUTDOWN_WAIT_DIR:-/tmp/train_parallel_shutdown_wait}"

# Python finds this checkout's showdown_gym before the installed one
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# The networks are tiny, so one torch thread per run; otherwise every run grabs every core.
export OMP_NUM_THREADS=1

# name | seed | steps (empty = $STEPS) | hidden layer width (empty = cares-rl's 64) |
#   environment settings | algorithm and its settings |
#   v4 checkpoint to start from (empty = from scratch; see scripts/expand_checkpoint.py) |
#   earlier run to carry on from (its last checkpoint, replay buffer and logs, up to steps in all;
#   its own settings, not the algorithm settings here - see scripts/continue_run.py) |
#   learning rate from there on (empty = the earlier run's)
# A run to carry on from that's missing on this machine is started from scratch instead, unless it
# changes the learning rate (then it isn't started).
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
# v6-1m (a5f8bf7): 94.0% over 700k-1M (v5-1m 92.8), tests 94.8 / 93.3 / 94.1, b128 94.4 / 94.5.
# The hand-written policy wins 95.4-95.7%; taking over its switches (+1.4) or heals (+1.5) closes
# most of the gap (diagnostics/v6-1m). The agent switches out of a Pokémon immune to or resisting
# the predicted move and back in a turn later, which the switch bonus pays for once per opponent -
# hidden from the agent - and heals above half HP while skipping heals below it.
# v7 reward sweep: v6's environment and network, 12 reward variants x seeds 10 and 20, 400k steps.
# Compare variants by pooled training win rate over 300-400k (summarize_runs.py's variant table),
# then run the best few at 1M with new seeds. The knobs are described at the top of
# showdown_environment.py; every one defaults to v6's reward, so ctrl is v6-s10 / v6-s20 again.
#   fix      switch bonus every time it matches + the same cost for switching out of a resisted
#            or immune spot (out and back in nets nothing)
#   sw05/15  fix + a cost for any other voluntary switch     hc10    fix + a cost for healing above half HP
#   strict   fix + both costs                                 half / double  every bonus and cost x0.5 / x2
#   taken50  fix + damage taken weighted like damage dealt    ko50    fix + 0.5 per knockout
#   win2     fix + 2 for a win                                markov  fix + heal / setup bonus every time too
# v7 sweep (c6cd4fb, the VARIANTS list is in e88fffb): no variant beat ctrl. 300-400k win rates of all
# 24 runs 91.7-94.5%, and the variants differ no more than seeds do (one-way ANOVA F = 0.77); tests
# pooled 93.3%, spread exactly as luck. fix halves switching out and back in (0.36 -> 0.10-0.22 a
# battle) without winning more; sw15 came first (93.8%, tests 94.5%) but not significantly.
# v7-2m: what more training buys, 2M steps. v6-1m gained ~0.15 points per 100k from 400k to 800k
# (93.3 -> 94.2%), then flat to 1M.
#   cont     v6-1m-s10/20/30 carried on from 1M with the same settings
#   lr1e4    the same with the learning rate down from 3e-4 to 1e-4 for the second million
#   sw15     v7's best-on-paper reward (fix + 0.15 per other voluntary switch), from scratch, new seeds
#   w128     v6's reward with 128-unit hidden layers (v3's 256 did much worse than 64)
DQN3="DQN --use_double_dqn 1 --n_step 3 --lr 0.0003 --decay_steps 200000"
RULES="SHOWDOWN_TERA_BONUS=0.6 SHOWDOWN_SWITCH_BONUS=0.3 SHOWDOWN_SETUP_BONUS=0.3"
FIX="$RULES SHOWDOWN_SWITCH_ONCE=0 SHOWDOWN_SWITCH_OUT_COST=0.3"
SW15="$FIX SHOWDOWN_SWITCH_COST=0.15"
RUNS=()
for seed in 10 20 30; do
  RUNS+=("v7-2m-cont-s$seed|$seed|2000000||$RULES|$DQN3||v6-1m-s$seed|")
  RUNS+=("v7-2m-lr1e4-s$seed|$seed|2000000||$RULES|$DQN3||v6-1m-s$seed|0.0001")
done
for seed in 30 40 50; do RUNS+=("v7-2m-sw15-s$seed|$seed|2000000||$SW15|$DQN3|"); done
for seed in 10 20; do RUNS+=("v7-2m-w128-s$seed|$seed|2000000|128|$RULES|$DQN3|"); done

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
  local name=$1 from=${2:-} seed steps width run_env alg_args transfer carry_on lr config_dir src
  local -a cli_args load command
  IFS='|' read -r _ seed steps width run_env alg_args transfer carry_on lr <<< "${spec[$name]}"
  if [[ -z "$from" && -n "$carry_on" ]]; then
    src=$(run_dir "$carry_on")
    if [[ -n "$src" ]] && "$PYTHON" "$(dirname "$0")/continue_run.py" "$src" "$OUT_DIR/continue/$name" "${steps:-$STEPS}" "$lr"; then
      from="$OUT_DIR/continue/$name"
    elif [[ -n "$lr" ]]; then
      echo "$name: can't carry on from $carry_on (${src:-no run folder}) - not starting it" >&2
      return 1
    else
      echo "$name: can't carry on from $carry_on (${src:-no run folder}) - training from scratch instead"
    fi
  fi
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

# Start a run that waited in the queue: from scratch, or with RESUME=1 from its last checkpoint
start_queued() {
  if [[ "$RESUME" == 1 ]]; then
    resume_run "$1"
  elif launch "$1"; then
    run_pid[$1]=$STARTED_PID
  else
    return 1
  fi
}

# Wait for the started runs to finish, starting queued ones while fewer than MAX_PARALLEL run.
# With WATCHDOG=1, once a minute: restart the Showdown server
# if it's gone, and resume a run that has logged no battle for STALL_MINUTES or ended without a
# final model, up to MAX_RESUMES times per run. (The server crashed, or dropped one run's
# connection, three times in v6-1m's first 4 hours; poke_env then waits for a message forever.)
watch_runs() {
  local name pid idle active running
  local -A over=()
  while :; do
    if [[ "$WATCHDOG" == 1 ]] && ! server_up; then
      echo "$(date) the Showdown server is down - restarting it"
      bash -c "$SERVER_START"
      wait_for_server || echo "$(date) the server didn't come back"
    fi
    active=0
    running=0
    for name in "${started[@]}"; do
      [[ -n "${over[$name]:-}" ]] && continue
      if [[ -n "${queued[$name]:-}" ]]; then
        active=1
        continue
      fi
      pid=${run_pid[$name]:-}
      if [[ -n "$pid" ]] && kill -0 "$pid" 2> /dev/null; then
        idle=$(idle_seconds "$name")
        if [[ "$WATCHDOG" != 1 ]] || (( idle < STALL_SECONDS )) || finished "$(run_dir "$name")"; then
          active=1
          running=$((running + 1))
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
        running=$((running + 1))
      else
        echo "$(date) $name didn't restart - giving up on it"
        over[$name]=given-up
      fi
    done
    while (( running < MAX_PARALLEL && ${#queue[@]} > 0 )); do
      name=${queue[0]}
      queue=("${queue[@]:1}")
      unset "queued[$name]"
      echo "$(date) starting $name from the queue (${#queue[@]} still waiting)"
      if start_queued "$name"; then
        running=$((running + 1))
      else
        echo "$(date) $name didn't start - giving up on it"
        over[$name]=given-up
      fi
    done
    (( active )) || return 0
    sleep "$WATCH_INTERVAL"
  done
}

wait_for_server || exit 1
mkdir -p "$OUT_DIR"
started=()  # every run this call is responsible for, queued ones too
failed=()
queue=()
declare -A queued=()
launched=0

# Showdown simulates battles in `simulator` child processes (config.js; setup_gcp_vm.sh sets 2,
# enough for 6-8 runs). Many more runs at once queue behind them.
config="$REPO/../pokemon-showdown/config/config.js"
sims=$(sed -nE 's/^\tsimulator: ([0-9]+),.*/\1/p' "$config" 2> /dev/null | head -n 1 || true)
at_once=$(( MAX_PARALLEL < ${#RUNS[@]} ? MAX_PARALLEL : ${#RUNS[@]} ))
if [[ -n "$sims" ]] && (( sims * 4 < at_once )); then
  echo "warning: $config has simulator: $sims for up to $at_once runs at once - for more, set" >&2
  echo "  sed -i 's/^\tsimulator: [0-9]*,/\tsimulator: $(( (at_once + 3) / 4 )),/' $config" >&2
  echo "  and restart the server (tmux kill-session -t showdown, then start it again)" >&2
fi

# Each run keeps its replay buffer on disk (~1 GB at 1M steps) and writes a new copy before
# replacing the old one, so allow 2 GB a run
selected=0
for run in "${RUNS[@]}"; do
  if (( $# == 0 )) || [[ " $* " == *" ${run%%|*} "* ]]; then selected=$((selected + 1)); fi
done
mkdir -p "$LOG_BASE"
free_gb=$(df -Pk "$LOG_BASE" | awk 'NR == 2 { print int($4 / 1048576) }')
if [[ "${DISK_CHECK:-1}" == 1 ]] && (( free_gb < 2 * selected + 5 )); then
  echo "only $free_gb GB free under $LOG_BASE for $selected runs (want $(( 2 * selected + 5 )) GB)." >&2
  echo "The biggest replay buffers (a finished run's is only needed to carry it on - v7-2m carries on v6-1m-s10/20/30):" >&2
  du -sh "$LOG_BASE"/*/*/*/memory 2> /dev/null | sort -h | tail -n 8 >&2
  echo "Delete some (rm -rf <run>/<seed>/memory) or start anyway with DISK_CHECK=0." >&2
  exit 1
fi

if [[ "$RESUME" == 1 ]]; then
  # An earlier call's watcher would test, push and shut down as soon as its stalled runs are stopped.
  # Only this checkout's: another checkout's batch may be training next to this one.
  for pid in $(pgrep -f -- '^[^ ]*bash [^ ]*train_parallel\.sh' || true); do
    [[ "$pid" == "$$" ]] && continue
    [[ "$(readlink "/proc/$pid/cwd" 2> /dev/null)" == "$(pwd -P)" ]] || continue
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
  fi
  if (( launched >= MAX_PARALLEL )); then
    queue+=("$name")
    queued[$name]=1
    started+=("$name")
  elif start_queued "$name"; then
    started+=("$name")
    launched=$((launched + 1))
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
  # Runs that never started (say, nothing to carry on from) have no folder to test or save
  ran=()
  for name in "${started[@]}"; do [[ -n "$(run_dir "$name")" ]] && ran+=("$name"); done
  if [[ "$TEST_AND_PUSH" == 1 && ${#ran[@]} -gt 0 ]]; then
    RUN_PREFIX="$prefix" "$(dirname "$0")/finish_batch.sh" "${ran[@]}" || echo "finish_batch.sh failed - see above"
  fi
  if [[ "$DIAGNOSE" == 1 && ${#ran[@]} -gt 0 ]]; then
    "$(dirname "$0")/diagnose_batch.sh" "${ran[@]}" || echo "diagnose_batch.sh failed - see above"
  fi
  if [[ "$SHUTDOWN_WHEN_DONE" == 1 ]]; then
    # Another checkout's batch may still be training, testing, committing or pushing: wait for its
    # watcher too - but not for one that's only waiting to shut down as well, or two would wait for
    # each other forever
    mkdir -p "$SHUTDOWN_WAIT_DIR"
    touch "$SHUTDOWN_WAIT_DIR/$BASHPID"
    while :; do
      busy=0
      pgrep -f -- "$CARES_RL (train|transfer|resume|test) .*--run-name" > /dev/null && busy=1
      for pid in $(pgrep -f -- '^[^ ]*bash [^ ]*train_parallel\.sh' || true); do
        [[ "$pid" == "$BASHPID" || -e "$SHUTDOWN_WAIT_DIR/$pid" ]] && continue
        # pgrep also lists the subshell that ran it, gone by now
        kill -0 "$pid" 2> /dev/null && busy=1
      done
      (( busy )) || break
      sleep "$WATCH_INTERVAL"
    done
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
if (( ${#queue[@]} > 0 )); then
  echo "${#queue[@]} runs wait in the queue and start as others finish ($MAX_PARALLEL at a time): ${queue[*]}"
fi
