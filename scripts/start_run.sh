# Sourced by train_parallel.sh and test_parallel.sh.
#
# Showdown account names are stamped with the second a run creates its environments
# (time.strftime("%H%M%S") in SingleShowdownWrapper), and how long a run takes to get there varies -
# the first one loads torch from a cold disk. So runs started a few seconds apart can still log in
# under the same names; one of them then never gets a battle and dies with "Agent is not
# challenging". start_run therefore starts runs one at a time: it waits for each to finish its first
# battle (a "total_steps:" line in its log) before returning, and retries a run that dies or stalls.
#
#   start_run NAME LOG_FILE COMMAND [ARGS...]
#     NAME is the --run-name; sets STARTED_PID; returns 1 if the run never started

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
STARTUP_ATTEMPTS="${STARTUP_ATTEMPTS:-3}"
LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"

start_run() {
  local name=$1 log=$2
  shift 2
  local attempt pid waited marker dir
  for (( attempt = 1; attempt <= STARTUP_ATTEMPTS; attempt++ )); do
    marker=$(mktemp)
    nohup "$@" > "$log" 2>&1 &
    pid=$!
    echo "starting $name (pid $pid) -> $log"

    waited=0
    while kill -0 "$pid" 2>/dev/null && ! grep -q "total_steps:" "$log" && (( waited < STARTUP_TIMEOUT )); do
      sleep 2
      waited=$((waited + 2))
    done
    if grep -q "total_steps:" "$log"; then
      rm -f "$marker"
      STARTED_PID=$pid
      return 0
    fi

    kill "$pid" 2>/dev/null || true
    cp "$log" "$log.failed-$attempt"
    echo "  $name didn't start (attempt $attempt of $STARTUP_ATTEMPTS): $(grep -v '^[[:space:]]*$' "$log" | tail -n 1)" >&2
    # Remove the run folder this attempt made - only if it holds no results
    for dir in "$LOG_BASE"/*/*-"$name"-*; do
      [[ -d "$dir" && "$dir" -nt "$marker" ]] || continue
      compgen -G "$dir/*/data/*.csv" > /dev/null || rm -rf "$dir"
    done
    rm -f "$marker"
    sleep 2
  done
  return 1
}
