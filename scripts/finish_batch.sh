#!/usr/bin/env bash
# After a batch: test the named runs - or, with no names, every run of one environment version
# (RUN_PREFIX, default v6) - for EPISODES battles (default 1000), copy the runs and their tests into
# runs/, commit and push. train_parallel.sh runs this with the runs it started once training ends
# when started with TEST_AND_PUSH=1; it can also be run by hand.
# Needs the Showdown server running. Pushing must work without a prompt (an SSH key or a stored
# token) - check with: GIT_TERMINAL_PROMPT=0 git push --dry-run
#
#   scripts/finish_batch.sh
#   scripts/finish_batch.sh v6-s10 v6-s20
#   RUN_PREFIX=v4s EPISODES=100 scripts/finish_batch.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"
export RUN_PREFIX="${RUN_PREFIX:-v6}"
export EPISODES="${EPISODES:-1000}"

echo "== $(date) testing ${*:-the $RUN_PREFIX runs}, $EPISODES battles each"
scripts/test_parallel.sh "$@" || echo "some tests failed - committing what there is"

echo "== $(date) summary"
"$PYTHON" scripts/summarize_runs.py --prefix "$RUN_PREFIX" || true

# showdown_environment.py on its own in a clean copy of the package, with each run's final network
echo "== $(date) submission check"
check=passed
if (( $# > 0 )); then
  for name in "$@"; do
    model=$(compgen -G "$LOG_BASE/*/*-random-max-$name-*/*/models/final/*_checkpoint.pth" | head -n 1) || true
    if [[ -z "$model" ]]; then
      echo "$name: no final model"
      check=FAILED
    elif ! "$PYTHON" scripts/check_submission.py "$model"; then
      check=FAILED
    fi
  done
elif ! "$PYTHON" scripts/check_submission.py; then
  check=FAILED
fi

echo "== $(date) saving and pushing"
# "-v5-1m-s10-" matches the run (...-max-v5-1m-s10-<date>) and its test (...-max-test-v5-1m-s10-<date>),
# "-v5-" every v5 run and test
if (( $# > 0 )); then
  patterns=()
  for name in "$@"; do patterns+=("-$name-"); done
  scripts/save_runs.sh "${patterns[@]}" || exit 1
  git commit -q -m "$* + $EPISODES-battle tests (submission check $check)" || exit 1
else
  scripts/save_runs.sh "-$RUN_PREFIX-" || exit 1
  git commit -q -m "$RUN_PREFIX runs + $EPISODES-battle tests (submission check $check)" || exit 1
fi
# Fail instead of waiting for a password nobody will type
GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push origin HEAD || {
  echo "push failed - the commit is on this machine: git push origin HEAD"
  exit 1
}
echo "== $(date) pushed $(git rev-parse --short HEAD)"
