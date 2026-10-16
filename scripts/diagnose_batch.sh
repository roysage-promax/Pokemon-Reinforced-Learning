#!/usr/bin/env bash
# After a batch has been tested (finish_batch.sh): find out where its best network loses compared
# with the hand-written policy, and why even the hand-written policy loses. Writes the reports to
# diagnostics/<batch>/ (batch = the first run name without its seed, v6-1m-s10 -> v6-1m), then
# commits and pushes them. train_parallel.sh runs this after finish_batch.sh when started with
# DIAGNOSE=1; it can also be run by hand. Needs the Showdown server running.
#
#   scripts/diagnose_batch.sh v6-1m-s10 v6-1m-s20 v6-1m-s30
#   BATTLES=100 scripts/diagnose_batch.sh v6-s10        # quick try
#
#   README.txt          which network was used, and every report's win rate in one table
#   override-<what>.txt audit_policy.py --hand_plays <what>: the network plays, but the hand-written
#                       policy takes over one kind of decision; the win rate gained over "none" is what
#                       the network's mistakes of that kind cost ("all" = the hand policy alone)
#   hand/, agent/       battle_log.py: luck counters for every battle and each loss turn by turn,
#                       for the hand-written policy and for the network
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"
OUT_DIR="${OUT_DIR:-rl_logs}"
BATTLES="${BATTLES:-3000}"
TIME_LIMIT="${TIME_LIMIT:-3h}"
TAKEOVERS=(none all forced switch heal setup tera attack)

export OMP_NUM_THREADS=1
# Python finds this checkout's showdown_gym before the installed one (see train_parallel.sh)
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

if (( $# == 0 )); then
  echo "usage: scripts/diagnose_batch.sh RUN_NAME..." >&2
  exit 1
fi

# The network: the run whose latest test (finish_batch.sh's 1000 battles) won the most
best="" best_wins=-1 best_n=1 best_model="" best_test=""
for name in "$@"; do
  csv=$(compgen -G "$LOG_BASE/*/*-random-max-test-$name-*/*/data/eval.csv" | tail -n 1) || true
  model=$(compgen -G "$LOG_BASE/*/*-random-max-$name-*/*/models/final/*_checkpoint.pth" | head -n 1) || true
  if [[ -z "$csv" || -z "$model" ]]; then
    echo "$name: no test or no final model - left out"
    continue
  fi
  read -r wins n < <(awk -F, 'NR > 1 { n++; w += ($3 == "True") } END { print w + 0, n + 0 }' "$csv")
  echo "$name: $wins/$n in its test"
  # Compare win fractions: wins * best_n > best_wins * n
  if (( best_wins < 0 || wins * best_n > best_wins * n )); then
    best=$name best_wins=$wins best_n=$n best_model=$model best_test="$wins/$n"
  fi
done
if [[ -z "$best" ]]; then
  echo "no tested run with a final model" >&2
  exit 1
fi

batch="${1%-s[0-9]*}"
out="diagnostics/$batch"
rm -rf "$out"
mkdir -p "$out" "$OUT_DIR"
echo "== $(date) diagnosing $best ($best_test in its test), $BATTLES battles per report -> $out"

# Player names are stamped with the second an environment starts, so start them a few seconds apart
pids=()
for what in "${TAKEOVERS[@]}"; do
  timeout "$TIME_LIMIT" "$PYTHON" scripts/audit_policy.py "$best_model" --battles "$BATTLES" --hand_plays "$what" \
    > "$out/override-$what.txt" 2> "$OUT_DIR/diagnose-override-$what.err" &
  pids+=($!)
  sleep 3
done
timeout "$TIME_LIMIT" "$PYTHON" scripts/battle_log.py --battles "$BATTLES" --out "$out/hand" \
  > /dev/null 2> "$OUT_DIR/diagnose-hand.err" &
pids+=($!)
sleep 3
timeout "$TIME_LIMIT" "$PYTHON" scripts/battle_log.py --checkpoint "$best_model" --battles "$BATTLES" --out "$out/agent" \
  > /dev/null 2> "$OUT_DIR/diagnose-agent.err" &
pids+=($!)
for pid in "${pids[@]}"; do
  wait "$pid" || echo "a report exited with an error - see $OUT_DIR/diagnose-*.err"
done

{
  echo "$batch diagnostics, $(date -u +%Y-%m-%d): network $best (final model, $best_test in its test)"
  echo "$best_model"
  echo
  echo "win rate with the hand-written policy taking over one kind of decision (scripts/audit_policy.py):"
  for what in "${TAKEOVERS[@]}"; do
    result=$(grep -m 1 "battles in" "$out/override-$what.txt" | sed 's/.*agent won //')
    changed=$(grep -m 1 "^hand policy plays" "$out/override-$what.txt" | sed 's/.* - //')
    printf "  %-7s %-26s %s\n" "$what" "${result:-no result}" "$changed"
  done
  echo
  echo "battle logs (scripts/battle_log.py):"
  for who in hand agent; do
    echo "  $who: $(head -n 1 "$out/$who/summary.txt" 2> /dev/null | sed 's/.*battles in [0-9]*s, //')"
  done
} > "$out/README.txt"
cat "$out/README.txt"

git add "$out" || exit 1
git commit -q -m "$batch diagnostics: hand-policy takeovers and loss logs ($best, $BATTLES battles each)" || exit 1
GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push origin HEAD || {
  echo "push failed - the commit is on this machine: git push origin HEAD"
  exit 1
}
echo "== $(date) pushed $(git rev-parse --short HEAD)"
