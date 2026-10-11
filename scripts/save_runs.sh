#!/usr/bin/env bash
# Copy runs from ~/cares_rl_logs into the repo's runs/ folder and stage them for a commit:
# configs, data/*.csv, figures and models. Leaves out the replay buffer (memory/, hundreds of
# MB), battle replays and videos. Re-running refreshes runs that were copied before.
#
#   scripts/save_runs.sh                 # every run and test under ~/cares_rl_logs/DQN
#   scripts/save_runs.sh v2-s10 v2-s20   # only folders whose name contains one of these
#
# Then: git commit -m "..." && git push
set -euo pipefail

LOG_BASE="${CARES_LOG_BASE_DIR:-$HOME/cares_rl_logs}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/runs"

mkdir -p "$DEST"
copied=()
for dir in "$LOG_BASE"/DQN/*/; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  if (( $# > 0 )); then
    match=0
    for want in "$@"; do [[ "$name" == *"$want"* ]] && match=1; done
    (( match )) || continue
  fi
  rm -rf "${DEST:?}/$name"
  tar -C "$LOG_BASE/DQN" --exclude=memory --exclude=replays --exclude=videos -cf - "$name" | tar -C "$DEST" -xf -
  copied+=("$name")
done

if (( ${#copied[@]} == 0 )); then
  echo "no runs to copy${*:+ matching: $*}" >&2
  exit 1
fi

# GitHub rejects files over 100 MB
big=$(cd "$DEST" && find "${copied[@]}" -type f -size +90M)
if [[ -n "$big" ]]; then
  echo "these files are too big for GitHub - remove them from runs/ first:" >&2
  echo "$big" >&2
  exit 1
fi

git -C "$REPO" add runs
for name in "${copied[@]}"; do du -sh "$DEST/$name"; done
echo "$(git -C "$REPO" diff --cached --name-only -- runs | wc -l) files staged in runs/ - now commit and push"
