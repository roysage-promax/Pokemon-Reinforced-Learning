#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 Google Cloud VM for training: the README install steps,
# pinned to the versions used locally. Run as your normal user (it uses sudo for apt):
#
#   curl -fsSL https://raw.githubusercontent.com/roysage-promax/Pokemon-Reinforced-Learning/state-reward-v2/scripts/setup_gcp_vm.sh | bash
set -euo pipefail

BASE="$HOME/compsys726"
VENV="$HOME/venv/pokemon"
BRANCH="${BRANCH:-state-reward-v2}"

SHOWDOWN_COMMIT=d43fb79a049f624c079c387d043ef53f62aed226
CARES_COMMIT=8ae229adb2beff0f110ea803aec1f9a0be55293c

sudo apt-get update
sudo apt-get install -y git tmux curl build-essential python3-venv python3-dev

# Node 22 LTS (Ubuntu's own nodejs package is older than Showdown likes)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

mkdir -p "$BASE"

# Pokémon Showdown server
git clone https://github.com/smogon/pokemon-showdown.git "$BASE/pokemon-showdown"
git -C "$BASE/pokemon-showdown" checkout -q "$SHOWDOWN_COMMIT"
(cd "$BASE/pokemon-showdown" && npm install)
cp "$BASE/pokemon-showdown/config/config-example.js" "$BASE/pokemon-showdown/config/config.js"
# Two battle-simulator processes so six parallel runs don't queue behind one
sed -i 's/^\tsimulator: 1,/\tsimulator: 2,/' "$BASE/pokemon-showdown/config/config.js"

# Python environment; CPU-only torch (same version as locally) instead of the multi-GB CUDA build
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu

git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git "$BASE/cares_reinforcement_learning"
git -C "$BASE/cares_reinforcement_learning" checkout -q "$CARES_COMMIT"
pip install -e "$BASE/cares_reinforcement_learning[gym]"

git clone -b "$BRANCH" https://github.com/roysage-promax/Pokemon-Reinforced-Learning.git "$BASE/showdown_gym"
pip install -r "$BASE/showdown_gym/requirements.txt"
pip install -e "$BASE/showdown_gym"

cat <<EOF

Setup done. To train:
  tmux new -d -s showdown 'cd $BASE/pokemon-showdown && node pokemon-showdown start --no-security'
  source $VENV/bin/activate
  cd $BASE/showdown_gym && scripts/train_parallel.sh

Check progress: tail -n 3 $BASE/showdown_gym/rl_logs/*.log
Results:        ~/cares_rl_logs/DQN/
EOF
