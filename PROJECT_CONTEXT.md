# Project context: Pokémon Showdown RL agent (COMPSYS 726)

Briefing for an AI assistant starting a fresh conversation about this repository. It covers what the
project is, how the code works, how the submitted model was trained, how the design got from v1 to
v8, and the practical gotchas. Facts are as of 2026-10-23. Commit hashes point to the evidence.

---

## 1. What this is and where it stands

- **Assignment:** design a reinforcement-learning *environment* (`showdown_gym/showdown_environment.py`:
  observation, action space, reward) so a DQN agent learns to beat poke-env's `MaxBasePowerPlayer`
  ("max expert") in `gen9randombattle` (random teams for both sides every battle). The RL algorithm
  comes from the staff's `cares_reinforcement_learning` package (`cares-rl`). We only write the
  environment and choose hyperparameters.
- **Grading:** staff run `cares-rl test --data_path <run folder> --eval_seed 10 --episodes 100` on a
  clean install. Win rate gives the mark: ≥95% → 10, ≥90% → 9, ≥80% → 8, ≥70% → 7, ≥60% → 6.
- **Submitted model:** run `DQN-random-max-v8-s30-26_10_21_11-27-12` (v8 environment, seed 30,
  1M steps). Its 1000-battle test was 956/1000. Across the three v8 seeds it scored 2865/3000 = **95.5%**
  [94.7–96.2]. The grading command gave 96/100 in a fresh venv (2026-10-22) and 94/100 in the dev
  venv (2026-10-23). A 100-battle test is noisy: at a true 95.5% it lands at ≥95 about 2 times in 3.
- **Submission files** (Drive upload, prepared and verified, uploaded by the user):
  `~/compsys726/submission/zyao804/` = `requirements.txt` + `showdown_environment.py` + the run folder
  trimmed to configs, `30/data`, `30/figures`, `30/models/final`. The same content is on local branch
  `submission` (commit `f64bdce`, worktree `~/compsys726/showdown_gym_submission`), which is not pushed.
  The full run, with every checkpoint, is on `origin/v8-damage`.

## 2. Assignment rules that constrain the design

- Only `showdown_environment.py`, `requirements.txt` and the run's logs are submitted. The env file
  must be **self-contained**: no data files, no edits to other package files, no environment
  variables needed at test time. (The strongest-move and immunity tables are therefore embedded
  in the file; see §5.)
- `requirements.txt` is the staff's (`poke-env==0.10.0`, `numpy==2.2.5`). It covers every import.
  Re-running pipreqs would pin newer, untested versions, so don't.
- The file's defaults must produce the trained network's sizes: **114 inputs, 20 outputs**. A v8
  network only works with the v8 env file, and v6/v7 networks (84 inputs) need their own.
- The user's rule: the environment may *offer* hand-written plays as actions, but it must never choose
  the action for the agent. Choosing for it would defeat "how effectively your environment enables an
  agent to learn".
- The opponent `MaxBasePowerPlayer` always uses its highest-base-power move, whatever the matchup, and
  never switches voluntarily. **Once its active Pokémon has attacked, its next move is known exactly.**
  Much of the design exploits this ("predicted move").

## 3. Layout

```
~/compsys726/
  showdown_gym/                 this repo (git: roysage-promax/Pokemon-Reinforced-Learning)
    showdown_gym/showdown_environment.py   THE file (CRLF line endings, as the staff's - keep them)
    showdown_gym/base_environment.py       staff base class (unchanged)
    scripts/                    our tooling (§8) - not submitted
    runs/                       162 committed run/test folders (configs, CSVs, figures, models)
    diagnostics/v6-1m, v8/      post-training analyses (§6)
    README.md                   staff's assignment README (unchanged)
  showdown_gym_submission/      worktree of branch `submission`
  submission/zyao804/           Drive upload folder
  cares_reinforcement_learning/ staff RL package (editable install)
  pokemon-showdown/             local Showdown server
~/cares_rl_logs/DQN/            where cares-rl writes training and test runs
~/venv/showdown_rl              dev venv with showdown_gym + cares-rl installed (editable from this repo)
```

**Branches.** Each environment version is its own branch, and each builds on the one before:
`main` (staff + first attempt) → `smart-switch` → `state-reward-v2` → `v3-switch-actions` → `v4-tera` →
`v4-tera-reward` → `v4-shaping` → `v5-heal-setup` → `v6-macro-actions` → `v7-reward-sweep` → **`v8-damage`**
(current, pushed, HEAD `672e499`). The `runs/` folder on each branch holds that batch's results.

## 4. How the pieces connect

`cares-rl train cli --gym showdown --domain random --task max DQN ...` builds
`SingleShowdownWrapper(team_type="random", opponent_type="max")` (bottom of the env file, staff code).
That wraps `ShowdownEnvironment(BaseShowdownEnv)`, which is a poke-env `SinglesEnv` connected to the
local Showdown server at `localhost:8000`. Each env step: the agent's integer action goes through
`process_action` (our action → poke-env action index), poke-env sends it, the server resolves the
turn, and `embed_battle` builds the next observation while `calc_reward` scores the step.
`get_additional_info` adds per-battle counters, which cares-rl writes as extra columns in
`train.csv` / `eval.csv`.

Notes on our overrides of staff behaviour:
- `step` no longer deep-copies both battles every step. That copy took 42 of 47 ms per step. It keeps 4
  numbers for the reward instead (`_reward_state`), bringing a step to ≈3.7 ms (`eeba5e1`).
- `reset` drops finished battles from poke-env's players (they leaked ~1.2 MB a battle and caused OOM
  in 200k-step runs; `152c91d`).
- poke-env runs non-strict: an illegal action is replaced by the server's default choice. On "wait"
  steps (only the opponent chooses, e.g. it is replacing a fainted Pokémon), our action is ignored, so
  it is neither penalised nor counted, and the observation marks nothing legal.

## 5. The v8 environment (submitted)

### Actions (20)

| Action | Meaning |
|---|---|
| 0–3 | use move slot 0–3 |
| 4–9 | switch to team slot 0–5 (the agent also picks the replacement when one of ours faints) |
| 10–13 | terastallize + move slot 0–3 (once tera is used up, just the move, with no penalty) |
| 14 `HEAL` | use our healing move, whichever slot it is in (best attack if we have none) |
| 15 `SETUP` | use our setup move (Swords Dance, Nasty Plot...), whichever slot (best attack if none) |
| 16 `BEST_ATTACK` | the move with the highest stat-based damage / opponent's HP |
| 17 `BEST_TERA` | terastallize + the best move after tera (plain best attack once tera is used up) |
| 18 `SAFE_SWITCH` | switch to the teammate the opponent's predicted move hurts least |
| 19 `MATCHUP_SWITCH` | switch to the best matchup (max 2·offence − threat) |

Actions 14–19 ("macro actions") are the hand-written policy's choices (below). They exist because a
per-slot network output learns separately for each slot. Measured: v4s healed mostly when the heal was
in slot 0, and v5 picked a weaker attack ~1.4×/battle. Actions 0–13 remain, so any play is still possible.

### Observation (114 floats, all in [0, 1]); full itemised list in `_observation_size`'s docstring

| Group | Size | Content |
|---|---|---|
| Move slots | 4×7 | legal, relative damage (vs best option incl. tera), power·acc/150, type effectiveness/4, is setup, is heal, relative damage with tera |
| Team slots | 6×5 | legal switch, HP, predicted move's multiplier on it/4, its best move's rough damage/300, faster (base speed) |
| Battle context | 6 | both actives' HP, Pokémon left each side /6, faster, opponent types' best multiplier on us/4 |
| Predicted opponent move | 6 | revealed, guessed, power, multiplier on our active, STAB, multiplier after we tera |
| Boosts | 2 | our atk / spa stage /6 |
| Tera | 3 | can tera now, active terastallized, tera unused |
| Forced switch | 1 | |
| Macro availability | 2+6 | have heal move, have setup move; best tera damage, can switch, safest/matchup switch's threat and offence |
| **v8 damage from stats** | 30 | each slot's damage as a fraction of the opponent's current HP (plain and after tera); predicted move's damage on our active (of current HP, of max HP, after tera); per teammate: predicted move's damage on it, its best move's damage on the opponent, faster (real speed); our active faster |

Key helpers:
- `_predicted_move`: the opponent's strongest *revealed* damaging move. Before it has attacked, a guess
  from `STRONGEST_MOVE_IDS` (species → highest-BP move in its gen9 random-battle movepools, embedded,
  generated by `scripts/build_strongest_moves.py`).
- `_threat`: type multiplier including immunity abilities (Levitate, Flash Fire, Water Absorb, ...)
  on **both** sides and Air Balloon. For the opponent, abilities come from the revealed one, its single
  possible ability, or `IMMUNE_ABILITY_IDS` (embedded).
- `_damage`: the game's damage formula with levels, stats (ours from the server; the opponent's
  estimated as random battles build them: 85 EVs, 31 IVs, neutral nature), boosts, burn, STAB (incl.
  tera rules), type, accuracy, average roll 0.925, expected hits, charge/recharge halved, and known
  attacking items/abilities. It leaves out defensive items/abilities, weather, screens and crits.
  `scripts/check_damage.py`: correlation with actual damage 0.71 for incoming hits, 0.40–0.49 for
  outgoing.
- Our tera type: poke-env 0.10 doesn't fill `Pokemon.tera_type` until we tera, so it is read from
  `mon._last_request["teraType"]`. STAB uses `original_types`.

### Reward (`calc_reward`, per step)

```
0.5 × damage dealt − 0.25 × damage taken      (in fractions of a Pokémon's HP, summed over the team)
+ 0.2 × (opponent Pokémon KO'd − ours fainted)
± 1 for a win / loss
− 0.5 for an illegal action (ILLEGAL_ACTION_PENALTY)
+ TERA_BONUS for a well-timed tera (_good_tera)
+ SWITCH_BONUS / SETUP_BONUS for a play matching a hand-policy rule (_rule_bonus)
```

The bonus rules all depend on the predicted move:
- **Good tera:** it turns a neutral/super-effective hit into a resisted one, or gives the used move
  (more) STAB without the predicted move hitting harder.
- **Switch:** the predicted move hits us super-effectively and the teammate resists it, or it hits at
  least neutrally and the teammate is immune. Paid once per opposing species.
- **Heal:** the predicted move is not super-effective and we're at ≤50% HP. Paid once per
  (opponent, our species).
- **Setup:** we resist the predicted move, HP ≥50%, attack and special attack boosts below +2 (+6 if
  immune). Paid once per (opponent, our species, boost stage).

> **⚠ Training used non-default bonus values.** The submitted run was trained with
> `SHOWDOWN_TERA_BONUS=0.6 SHOWDOWN_SWITCH_BONUS=0.3 SHOWDOWN_SETUP_BONUS=0.3` (the `RULES` string
> in `scripts/train_parallel.sh`), set as environment variables. The file's defaults are 0.3 / 0 / 0,
> and the run folder's configs don't record them. Testing never uses the reward, so the grade is
> unaffected, but anyone reproducing training must set them. Every v4s–v8 run used the same RULES.

The other knobs at the top of the file (`SHOWDOWN_SWITCH_ONCE`, `*_COST`, `*_WEIGHT`, `WIN_REWARD`) come
from the v7 reward sweep. They default to the v6 reward above and were left at their defaults for v8.

### The hand-written policy (a reference point, not used at test time)

`policy()` in `scripts/check_env.py` reads only the observation vector. It's the source of the macro
actions and the bonus rules, and it measures what the observation can support. In order:
1. Forced switch → best matchup.
2. No predicted move → best attack.
3. Predicted move hits ≥2× and the safest teammate resists it, or ≥1× and the teammate is immune → safest switch.
4. Tera if good (rule above).
5. Heal if the predicted move hits ≤1× and HP ≤50% → heal; else setup if it hits ≤0.5×, HP ≥50% and boosts below the cap.
6. Otherwise best attack (with tera if step 4 said so).

It wins **96.7%** on v8 observations (v7: 96.0%). It's equivalent to "probe variant E + I" from
`scripts/scripted_probe.py`.

## 6. Training method

**Algorithm and settings** (all from `runs/DQN-random-max-v8-s30-*/alg_config.json`): DQN with
**double DQN**, **3-step returns**, γ 0.99, Adam lr **3e-4**, batch 32, replay buffer 1M (never evicts at
1M steps), no PER, hard target update every 1000 steps. ε-greedy from 1.0 to 0.001 over the first
**200k** steps, no pure-exploration phase. Network: MLP 114 → 64 → 64 → 20, ReLU. **1M steps**, seed 30.
cares-rl evaluates 10 battles every 10k steps.

**Exact launch** (built by `scripts/train_parallel.sh`, run on a Google Cloud VM):
```bash
SHOWDOWN_TERA_BONUS=0.6 SHOWDOWN_SWITCH_BONUS=0.3 SHOWDOWN_SETUP_BONUS=0.3 \
cares-rl train --run-name v8-s30 --skip-prompts cli --gym showdown --domain random --task max \
  --seeds 30 --save_train_checkpoints 1 --checkpoint_interval 300 --plot_interval 100 --record_eval_video 0 \
  DQN --use_double_dqn 1 --n_step 3 --lr 0.0003 --decay_steps 200000 --max_steps_training 1000000
```
`--checkpoint_interval 300 --plot_interval 100` (episodes) matter. cares-rl's defaults pickle the
whole replay buffer and redraw the plot every episode, which made runs disk-bound.

**Compute:** GCP VM `pokemon-train` (c2d-standard-8/16, us-east1-b; project quota is 16 vCPUs total).
Runs are CPU-bound at about one vCPU each (`OMP_NUM_THREADS=1`), and a GPU doesn't help (network ≈1.2 ms/step).
Three v8 runs at 1M took ≈16 h, sharing the VM with another batch. The user's rule: **never train on
the local PC.**

**How runs were compared** (100-battle tests are too noisy: ±~2 pts SD):
1. Training win rate after ε reaches its floor. Every training battle is then effectively greedy.
   Pool per 50k/100k window and across seeds (`scripts/summarize_runs.py`).
2. 1000-battle `cares-rl test` runs of final models (`scripts/test_parallel.sh`, `finish_batch.sh`).
   `--eval_seed` does **not** make battles repeat, because the server generates the teams.
3. Diagnostics on the best run (`scripts/diagnose_batch.sh` → `diagnostics/<batch>/`):
   - `audit_policy.py --hand_plays <kind>`: the hand policy takes over one kind of decision, and the
     win-rate gain is what the agent's mistakes of that kind cost.
   - `battle_log.py`: a turn-by-turn log of each loss.
4. `scripts/check_submission.py`: the env file in a clean copy of the staff package (commit `8d0a390`),
   plays battles, and checks the checkpoint's input/output sizes. Every batch since v6 passed.

## 7. How the design got to v8

The scripted probe (2026-10-12) ran hand-written rules with no learning. The best rules reach ≈95.6%,
so they set the targets. From then on, each version gave the agent what a rule needed, then made
it easier to learn.

| Version (commit) | Change | Result vs max |
|---|---|---|
| v1 first attempt (`b823486`), smart-switch (`7ca3171`) | 4 move actions, 12 move features, small damage reward; then forced switch by a type-matchup rule | ≈50% (tests 49, 57); random moves level |
| v2 `state-reward-v2` (`563bda4`) | 22 features (available flags, relative damage, HP, Pokémon left, speed, type threat); reward 0.5 / 0.25 / 0.2 / 1 | ≈86% (tests 77–87); scripted ceiling for these actions ≈89% |
| v3 (`b6357ec`) | + 6 switch actions, agent picks forced switches; **predicted opponent move**; 68 features; illegal penalty 0.1 → 0.5 | ≈90–92% (best test 94/100) |
| v4 (`3bd8547`) | + 4 tera actions, 76 features | ≈90–91%: tera almost never used (penalised illegal teras taught it to avoid them) |
| v4r (`4d78f32`) | tera actions fall back to the plain move; **tera bonus** | +1–2 pts at matched steps; tera used 0.8–1.0/battle |
| v4s (`c214910`) | **switch / setup / heal bonuses** = hand-policy rules | 1M: 92.8%; switch learned, setup not (5% of hand's), heals mostly from slot 0 |
| v5 (`09f89ec`) | slot-independent **heal (14) and setup (15)** actions, 78 features | 1M tests 93.1% pooled; setup learned but heal/setup overused; warm start from v4 worse |
| v6 (`5e3d59c`) | **macro actions 16–19** (best attack, best tera, safest / matchup switch), 84 features; tables embedded (submittable) | 1M tests **94.2%**; diagnostics: losing ~1.4 pts on switching, ~1.5 on heal timing |
| v7 (`e88fffb`, `9d4478a`) | 12 reward variants × 2 seeds (switch-back cost, heal cost, weights...); 2M-step runs; 128-wide net | **no gain**: variants differ less than seeds (ANOVA F 0.77); 2M ≤ 1M; width 128 worse |
| **v8** (`095b7cb`) | **damage from real stats as fractions of HP** (30 new features, 114 total), opponent immunity abilities (bug: best attack used Earthquake into Levitate), best attack by real damage | 1M tests **95.5%** pooled (s10/20/30: 95.5/95.4/95.6) |

**Lessons behind the design:**
- **Measure a rule's value before building it.** `scripted_probe.py` showed setup is worth ~1.5–2 pts,
  healing only on neutral hits (+1.2), and that hazards, status moves, priority finishing and neutral
  switches are worth ~0.
- **Per-slot outputs don't share learning**, so a play that can sit in any slot gets its own action (v5, v6).
- **Plays with delayed payoff need shaping.** Tera and setup barely appear without bonuses.
- **Reward shaping saturated at v6/v7**, and v8's gain came from *information*: before v8 the agent
  had no notion of how much HP a hit takes, so nothing to time heals and switches by.
- **Things that didn't help:** wider networks (256, 128), dueling DQN (diverged), batch 128 (= 32),
  2M steps, warm starts from older networks, end ε 0.02, a 200k buffer.

**What's still lost (v8-s30 diagnostics, `672e499`, 3000 battles each):** agent alone 95.1%, hand policy
alone 97.1%. The gain from hand takeover is heal +1.1, attack +0.9, switch/forced +0.7, tera +0.2,
setup −0.4 (the agent's setups now beat the rule). Agent losses are less close and have twice the
can't-move turns (status). Different seeds play very different styles (s10 heals 8×/battle, s30 1.3×)
at the same win rate.

## 8. Scripts (`scripts/`, not part of the submission)

| Script | Purpose |
|---|---|
| `train_parallel.sh` | launch a batch on the VM (run table, RULES, watchdog/resume, test + push + shutdown); its comments are a running log of every batch |
| `start_run.sh` | starts runs one at a time (account names are stamped to the second; a clash gives "Agent is not challenging") |
| `test_parallel.sh`, `finish_batch.sh` | grading-style tests (100 or 1000 battles), copy runs into `runs/`, commit, push |
| `summarize_runs.py` | windowed / pooled training win rates and counters per run or variant |
| `progress.sh` | live progress, eval win rates, memory |
| `check_env.py` | random-action sanity checks plus the hand-written policies' win rates on the observation |
| `scripted_probe.py` | rule variants (base, A ... K) with no training |
| `check_reward.py`, `check_damage.py` | reward variants on scripted play; damage estimate vs actual damage |
| `audit_policy.py`, `battle_log.py`, `diagnose_batch.sh` | agent vs hand-policy comparison, loss logs |
| `check_submission.py` | env file standalone in a clean staff package, plus checkpoint sizes |
| `build_strongest_moves.py` | regenerates the embedded tables (keeps CRLF) |
| `continue_run.py`, `expand_checkpoint.py`, `write_config.py` | carry a run on, widen a v4 checkpoint for v5, configs for other network widths |
| `save_runs.sh`, `setup_gcp_vm.sh` | copy runs into the repo; one-time VM setup |

## 9. Practical gotchas

- **Showdown server first:** `cd ~/compsys726/pokemon-showdown && node pokemon-showdown start --no-security`.
- **`unset PYTHONPATH`** before running anything: the user's shell profile adds ROS Humble Python 3.10
  paths that break imports.
- `cares-rl test` asks two questions (run name, use CPU). Non-interactively: `printf '\ny\n' | cares-rl test ...`.
  Test output goes to a new folder in `~/cares_rl_logs/DQN/`. Results are in `<seed>/data/eval.csv`, `win` column.
- Grading-style test of the submission:
  `cares-rl test --data_path ~/compsys726/submission/zyao804/DQN-random-max-v8-s30-26_10_21_11-27-12 --eval_seed 10 --episodes 100`.
- The env file has CRLF line endings. Keep them when editing.
- If the submission changes: rebuild the `submission` branch commit **and** the upload folder, then
  rerun the clean-venv test (README "Step 2").
- Don't `pkill -f <pattern>` from a shell whose own command line contains that pattern (it kills
  itself). Anchor it, e.g. `pkill -f '^node pokemon-showdown'`.
- VM: if a zone stockout stops it restarting, change the machine type while it is stopped. The local
  gcloud account can't see the VM, so VM commands are run by the user.

## 10. Known weaknesses of the submitted file (as documentation)

- The bonus defaults don't match the trained values (§5 warning).
- Comments refer to version names (v4s, v6-1m...) and to scripts that aren't in the submission
  (`check_env.py`, `scripted_probe.py`). This document is the key to those references.
- The v7 reward knobs remain in the file although the sweep found no effect.
