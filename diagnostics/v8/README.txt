v8 diagnostics, 2026-10-22: network v8-s30 (final model, 956/1000 in its test)
/home/rossyao10/cares_rl_logs/DQN/DQN-random-max-v8-s30-26_10_21_11-27-12/30/models/final/DQN_checkpoint.pth

win rate with the hand-written policy taking over one kind of decision (scripts/audit_policy.py):
  none    2854 = 95.1% [94.3, 95.8]  
  all     2913 = 97.1% [96.4, 97.6]  22.29 steps a battle, 8.45 of them different from the agent's choice
  forced  2874 = 95.8% [95.0, 96.5]  1.74 steps a battle, 0.53 of them different from the agent's choice
  switch  2874 = 95.8% [95.0, 96.5]  5.94 steps a battle, 4.64 of them different from the agent's choice
  heal    2887 = 96.2% [95.5, 96.9]  1.73 steps a battle, 1.40 of them different from the agent's choice
  setup   2840 = 94.7% [93.8, 95.4]  3.84 steps a battle, 1.33 of them different from the agent's choice
  tera    2860 = 95.3% [94.5, 96.0]  1.73 steps a battle, 1.25 of them different from the agent's choice
  attack  2879 = 96.0% [95.2, 96.6]  10.67 steps a battle, 0.69 of them different from the agent's choice

battle logs (scripts/battle_log.py):
  hand: won 2894 = 96.5% [95.7, 97.1]
  agent: won 2860 = 95.3% [94.5, 96.0]
