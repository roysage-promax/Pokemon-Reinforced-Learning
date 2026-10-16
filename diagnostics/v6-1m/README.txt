v6-1m diagnostics, 2026-10-16: network v6-1m-s10 (final model, 948/1000 in its test)
/home/rossyao10/cares_rl_logs/DQN/DQN-random-max-v6-1m-s10-26_10_16_09-53-14/10/models/final/DQN_checkpoint.pth

win rate with the hand-written policy taking over one kind of decision (scripts/audit_policy.py):
  none    2807 = 93.6% [92.6, 94.4]  
  all     2872 = 95.7% [94.9, 96.4]  23.20 steps a battle, 7.31 of them different from the agent's choice
  forced  2826 = 94.2% [93.3, 95.0]  1.95 steps a battle, 0.28 of them different from the agent's choice
  switch  2851 = 95.0% [94.2, 95.8]  5.71 steps a battle, 4.30 of them different from the agent's choice
  heal    2854 = 95.1% [94.3, 95.8]  1.69 steps a battle, 1.28 of them different from the agent's choice
  setup   2812 = 93.7% [92.8, 94.5]  4.06 steps a battle, 1.10 of them different from the agent's choice
  tera    2816 = 93.9% [93.0, 94.7]  1.47 steps a battle, 1.06 of them different from the agent's choice
  attack  2828 = 94.3% [93.4, 95.0]  12.04 steps a battle, 0.37 of them different from the agent's choice

battle logs (scripts/battle_log.py):
  hand: won 2863 = 95.4% [94.6, 96.1]
  agent: won 2814 = 93.8% [92.9, 94.6]
