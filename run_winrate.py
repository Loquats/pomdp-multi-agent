from src.markov_game_env import *
from src.policies import *
from src.winrate import run_winrate

env = MarkovGameEnvironment(fully_observable=False, render_mode="none", initial_state=InitialState.UNIFORM)
env.reset()

# Always use Policy.get() to get a fresh policy. You cannot reuse stateful policies like the heuristic policy
opponent_policies = ["random"]
you_policies = ["useless"]
# you_policies = ["results/dqn_2024_12_03_00:56:28/policy_1499.pth"]
# you_policies = ["results/dqn_2024_12_02_00:31:12/policy_final.pth"]
# you_policies = ["results/dqn_2024_12_01_23:42:33/policy_final.pth"]
# you_policies = ["results/databricks/dqn_2024_12_02_00:05:51/policy_final.pth"]
# you_policies = ["heuristic"]

for opponent_policy in opponent_policies:
    for you_policy in you_policies:
        print(f"running winrate for {you_policy} vs {opponent_policy}")
        wins, timesteps = run_winrate(
            env=env,
            you_policy_name=you_policy,
            opp_policy_name=opponent_policy,
            verbose=True,
            num_iterations=100,
        )
