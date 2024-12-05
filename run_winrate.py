import os
from datetime import datetime
import json

from src.markov_game_env import *
from src.policies import *
from src.winrate import run_winrate

"""
full square run: /home/andy/aa228/pz/results/winrate/2024_12_05_00:51:18
fast policies 100 iters (15 runs): /home/andy/aa228/pz/results/winrate/2024_12_05_01:25:59
fast policies 1000 iters (15 runs): /home/andy/aa228/pz/results/winrate/2024_12_05_01:47:30
"""

def create_save_directory():
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    if is_databricks_cluster():
        save_dir = f'/Volumes/datasets/andyzhang/hw/winrate/{timestamp}'
    else:
        save_dir = f'results/winrate/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


save_dir = create_save_directory()

env = MarkovGameEnvironment(fully_observable=False, render_mode="none", initial_state=InitialState.UNIFORM)
observations, infos = env.reset()

# test policies
# for name in POLICIES:
#     print(Policy.get(name, env, 3, 4))
# 1/0

# Always use Policy.get() to get a fresh policy. You cannot reuse stateful policies like the heuristic policy
# you_policies = ["useless"]
# you_policies = ["results/dqn_2024_12_03_00:56:28/policy_1499.pth"]
# you_policies = ["results/dqn_2024_12_02_00:31:12/policy_final.pth"]
# you_policies = ["results/dqn_2024_12_01_23:42:33/policy_final.pth"]
# you_policies = ["results/databricks/dqn_2024_12_02_00:05:51/policy_final.pth"]
# you_policies = ["heuristic_sample"]
# opponent_policies = ["random"]
# opponent_policies = ["heuristic_sample"]

# you_policies = POLICIES
# opponent_policies = POLICIES

you_policies = FAST_POLICIES
opponent_policies = FAST_POLICIES

print("plan:")
num_runs = 0
for i, you_policy in enumerate(you_policies):
    for opponent_policy in opponent_policies[i:]: # only fill in upper diagonal, because if we have A vs. B, we don't need B vs. A
        print(f'{you_policy}_vs_{opponent_policy}')
        num_runs += 1
print(f"total {num_runs} runs") # expect 28 for 1 + ... + 7
print()
# 1/0

for i, you_policy in enumerate(you_policies):
    for opponent_policy in opponent_policies[i:]: # only fill in upper diagonal, because if we have A vs. B, we don't need B vs. A
        print(f"running winrate for {you_policy} vs {opponent_policy}")
        results = run_winrate(
            env=env,
            you_policy_name=you_policy,
            opp_policy_name=opponent_policy,
            verbose=False,
            num_iterations=1000,
            # num_iterations="auto",
        )

        for k, v in results.items():
            if k not in ["outcomes", "timesteps", "returns"]:
                print(k, v)
        print()

        save_file = os.path.join(save_dir, f'{you_policy}_vs_{opponent_policy}.json')
        with open(save_file, 'w') as fp:
            json.dump(results, fp, indent=4)