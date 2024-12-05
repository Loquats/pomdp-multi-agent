import os
from datetime import datetime
import json

from src.markov_game_env import *
from src.policies import *
from src.winrate import run_winrate

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
you_policies = POLICIES
opponent_policies = POLICIES

for opponent_policy in opponent_policies:
    for you_policy in you_policies:
        print(f"running winrate for {you_policy} vs {opponent_policy}")
        results = run_winrate(
            env=env,
            you_policy_name=you_policy,
            opp_policy_name=opponent_policy,
            verbose=False,
            num_iterations=1,
            # num_iterations="auto",
        )

        for k, v in results.items():
            print(k, v)
        print()

        save_file = os.path.join(save_dir, f'{you_policy}_vs_{opponent_policy}.json')
        with open(save_file, 'w') as fp:
            json.dump(results, fp, indent=4)