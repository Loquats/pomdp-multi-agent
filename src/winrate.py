from tqdm import tqdm

from src.markov_game_env import *
from src.env_utils import *
from src.policies import *

def run_winrate(env, you_policy_name, opp_policy_name, num_iterations=1000, verbose=False):
    wins = {
        "you": 0,
        "opp": 0,
        "both": 0,
        "neither": 0,
    }

    timesteps = {
        "you": [],
        "opp": [],
        "both": [],
        "neither": [],
    }

    for i in tqdm(range(num_iterations)):
        observations, infos = env.reset()

        policies = {
            # "you": Policy.get(you_policy_name, env),
            "you": PolicyWithRollouts(observations["you"].my_row, observations["you"].my_col, env.num_rows, env.num_cols, depth=15, num_rollouts=100),
            "opp": Policy.get(opp_policy_name, env),
        }

        prev_action = None
        while env.agent_names:
            # env.print_locations() # helpful for debugging weird policies
            actions = {agent: policies[agent].get_action(observations[agent], prev_action) for agent in env.agent_names}
            prev_action = actions["you"]

            actions = {agent: action_to_index(action) for agent, action in actions.items()}

            if verbose:
                print(prev_action)

            observations, rewards, terminations, truncations, infos = env.step(actions)

        you_win = infos[env.you.name]["win"]
        opp_win = infos[env.opp.name]["win"]
        if you_win and opp_win:
            wins["both"] += 1
            timesteps["both"].append(env.timestep)
        elif you_win:
            wins["you"] += 1
            timesteps["you"].append(env.timestep)
        elif opp_win:
            wins["opp"] += 1
            timesteps["opp"].append(env.timestep)
        else:
            wins["neither"] += 1
            timesteps["neither"].append(env.timestep)
        env.close()

        if verbose:
            print(wins)

    if verbose:
        print(wins)
        for outcome in wins.keys():
            print(f"winner is {outcome}: {wins[outcome] / num_iterations}")

        for outcome in timesteps.keys():
            print(f"mean timesteps for {outcome}: {np.mean(timesteps[outcome])}")

        print(f"mean timesteps overall: {np.mean(timesteps['you'] + timesteps['opp'] + timesteps['both'] + timesteps['neither'])}")

    assert sum(wins.values()) == num_iterations

    return wins, timesteps
