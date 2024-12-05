from tqdm import tqdm

from src.markov_game_env import *
from src.env_utils import *
from src.policies import *


def get_discounted_reward(rewards, gamma):
    discounted_reward = 0
    for t, reward in enumerate(rewards):
        discounted_reward += (gamma**t) * reward
    return discounted_reward


def run_winrate(env, you_policy_name, opp_policy_name, num_iterations=100, verbose=False, gamma=0.95):
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

    outcomes = []

    returns = {
        "you": [],
        "opp": [],
    }

    if num_iterations == "auto":
        if you_policy_name in SLOW_POLICIES or opp_policy_name in SLOW_POLICIES:
            num_iterations = 100
        else:
            num_iterations = 1000

    for _ in tqdm(range(num_iterations)):
        observations, infos = env.reset()

        policies = {
            "you": Policy.get(you_policy_name, env, observations["you"].my_row, observations["you"].my_col),
            "opp": Policy.get(opp_policy_name, env, observations["opp"].my_row, observations["opp"].my_col),
        }

        all_rewards = {
            "you": [],
            "opp": [],
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

            for agent in ["you", "opp"]:
                all_rewards[agent].append(rewards[agent])

        # print("all_rewards", all_rewards)

        you_win = infos[env.you.name]["win"]
        opp_win = infos[env.opp.name]["win"]
        if you_win and opp_win:
            wins["both"] += 1
            timesteps["both"].append(env.timestep)
            outcomes.append("both")
        elif you_win:
            wins["you"] += 1
            timesteps["you"].append(env.timestep)
            outcomes.append("you")
        elif opp_win:
            wins["opp"] += 1
            timesteps["opp"].append(env.timestep)
            outcomes.append("opp")
        else:
            wins["neither"] += 1
            timesteps["neither"].append(env.timestep)
            outcomes.append("neither")
        env.close()

        if verbose:
            print(wins)
        
        for agent in ["you", "opp"]:
            returns[agent].append(get_discounted_reward(all_rewards[agent], gamma))

    if verbose:
        print(wins)
        for outcome in wins.keys():
            print(f"winner is {outcome}: {wins[outcome] / num_iterations}")

        for outcome in timesteps.keys():
            print(f"mean timesteps for {outcome}: {np.mean(timesteps[outcome])}")

        print(f"mean timesteps overall: {np.mean(timesteps['you'] + timesteps['opp'] + timesteps['both'] + timesteps['neither'])}")

    assert sum(wins.values()) == num_iterations



    results = {
        # key inputs
        "num_iterations": num_iterations,
        "gamma": gamma,
        "you_policy_name": you_policy_name,
        "opp_policy_name": opp_policy_name,
        # key outputs
        "wins": wins,
        "winrates": {outcome: (num / num_iterations) for outcome, num in wins.items()},
        "average_return": {agent: np.mean(returns) for agent, returns in returns.items()},
        # detailed debug info
        "outcomes": outcomes,
        "timesteps": timesteps,
        "returns": returns,
    }

    return results
