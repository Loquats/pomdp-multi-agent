import markov_game_env as markov_game_env
from custom_utils import *
from policies import *
from tqdm import tqdm
from markov_game_env import MarkovGameEnvironment, InitialState

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

num_iterations = 1000
print(f"running {num_iterations} iterations")

for i in tqdm(range(num_iterations)):
    env = MarkovGameEnvironment(fully_observable=False, render_mode="none", initial_state=InitialState.UNIFORM)
    observations, infos = env.reset()

    policies = {
        # "you": RandomPolicy(env.action_space(env.agent_names[0])),
        "you": SamplingHeuristicPolicy(env.num_rows, env.num_cols, env.action_space(env.agent_names[0])),
        "opp": RandomPolicy(env.action_space(env.agent_names[1])),
    }

    while env.agent_names:
        actions = {agent: policies[agent].get_action(observations[agent]) for agent in env.agent_names}
        # print(policies["you"].belief_filter)
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
    # print(f"you win: {you_win}, opp win: {opp_win}")
    env.close()

print(wins)
for outcome in wins.keys():
    print(f"winner is {outcome}: {wins[outcome] / num_iterations}")

for outcome in timesteps.keys():
    print(f"mean timesteps for {outcome}: {np.mean(timesteps[outcome])}")

print(f"mean timesteps overall: {np.mean(timesteps['you'] + timesteps['opp'] + timesteps['both'] + timesteps['neither'])}")

assert sum(wins.values()) == num_iterations
