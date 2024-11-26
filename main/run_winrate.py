import markov_game_env as markov_game_env
from custom_utils import *
from policies import *
from tqdm import tqdm

policy = random_policy

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
    env = markov_game_env.make_env(render_mode="none")
    observations, infos = env.reset()

    while env.agent_names:
        actions = {agent: policy(env.action_space(agent)) for agent in env.agent_names}
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
