from src.markov_game_env import MarkovGameEnvironment, InitialState
from src.env_utils import *
from src.policies import *

env = MarkovGameEnvironment(fully_observable=False, render_mode="pygame", initial_state=InitialState.UNIFORM)
# as a sanity check, run with full observability:
# env = MarkovGameEnvironment(fully_observable=True, render_mode="pygame")
observations, infos = env.reset()
print(f"Agent names: {env.agent_names}")
print("Action spaces:")
print(env.action_space(env.agent_names[0]))
print(env.action_space(env.agent_names[1]))
print(f"Observations: {observations}")

policies = {
    "you": Policy.get("results/dqn_2024_12_02_00:31:12/policy_final.pth", env),
    # "you": Policy.get("results/databricks/dqn_2024_12_02_00:05:51/policy_final.pth", env),
    "opp": Policy.get("random", env),
}

while env.agent_names:
    # env.print_locations() # helpful for debugging weird policies
    actions = {agent: policies[agent].get_action(observations[agent]) for agent in env.agent_names}

    for agent, action in actions.items():
        print(agent, index_to_action(action))

    observations, rewards, terminations, truncations, infos = env.step(actions)

    you_win = infos[env.you.name]["win"]
    opp_win = infos[env.opp.name]["win"]
    if you_win and opp_win:
        print("both win")
    elif you_win:
        print("you win")
    elif opp_win:
        print("opp win")
    else:
        print("neither win")

env.close()
