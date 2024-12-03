from src.markov_game_env import MarkovGameEnvironment, InitialState
from src.env_utils import *
from src.policies import *
from pettingzoo.utils import wrappers
from src.belief import DiscreteStateFilter

env = MarkovGameEnvironment(fully_observable=False, render_mode="pygame", initial_state=InitialState.UNIFORM)
# as a sanity check, run with full observability:
# env = MarkovGameEnvironment(fully_observable=True, render_mode="pygame")
observations, infos = env.reset()
print(f"Agent names: {env.agent_names}")
print("Action spaces:")
print(env.action_space(env.agent_names[0]))
print(env.action_space(env.agent_names[1]))
print(f"Observations: {observations}")

belief_filter = DiscreteStateFilter(env.num_rows, env.num_cols)
print("initial belief (should be uniform):")
print(belief_filter)

policy = SamplingHeuristicPolicy(env.num_rows, env.num_cols)

while env.agent_names:
    # actions = {agent: random_policy(env.action_space(agent)) for agent in env.agent_names}
    actions = {}
    for agent in env.agent_names:
        if agent == "you":
            actions[agent] = policy.get_action(observations[agent])
        else:
            actions[agent] = random_policy(env.action_space(agent))

    for agent, action in actions.items():
        print(agent, index_to_action(action))

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if env.agent_names and env.agent_names[0] == "you":
        your_observation = observations[env.agent_names[0]]
        your_action = index_to_action(actions[env.agent_names[0]])
        belief_filter.update(your_observation, your_action)
        print(f"your observation: {your_observation}")
        print(f"your action: {your_action}")
        print(belief_filter.sample())
        print(belief_filter)

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

    # print("obs:", observations)
    # print("reward:", rewards)
    # print("termination:", any(terminations.values()))
    # print("truncation:", any(truncations.values()))
    # print("info:", infos)

    # break
    # import pdb; pdb.set_trace()
    # import time; time.sleep(10000)
env.close()
