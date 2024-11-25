from tutorial_action_mask_env import MarkovGameEnvironment
from custom_utils import *
from policies import *
from pettingzoo.utils import wrappers
from belief import DiscreteStateFilter

env = MarkovGameEnvironment(fully_observable=False, render_mode="pygame")
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

while env.agent_names:
    actions = {agent: random_policy(env.action_space(agent)) for agent in env.agent_names}

    for agent, action in actions.items():
        print(agent, index_to_action(action))

    observations, rewards, terminations, truncations, infos = env.step(actions)

    your_observation = observations[env.agent_names[0]]
    your_action = index_to_action(actions[env.agent_names[0]])
    belief_filter.update(your_observation, your_action)
    print(f"your observation: {your_observation}")
    print(f"your action: {your_action}")
    print(belief_filter)

    # print("obs:", observations)
    # print("reward:", rewards)
    # print("termination:", any(terminations.values()))
    # print("truncation:", any(truncations.values()))
    # print("info:", infos)

    break
    # import pdb; pdb.set_trace()
    # import time; time.sleep(10000)
env.close()
