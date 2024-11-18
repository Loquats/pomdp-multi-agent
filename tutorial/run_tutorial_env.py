import tutorial_action_mask_env
from custom_utils import *
from policies import *
from pettingzoo.utils import wrappers

env = tutorial_action_mask_env.make_env(render_mode="pygame")
observations, infos = env.reset()
print(env.action_space(env.agent_names[0]))
print(env.action_space(env.agent_names[1]))
print(observations)

while env.agent_names:
    # this is where you would insert your policy
    actions = {agent: random_policy(env.action_space(agent)) for agent in env.agent_names}

    for agent, action in actions.items():
        print(agent, index_to_action(action))

    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print("obs:", observations)
    # print("reward:", rewards)
    # print("termination:", any(terminations.values()))
    # print("truncation:", any(truncations.values()))
    # print("info:", infos)

    # break
    # import pdb; pdb.set_trace()
    # import time; time.sleep(10000)
env.close()
