import tutorial_action_mask_env
from custom_utils import *
from pettingzoo.utils import wrappers

env = tutorial_action_mask_env.make_env(render_mode="human")
observations, infos = env.reset()

while env.agent_names:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agent_names}
    print(env.action_space(env.agent_names[0]))
    for agent, action in actions.items():
        print(agent, index_to_action(action))

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("obs:", observations)
    print("reward:", rewards)
    print("termination:", terminations)
    print("truncation:", truncations)
    print("info:", infos)

    # break
    # import pdb; pdb.set_trace()
    # import time; time.sleep(10000)
env.close()
