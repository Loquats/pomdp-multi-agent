import markov_game_env as markov_game_env
from pettingzoo.utils import wrappers
from src.env_utils import *


env = markov_game_env.make_env(render_mode="human")
observations, infos = env.reset()

print(env.agent_names)
for agent in env.agent_names:
    action = env.action_space(agent).sample()
    print(agent)
    print(action)
    print(env.action_space(agent))

for i in range(env.action_space(env.agent_names[0]).n):
    print(i, index_to_action(i))
