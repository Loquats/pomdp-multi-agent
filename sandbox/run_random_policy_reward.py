from pettingzoo.utils import average_total_reward
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
average_total_reward(env, max_episodes=100, max_steps=10000000000)