import tutorial_env
import tutorial_action_mask_env
from pettingzoo.utils import wrappers


def make_env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    # internal_render_mode = render_mode if render_mode != "ansi" else "human"
    # env = raw_env(render_mode=internal_render_mode)

    # env = tutorial_env.CustomEnvironment()
    env = tutorial_action_mask_env.CustomActionMaskedEnvironment(render_mode=render_mode)

    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env) # only works for AEC

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)

    return env

env = make_env(render_mode="human")
observations, infos = env.reset()

while env.agent_names:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agent_names}
    print(actions)

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
