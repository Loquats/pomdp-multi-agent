from pettingzoo.mpe import simple_v3

env = simple_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

prev = 1

while env.agents:
    # this is where you would insert your policy
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    """
    0: do nothing

    agent is GREY
    target is RED and will move away from agent once agent is close enough
    """
    actions = {agent: (prev + 1) % 5 for agent in env.agents}
    prev += 1
    print(actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("obs:", observations)
    print("reward:", rewards)
    print("termination:", terminations)
    print("truncation:", truncations)
    print("info:", infos)

    # break
env.close()
