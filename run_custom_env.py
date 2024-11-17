import custom_env

env = custom_env.parallel_env(render_mode="human")
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

    # TODO: perturb action before env.step? or perturb using some pettingzoo feature like u_noise?

    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print("obs:", observations)
    # print("reward:", rewards)
    # print("termination:", terminations)
    # print("truncation:", truncations)
    # print("info:", infos)

    # break
env.close()
