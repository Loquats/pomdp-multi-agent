from pettingzoo.mpe import simple_tag_v3
env = simple_tag_v3.env(render_mode='human')

env.reset()
i = 0

print(env.num_agents)
print(env.max_cycles)

for agent in env.agent_iter():
    print(i, type(agent), agent)
    observation, reward, termination, truncation, info = env.last()
    print("obs:", observation.shape)
    print("reward:", reward)
    print("termination:", termination)
    print("truncation:", truncation)
    print("info:", info)

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)

    i += 1
    # if i == 5:
    #     break

env.close()
