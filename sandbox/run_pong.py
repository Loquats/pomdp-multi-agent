from pettingzoo.butterfly import cooperative_pong_v5

env = cooperative_pong_v5.env(render_mode="human")
env.reset(seed=42)

i = 0
for agent in env.agent_iter():
    print(type(agent), agent)
    observation, reward, termination, truncation, info = env.last()
    print(observation.shape)
    print(reward)
    print(termination)
    print(truncation)
    print(info)

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)

    i += 1
    if i == 5:
        break
env.close()
print("done")