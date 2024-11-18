import tutorial_action_mask_env
from custom_utils import *
from policies import *
from tqdm import tqdm
import json
policy = random_policy

def make_line(prev_observation, action, observation, reward):
    line = {
        "s": prev_observation,
        "a": action,
        "sp": observation,
        "r": reward,
    }
    return line

num_iterations = 1000
print(f"running {num_iterations} iterations")

for i in tqdm(range(num_iterations)):
    env = tutorial_action_mask_env.make_env(render_mode="none")
    observations, infos = env.reset()

    prev_observations = observations
    lines = []
    while env.agent_names:
        actions = {agent: policy(env.action_space(agent)) for agent in env.agent_names}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        line = make_line(prev_observations["you"], actions["you"].item(), observations["you"], rewards["you"])
        lines.append(line)

        prev_observations = observations

    with open(f'rollouts/random/{i}.txt', 'w') as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")

    env.close()
