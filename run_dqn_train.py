import json
import os
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
from dataclasses import asdict

import torch
import torch.optim as optim
from tqdm import tqdm

from src.env_utils import MovementActions, GazeActions, index_to_action
from src.belief import DiscreteStateFilter
from src.policies import *
from src.markov_game_env import MarkovGameEnvironment, InitialState
from src.dqn import *
from src.plotting import *


env =  MarkovGameEnvironment(fully_observable=False, render_mode="none", initial_state=InitialState.UNIFORM)

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"device: {device}")

###########

params = DQNParams()
print(params)

# Get number of actions from gym action space
n_actions = len(MovementActions) * len(GazeActions)
# print(f"n_actions: {n_actions}")
# Get size dimensions of belief space
observations, info = env.reset()

n_observations = env.num_rows * env.num_cols
policy_net = ConvDQN(n_observations, n_actions, size="medium").to(device)
target_net = ConvDQN(n_observations, n_actions, size="medium").to(device)
target_net.load_state_dict(policy_net.state_dict())

print(f"{sum(p.numel() for p in policy_net.parameters() if p.requires_grad)} trainable parameters")
print(policy_net)
print()

optimizer = optim.AdamW(policy_net.parameters(), lr=params.LR, amsgrad=True)
memory = ReplayMemory(100_000)

global_steps_done = 0
episode_rewards = []
episode_timesteps = []
episode_avg_losses = [] # the average loss of each episode

if torch.cuda.is_available():
    num_episodes = 300
    num_saves = 5
elif torch.backends.mps.is_available():
    # macbook
    num_episodes = 10
    num_saves = 5
else:
    # pc
    num_episodes = 10
    num_saves = 5
episodes_per_save = num_episodes // num_saves

print(f"num_episodes: {num_episodes}")
print(f"num_saves: {num_saves}")
print(f"episodes_per_save: {episodes_per_save}")
print()

save_dir = create_save_directory()
random_policy = RandomPolicy(env.action_space(env.agent_names[0]))

for i in range(num_episodes):
# for i in tqdm(range(num_episodes)):
    belief_filter = DiscreteStateFilter(env.num_rows, env.num_cols)

    # Initialize the environment and get its state
    observations, infos = env.reset()
    bootstrap_policy = Policy.get("heuristic", env)

    belief_state = create_dqn_belief_state(observations["you"], belief_filter.get_belief(), device)
    # print(belief_state.shape)
    # print(belief_state)

    episode_memory = ReplayMemory(1000)
    while env.agent_names:
        print("timestep", env.timestep)
        env.print_locations()
        actions = {}
        for agent in env.agent_names:
            if agent == "you":
                tensor_action = select_action(global_steps_done, observations[agent], bootstrap_policy, belief_state, policy_net, params, env, device)
                actions[agent] = tensor_action.item()
            else:
                actions[agent] = random_policy.get_action(observations[agent])
        global_steps_done += 1

        # observation, reward, terminated, truncated, _ = env.step(action.item())
        observations, rewards, terminations, truncations, infos = env.step(actions)
        your_action = index_to_action(actions["you"])
        belief_filter.update(observations["you"], your_action)

        if "you" in rewards:
            reward = torch.tensor([rewards["you"]], device=device)
        else:
            print("WTF!")
            reward = torch.tensor([0], device=device)
        
        done = any(terminations.values()) or any(truncations.values()) # TODO: check if this is correct

        if done:
            next_belief_state = None
        else:
            next_belief_state = create_dqn_belief_state(observations["you"], belief_filter.get_belief(), device)

        episode_memory.push(belief_state, tensor_action, next_belief_state, reward)

        # Move to the next state
        belief_state = next_belief_state

        you_win = infos[env.you.name]["win"]
        opp_win = infos[env.opp.name]["win"]


    if you_win or opp_win:
        # Move episode_memory to true memory, with reward shaping
        for transition in episode_memory.memory:
            shaped_reward = shape_reward(transition.reward, you_win).to(device)
            memory.push(transition.state, transition.action, transition.next_state, shaped_reward)
    else:
        print(f"skipping episode {i} because neither agent won")

    # print(len(memory))
    if len(memory) >= params.BATCH_SIZE:
        #####
        # Perform one step of the optimization (on the policy network)
        loss = optimize_model(optimizer, policy_net, target_net, memory, params, device)
        if loss is None:
            loss = 0
            print(f"WARNING: loss is None for episode {i}")
        episode_avg_losses.append(loss)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*params.TAU + target_net_state_dict[key]*(1-params.TAU)
        target_net.load_state_dict(target_net_state_dict)
        #####
    else:
        loss = 0
        episode_avg_losses.append(loss)


    discounted_reward = rewards["you"] * params.GAMMA ** env.timestep
    episode_rewards.append(discounted_reward)
    episode_timesteps.append(env.timestep)

    plot_rewards(episode_rewards)
    plot_loss(episode_avg_losses)
    # print(f"""DEBUG: i={i}, done={done}, rewards["you"]={rewards["you"]}, env.timestep={env.timestep}, discounted_reward={discounted_reward}""")

    if (i + 1) % episodes_per_save == 0:
        save_file = os.path.join(save_dir, f'policy_{i}.pth')
        save_weights(policy_net, save_file)
        mean_reward = np.mean(episode_rewards[-episodes_per_save:])
        print(f'Mean reward for last {episodes_per_save} episodes: {mean_reward}')


# should be the same as the last checkpoint, but just for convenience:
save_file = os.path.join(save_dir, f'policy_final.pth')
save_weights(policy_net, save_file)

plot_rewards(episode_rewards, show_result=True, save_dir=save_dir)
plot_loss(episode_avg_losses, show_result=True, save_dir=save_dir)

rewards_path = os.path.join(save_dir, 'dqn.json')
with open(rewards_path, 'w') as f:
    json.dump({
        'metadata': {
            'num_episodes': num_episodes,
            **asdict(params),
        },
        'episode_rewards': episode_rewards,
        'episode_timesteps': episode_timesteps,
    }, f, indent=4)
print(f'Episode rewards saved to {rewards_path}')

if not is_databricks_cluster():
    plt.ioff()
    plt.show()

print("Done!")
