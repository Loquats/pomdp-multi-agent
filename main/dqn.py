import json
from datetime import datetime
import os
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from custom_utils import MovementActions, GazeActions, index_to_action
from belief import DiscreteStateFilter
from policies import RandomPolicy
from markov_game_env import MarkovGameEnvironment, InitialState

env =  MarkovGameEnvironment(fully_observable=False, render_mode="none", initial_state=InitialState.UNIFORM)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"device: {device}")

###########

# state = belief, next_state = next_belief
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
###########

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

###########

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = len(MovementActions) * len(GazeActions)
# print(f"n_actions: {n_actions}")
# Get size dimensions of belief space
observations, info = env.reset()


n_observations = env.num_rows * env.num_cols
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space(env.agent_names[0]).sample()]], device=device, dtype=torch.long)


episode_rewards = []
episode_timesteps = []


def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


############


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 100
    num_saves = 10
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

def is_databricks_cluster():
    """Is the code running on a Databricks cluster?"""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def create_save_directory():
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    if is_databricks_cluster():
        save_dir = f'/Volumes/datasets/andyzhang/hw/dqn_{timestamp}'
    else:
        save_dir = f'results/dqn_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_weights(net, filepath):
    torch.save(net.state_dict(), filepath)
    print(f"saved weights: {save_file}")

save_dir = create_save_directory()
random_policy = RandomPolicy(env.action_space(env.agent_names[0]))

for i in range(num_episodes):
# for i in tqdm(range(num_episodes)):
    belief_filter = DiscreteStateFilter(env.num_rows, env.num_cols)

    # Initialize the environment and get its state
    observations, infos = env.reset()
    belief_state = torch.tensor(belief_filter.get_belief_vector(), dtype=torch.float32, device=device).unsqueeze(0)
    while env.agent_names:
        actions = {}
        for agent in env.agent_names:
            if agent == "you":
                tensor_action = select_action(belief_state)
                actions[agent] = tensor_action.item()
            else:
                actions[agent] = random_policy.get_action(observations[agent])

        # observation, reward, terminated, truncated, _ = env.step(action.item())
        observations, rewards, terminations, truncations, infos = env.step(actions)
        your_action = index_to_action(actions["you"])
        belief_filter.update(observations["you"], your_action)

        if "you" in rewards:
            reward = torch.tensor([rewards["you"]], device=device)
        else:
            print("WTF!")
            reward = 0
        
        done = any(terminations.values()) or any(truncations.values()) # TODO: check if this is correct

        if done:
            next_belief_state = None
        else:
            next_belief_state = torch.tensor(belief_filter.get_belief_vector(), dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(belief_state, tensor_action, next_belief_state, reward)

        # Move to the next state
        belief_state = next_belief_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

    discounted_reward = rewards["you"] * GAMMA ** env.timestep
    episode_rewards.append(discounted_reward)
    episode_timesteps.append(env.timestep)
    plot_rewards()
    # print(f"""DEBUG: i={i}, done={done}, rewards["you"]={rewards["you"]}, env.timestep={env.timestep}, discounted_reward={discounted_reward}""")

    if (i + 1) % episodes_per_save == 0:
        save_file = os.path.join(save_dir, f'policy_{i}.pth')
        save_weights(policy_net, save_file)

# should be the same as the last checkpoint, but just for convenience:
save_file = os.path.join(save_dir, f'policy_final.pth')
save_weights(policy_net, save_file)

# print('Complete')
plot_rewards(show_result=True)
plt.ioff()
# plt.show()
plt.savefig(os.path.join(save_dir, f"rewards.png"))

rewards_path = os.path.join(save_dir, 'dqn.json')
with open(rewards_path, 'w') as f:
    json.dump({
        'metadata': {
            'num_episodes': num_episodes,
            'batch_size': BATCH_SIZE,
            'gamma': GAMMA,
            'eps_start': EPS_START,
            'eps_end': EPS_END,
            'eps_decay': EPS_DECAY,
            'tau': TAU,
            'learning_rate': LR
        },
        'episode_rewards': episode_rewards,
        'episode_timesteps': episode_timesteps,
    }, f, indent=4)
print(f'Episode rewards saved to {rewards_path}')

print("Done!")
