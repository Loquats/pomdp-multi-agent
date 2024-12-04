from collections import namedtuple, deque
import random
import math
from dataclasses import dataclass
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DQNParams:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    # BATCH_SIZE = 128
    BATCH_SIZE = 2**14 # 16,384 
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.25
    EPS_DECAY = 100000
    TAU = 0.005
    # LR = 1e-4
    LR = 3e-5

# state = belief, next_state = next_belief
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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

import torch.nn as nn
import torch.nn.functional as F


class ConvDQN(nn.Module):
    def __init__(self, n_observations, n_actions, size):
        """
        62706 trainable parameters
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=5, kernel_size=3, padding="same")
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 6, 3, padding="same")
        self.fc1 = nn.Linear(6 * 10 * 20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        # print(0, x.shape)
        # x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        # print(1, x.shape)
        # x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv2(x))
        # print(2, x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(3, x.shape)
        # print(self.fc1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleDQN(nn.Module):

    def __init__(self, n_observations, n_actions, size="small"):
        """
        small: 44820 trainable parameters
        medium: 86932 trainable parameters
        large: 375828 trainable parameters
        """
        super(SimpleDQN, self).__init__()
        if size == "small":
            layer_1_size = 128
            layer_2_size = 128
        elif size == "medium":
            layer_1_size = 256
            layer_2_size = 128
        elif size == "large":
            layer_1_size = 512
            layer_2_size = 512
        else:
            raise Exception(f"invalid size {size}")
        self.layer1 = nn.Linear(n_observations, layer_1_size)
        self.layer2 = nn.Linear(layer_1_size, layer_2_size)
        self.layer3 = nn.Linear(layer_2_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def get_eps_threshold(global_steps_done, params):
    return params.EPS_END + (params.EPS_START - params.EPS_END) * math.exp(-1. * global_steps_done / params.EPS_DECAY)

def select_action(global_steps_done, observation, prev_action, bootstrap_policy, state, policy_net, params, env, device):
    sample = random.random()
    eps_threshold = get_eps_threshold(global_steps_done, params)
    
    # get bootstrap action and maybe don't use it. Must always feed observaiton to update belief though!
    bootstrap_action = bootstrap_policy.get_action(observation, prev_action)
    if sample > eps_threshold:
        return get_policy_action(state, policy_net)
    else:
        # get random action
        # return torch.tensor([[env.action_space(env.agent_names[0]).sample()]], device=device, dtype=torch.long)
        # return bootstrap action
        return torch.tensor([[bootstrap_action]], device=device, dtype=torch.long)
    
def get_policy_action(state, policy_net):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1).indices.view(1, 1)

def optimize_model(optimizer, policy_net, target_net, memory, params, device):
    if len(memory) < params.BATCH_SIZE:
        return
    transitions = memory.sample(params.BATCH_SIZE)
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
    next_state_values = torch.zeros(params.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * params.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss

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
    print(f"saved weights: {filepath}")

def is_databricks_cluster():
    """Is the code running on a Databricks cluster?"""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

reward_map = {
    -100: -3,
    0: 0,
    1: 1,
    100: 3,
}

def shape_reward(reward_tensor, you_win):
    reward = reward_map[reward_tensor.item()]

    if you_win:
        reward += 2
    else:
        reward -= 2

    # assert reward != 0
    
    return torch.tensor([reward])
    
def create_dqn_belief_state(observation, belief, device):
    """
    Return a tensor with shape (1, 2, *(belief.shape)), eg. (1, 2, 10, 20)
    First channel is the current agent's location
    Second channel is the belief
    1 is the batch dimension
    """
    your_row, your_col, _, _ = observation

    channel2 = torch.from_numpy(belief).float()
    channel1 = torch.zeros_like(channel2)
    channel1[your_row, your_col] = 1.0
    
    tensor = torch.stack([channel1, channel2], dim=0)
    
    # Move to device and add batch dimension
    return tensor.unsqueeze(0).to(device)
