import random
from src.dqn import DQN, get_policy_action
from src.belief import DiscreteStateFilter
from src.env_utils import GazeActions, MovementActions, action_to_index, index_to_action
from abc import ABC, abstractmethod

import torch

def template_policy(observation, action_space, agent):
    pass

class Policy(ABC):

    @abstractmethod
    def get_action(self, observation):
        pass

    @staticmethod
    def get(name, env):
        if name == "random":
            return RandomPolicy(env.action_space())
        if name == "heuristic":
            return SamplingHeuristicPolicy(env.num_rows, env.num_cols)
        return BeliefDQNPolicy(name, env.num_rows, env.num_cols)
        

class RandomPolicy(Policy):

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()
    
    def __str__(self):
        return "RandomPolicy"


class SamplingHeuristicPolicy:
    """
    find which quadrant the opponent is in
    move in one of the directions towards the opponent (break ties randomly)
    gaze in the direction of the opponent (break ties randomly)
    """
    def __init__(self, num_rows, num_cols):
        self.belief_filter = DiscreteStateFilter(num_rows, num_cols)
        self.prev_action = None

    def __str__(self):
        return "SamplingHeuristicPolicy"

    def get_action(self, observation):
        # print(self.belief_filter) # for debugging, it is helpful to print the belief filter before the update happens
        if self.prev_action:
            self.belief_filter.update(observation, self.prev_action)

        # get an action based on the current belief
        my_row, my_col, opp_row, opp_col = observation
        target_row, target_col = self.belief_filter.sample()

        move_actions = []
        if my_row == target_row and my_col == target_col:
            move_actions.append(MovementActions.DO_NOTHING)
        if my_row < target_row:
            move_actions.append(MovementActions.S)
        if my_row > target_row:
            move_actions.append(MovementActions.N)
        if my_col < target_col:
            move_actions.append(MovementActions.E)
        if my_col > target_col:
            move_actions.append(MovementActions.W)

        gaze_actions = []
        if my_row < target_row and my_col < target_col:
            gaze_actions.append(GazeActions.SE)
        elif my_row < target_row and my_col > target_col:
            gaze_actions.append(GazeActions.SW)
        elif my_row > target_row and my_col > target_col:
            gaze_actions.append(GazeActions.NW)
        elif my_row > target_row and my_col < target_col:
            gaze_actions.append(GazeActions.NE)
        
        if not gaze_actions:
            gaze_actions = list(GazeActions)

        move_action = random.choice(move_actions)
        gaze_action = random.choice(gaze_actions)

        # print(f"move: {move_action}, gaze: {gaze_action}")

        self.prev_action = (move_action, gaze_action)
        return action_to_index(move_action, gaze_action)

class BeliefDQNPolicy(Policy):
    
    def __init__(self, filepath, num_rows, num_cols):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # print(f"BeliefDQNPolicy device: {self.device}")

        self.filepath = filepath

        n_actions = len(MovementActions) * len(GazeActions)
        n_observations = num_rows * num_cols

        if is_large_policy(filepath):
            size = "large"
        elif is_medium_policy(filepath):
            size = "medium"
        else:
            size = "small"
        self.policy_net = DQN(n_observations, n_actions, size=size).to(self.device)
        self.policy_net.load_state_dict(torch.load(self.filepath))

        self.belief_filter = DiscreteStateFilter(num_rows, num_cols)
        self.prev_action = None

    def __str__(self):
        return f"BeliefDQNPolicy({self.filepath})"
    
    def get_action(self, observation):
        # print(self.belief_filter) # for debugging, it is helpful to print the belief filter before the update happens
        if self.prev_action:
            self.belief_filter.update(observation, self.prev_action)

        belief_state = torch.tensor(self.belief_filter.get_belief_vector(), dtype=torch.float32, device=self.device).unsqueeze(0)
        action_index = get_policy_action(belief_state, self.policy_net).item()

        self.prev_action = index_to_action(action_index)
        return action_index

def is_medium_policy(path):
    return path in ["results/databricks/dqn_2024_12_02_00:05:51/policy_final.pth"]

def is_large_policy(path):
    return path in ["results/dqn_2024_12_02_00:31:12/policy_final.pth"]
