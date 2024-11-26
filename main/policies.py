import random
from belief import DiscreteStateFilter
from custom_utils import GazeActions, MovementActions, action_to_index, index_to_action
from abc import ABC, abstractmethod

def template_policy(observation, action_space, agent):
    pass

class Policy(ABC):

    @abstractmethod
    def get_action(self, observation):
        pass


class RandomPolicy(Policy):

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()


class SamplingHeuristicPolicy:
    """
    find which quadrant the opponent is in
    move in one of the directions towards the opponent (break ties randomly)
    gaze in the direction of the opponent (break ties randomly)
    """
    def __init__(self, num_rows, num_cols, action_space):
        self.belief_filter = DiscreteStateFilter(num_rows, num_cols)
        self.action_space = action_space
        self.prev_action = None
    def get_action(self, observation):
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
