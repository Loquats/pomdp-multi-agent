from dataclasses import dataclass
from enum import Enum
import random

from src.belief import *
from src.env_utils import *

# class Belief:
#     # b(s)

#     def __init__(self, my_row, my_col, opp_distribution):
#         self.my_row = my_row
#         self.my_col = my_col
#         self.opp_distribution = opp_distribution

#     def get_prob(self, state):
#         if state.my_row == self.my_row and state.my_col == self.my_col:
#             return self.opp_distribution[state.opp_row,state.opp_col]
#         else:
#             return 0
    
#     def sample_state(self):
#         return State(my_row, my_col, )
        
#     def __eq__(self, other):
#         return self.my_row == other.my_row and self.my_col == other.my_col and np.all(self.opp_distribution == other.opp_distribution)
        
class BeliefStateMDP:
    GAZE_REWARD = 1
    NO_GAZE_REWARD = 0

    def __init__(self, num_rows, num_cols, gamma=0.95):
        self.num_rows = num_rows
        self.num_cols = num_cols
        # self.dsf = dsf
        self.gamma = gamma

    def in_gaze_box(self, state, action):
        """
        action.move does not affect in_gaze_box because state is actually next_state,
        after the move action has been resolved
        """
        return in_gaze_box(state.my_row, state.my_col, action.gaze, state.opp_row, state.opp_col, self.num_rows, self.num_cols)

    # def belief(self, state: State):
    #     """
    #     b(s), the probability of this state, given our current belief
    #     use dsf to get probability
    #     """
    #     if state.my_row == self.dsf.my_row and state.my_col == self.dsf.my_col:
    #         return self.dsf.get_belief()[state.opp_row,state.opp_col]
    #     else:
    #         return 0
        
    def state_reward(self, state, action):
        """
        R(s,a)
        This is not the environment's reward function, but a proxy reward function.
        Make a simplifying assumption because we don't want to keep track of how many times I've seen the opponent.
        That would increase the state space annoyingly by 3x.

        TODO: should we use 1 or -1 as the no-gaze reward? Probably doesn't matter too much
        """
        if self.in_gaze_box(state, action):
            return self.GAZE_REWARD
        else:
            return self.NO_GAZE_REWARD

    def belief_reward(self, dsf_belief, action):
        """
        R(b,a) where b is self.belief function
        Instead of summing over all states, we only need to sum over nonzero states where
        state.my_row == dsf_belief.my_row and state.my_col == dsf_belief.my_col
        """
        # belief = self.belief(state)
        reward_gaze_mask = env_utils.get_gaze_mask(
            dsf_belief.my_row, dsf_belief.my_col, action.gaze,
            self.num_rows, self.num_cols,
            pos_value=self.GAZE_REWARD, neg_value=self.NO_GAZE_REWARD)
        
        return np.sum(reward_gaze_mask * dsf_belief.opp_belief)
    
    def move(self, row, col, move_action):
        if move_action == MovementActions.N and row > 0:
            return row - 1, col
        elif move_action == MovementActions.E and col < self.num_cols - 1:
            return row, col + 1
        elif move_action == MovementActions.S and row < self.num_rows - 1:
            return row + 1, col
        elif move_action == MovementActions.W and col > 0:
            return row, col - 1
        else:
            return row, col
        
    def possible_next_positions(self, row, col):
        moves = [(row, col)]
        if row > 0:
            moves.append((row - 1, col))
        if col > 0:
            moves.append((row, col - 1))
        if col < self.num_cols - 1:
            moves.append((row, col + 1))
        if row < self.num_rows - 1:
            moves.append((row + 1, col))
        return moves

    def possible_next_states(self, state, action):
        """
        Generative state transition model
        """
        my_next_row, my_next_col = self.move(state.my_row, state.my_col, action.move)
        next_states = []
        for opp_next_row, opp_next_col in self.possible_next_positions(state.opp_row, state.opp_col):
            next_states.append(State(my_next_row, my_next_col, opp_next_row, opp_next_col))
        return next_states
    
    def sample_next_state(self, state, action):
        return random.choice(self.possible_next_states(state, action))

    def state_transition_prob(self, next_state, state, action):
        """
        T(s' | s,a) probability distribution
        Don't need to implement this to solve the MDP
        """
        possible_next_states = self.possible_next_states(state, action)
        
        if state in possible_next_states:
            return 1.0 / len(possible_next_states)
        else:
            return 0.0
        # my_true_next_row, my_true_next_col = self.move(state.my_row, state.my_col, action.move)
        # opp_next_positions = self.possible_next_positions(state.opp_row, state.opp_col)
        # if next_state.my_row == my_true_next_row and next_state.my_col == my_true_next_col:
        #     if (next_state.opp_row, next_state.opp_col) in opp_next_positions:
        #         return 1.0/len(opp_next_positions)
        # else:
        #     return 0.0

    def belief_transition_prob(self, action):
        pass

    def get_observation(self, action, next_state):
        """
        AKA sample_observation
        Generative observation model
        there is only one possible observation, given an action and next_state
        """
        if self.in_gaze_box(next_state, action):
            return Observation(next_state.my_row, next_state.my_col, next_state.opp_row, next_state.opp_col)
        else:
            return Observation(next_state.my_row, next_state.my_col, -1, -1)

    def observation_prob(self, observation, action, next_state: State):
        return observation == self.possible_observation(action, next_state)
