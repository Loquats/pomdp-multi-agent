import numpy as np
from src.env_utils import *

def validate(dsf, obs, action):
    new_row, new_col = dsf.move(action.move)

    assert new_row == obs.my_row
    assert new_col == obs.my_col

def update(dsf, obs, action):
    """
    returns a NEW dsf
    Update the belief based on the observation, and the action resulting in that observation.
    observation: (my_row, my_col, opponent_row, opponent_col)
        (opponent_row, opponent_col) is (-1, -1) if not observed
    action: 
    """
    try:
        validate(dsf, obs, action)
    except AssertionError as e:
        print(obs)
        print(action)
        print(dsf)
        raise e
    
    if obs.opp_row == -1 and obs.opp_col == -1:
        # print("CAN'T SEE OPPONENT")
        new_belief = np.zeros_like(dsf.opp_belief)
        for row in range(dsf.num_rows):
            for col in range(dsf.num_cols):
                # get valid recipients
                valid_recipients = []
                for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    recipient_row = row + drow
                    recipient_col = col + dcol

                    if dsf.is_valid_cell(recipient_row, recipient_col):
                        valid_recipients.append((recipient_row, recipient_col))

                # diffuse the mass
                parcel = dsf.opp_belief[row, col] / len(valid_recipients)
                for recipient_row, recipient_col in valid_recipients:
                    new_belief[recipient_row, recipient_col] += parcel

        # zero out the gaze box
        min_row, min_col, max_row, max_col = get_gaze_bounds(action.gaze, obs.my_row, obs.my_col, dsf.num_rows, dsf.num_cols)
        for row in range(min_row, max_row+1):
            for col in range(min_col, max_col+1):
                new_belief[row, col] = 0
        
        total = np.sum(new_belief)
        if total == 0:
            print("WTF. This is a real error you need to debug. Likely cause: you should not reuse a heuristic policy/DiscreteStateFilter across environments. Make a fresh heuristic policy instead!")
            print(obs)
            print(action)
            print("DSF gaze bounds", min_row, min_col, max_row, max_col)
            print(dsf)
            raise Exception("belief sum is 0")
        
        new_opp_belief = new_belief / total
    else:
        # print("CAN SEE OPPONENT")
        # we know exactly where the opponent is
        new_opp_belief = np.zeros_like(dsf.opp_belief)
        new_opp_belief[obs.opp_row, obs.opp_col] = 1.0

    return DiscreteStateFilter(obs.my_row, obs.my_col, dsf.num_rows, dsf.num_cols, opp_belief=new_opp_belief)

class DiscreteStateFilter:
    """
    this must be immutable to avoid bugs!
    """
    def __init__(self, my_row, my_col, num_rows, num_cols, opp_belief=None):
        if opp_belief is not None:
            self.opp_belief = opp_belief
            self.opp_belief.setflags(write=False)
        else:
            # initialize uniform probability
            self.opp_belief = np.ones((num_rows, num_cols)) / (num_rows * num_cols)
            self.opp_belief.setflags(write=False)
        self.num_rows, self.num_cols = num_rows, num_cols
        self.my_row = my_row
        self.my_col = my_col

    def is_valid_cell(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols

    def __str__(self):
        # Convert each row to space-separated strings with rounded numbers
        rows = []
        for row in self.opp_belief:
            # Round each number to 3 decimal places and join with spaces
            row_str = ' '.join(f'{x:.3f}' for x in row)
            rows.append(row_str)
        # Join rows with newlines
        opp = '\n'.join(rows)
        return f"DiscreteStateFilter(my_row={self.my_row},my_col={self.my_col},opp_belief=\n{opp})"
    
    def sample(self):
        """
        Returns a random sample from the belief distribution as (row, col)
        """
        flat_index = np.random.choice(np.arange(self.num_rows * self.num_cols), p=self.opp_belief.flatten())

        # Convert flat index back to 2D coordinates
        row = flat_index // self.num_cols
        col = flat_index % self.num_cols
        
        return State(self.my_row, self.my_col, row, col)

    def get_center_of_mass(self):
        """
        Returns the center of mass of the belief distribution as (row, col)

        Using this for policies only works if the distribution doesn't have multiple peaks.
        """
        # Calculate weighted sum of row and column indices
        row_indices = np.arange(self.num_rows)
        col_indices = np.arange(self.num_cols)
        
        # Get marginal distributions
        row_marginal = np.sum(self.opp_belief, axis=1)
        col_marginal = np.sum(self.opp_belief, axis=0)
        
        # Calculate center of mass
        row_com = np.sum(row_indices * row_marginal)
        col_com = np.sum(col_indices * col_marginal)
        
        return (row_com, col_com)
    
    def get_belief_vector(self):
        return self.opp_belief.flatten()
    
    def get_belief(self):
        """
        TODO: make this immutable
        """
        return self.opp_belief

    # def obs_prob(self, observation, gaze_action, row, col):
    #     """
    #     O(o | a,s’)
    #     row, col is s'
    #     """
    #     my_row, my_col, opp_row, opp_col = observation

    #     if opp_row == row and opp_col == col:
    #         # we observe the opponent at s'
    #         return 1.0

    def move(self, move_action):
        if move_action == MovementActions.N and self.my_row > 0:
            return self.my_row - 1, self.my_col
        elif move_action == MovementActions.E and self.my_col < self.num_cols - 1:
            return self.my_row, self.my_col + 1
        elif move_action == MovementActions.S and self.my_row < self.num_rows - 1:
            return self.my_row + 1, self.my_col
        elif move_action == MovementActions.W and self.my_col > 0:
            return self.my_row, self.my_col - 1
        else:
            return self.my_row, self.my_col