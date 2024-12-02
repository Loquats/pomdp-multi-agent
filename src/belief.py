import numpy as np
from src.env_utils import get_gaze_bounds

class DiscreteStateFilter:
    def __init__(self, num_rows, num_cols):
        # initialize uniform probability
        self.belief = np.ones((num_rows, num_cols)) / (num_rows * num_cols)
        self.num_rows, self.num_cols = num_rows, num_cols

    def is_valid_cell(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols

    def update(self, observation, action):
        """
        Update the belief based on the observation, and the action resulting in that observation.
        observation: (my_row, my_col, opponent_row, opponent_col)
            (opponent_row, opponent_col) is (-1, -1) if not observed
        action: 
        """
        my_row, my_col, opp_row, opp_col = observation
        _my_move_action, my_gaze_action = action # we don't need the move action
        
        if opp_row == -1 and opp_col == -1:
            # print("CAN'T SEE OPPONENT")
            """
            complicated diffusion of probability

            for each cell in the belief grid, the probability is
            diffused to up to 4 neighbors plus itself
            However, the cell must not donate to cells in the gaze box
            First count how many neighbors are NOT in the gaze box: num_valid
            These are valid recipients of the probability mass
            Divide the current cell's probability mass by num_valid
            Add each mass/num_valid to the recipient cells

            at the very end, normalize the belief
            """
            new_belief = np.zeros_like(self.belief)
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    # get valid recipients
                    valid_recipients = []
                    for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                        recipient_row = row + drow
                        recipient_col = col + dcol

                        if self.is_valid_cell(recipient_row, recipient_col):
                            valid_recipients.append((recipient_row, recipient_col))

                    # diffuse the mass
                    parcel = self.belief[row, col] / len(valid_recipients)
                    for recipient_row, recipient_col in valid_recipients:
                        new_belief[recipient_row, recipient_col] += parcel

            # zero out the gaze box
            min_row, min_col, max_row, max_col = get_gaze_bounds(my_gaze_action, my_row, my_col, self.num_rows, self.num_cols)
            for row in range(min_row, max_row+1):
                for col in range(min_col, max_col+1):
                    new_belief[row, col] = 0
            
            # Normalize to probability distribution. Is this really necessary? Probably yes, to avoid precision issues.
            total = np.sum(new_belief)
            if total == 0:
                print("WTF")
                print(observation)
                print(action)
                print("DSF gaze bounds", min_row, min_col, max_row, max_col)
                print(self)
                raise Exception("belief sum is 0")
            
            self.belief = new_belief / total
        else:
            # print("CAN SEE OPPONENT")
            # we know exactly where the opponent is
            self.belief.fill(0.0)
            self.belief[opp_row, opp_col] = 1.0

    def __str__(self):
        # Convert each row to space-separated strings with rounded numbers
        rows = []
        for row in self.belief:
            # Round each number to 3 decimal places and join with spaces
            row_str = ' '.join(f'{x:.3f}' for x in row)
            rows.append(row_str)
        # Join rows with newlines
        return '\n'.join(rows)
    
    def sample(self):
        """
        Returns a random sample from the belief distribution as (row, col)
        """
        flat_index = np.random.choice(np.arange(self.num_rows * self.num_cols), p=self.belief.flatten())

        # Convert flat index back to 2D coordinates
        row = flat_index // self.num_cols
        col = flat_index % self.num_cols
        
        return row, col

    def get_center_of_mass(self):
        """
        Returns the center of mass of the belief distribution as (row, col)

        Using this for policies only works if the distribution doesn't have multiple peaks.
        """
        # Calculate weighted sum of row and column indices
        row_indices = np.arange(self.num_rows)
        col_indices = np.arange(self.num_cols)
        
        # Get marginal distributions
        row_marginal = np.sum(self.belief, axis=1)
        col_marginal = np.sum(self.belief, axis=0)
        
        # Calculate center of mass
        row_com = np.sum(row_indices * row_marginal)
        col_com = np.sum(col_indices * col_marginal)
        
        return (row_com, col_com)
    
    def get_belief_vector(self):
        return self.belief.flatten()

    # def obs_prob(self, observation, gaze_action, row, col):
    #     """
    #     O(o | a,s’)
    #     row, col is s'
    #     """
    #     my_row, my_col, opp_row, opp_col = observation

    #     if opp_row == row and opp_col == col:
    #         # we observe the opponent at s'
    #         return 1.0

