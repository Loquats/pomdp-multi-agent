import numpy as np
from custom_utils import in_gaze_box

class DiscreteStateFilter:
    def __init__(self, num_rows, num_cols):
        # initialize uniform probability
        self.belief = np.ones((num_rows, num_cols)) / (num_rows * num_cols)
        self.num_rows, self.num_cols = num_rows, num_cols

    def is_valid(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols

    def update(self, observation, action):
        """
        Update the belief based on the observation.
        observation: (my_row, my_col, opponent_row, opponent_col)
            (opponent_row, opponent_col) is (-1, -1) if not observed
        action: 
        """
        my_row, my_col, opp_row, opp_col = observation
        _my_move_action, my_gaze_action = action # we don't need the move action
        
        if opp_row == -1 and opp_col == -1:
            print("CAN'T SEE OPPONENT")
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
            num_rows, num_cols = self.belief.shape
            for row in range(num_rows):
                for col in range(num_cols):
                    # get valid recipients, which are not in gaze box
                    valid_recipients = []
                    for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                        target_row = row + drow
                        target_col = col + dcol

                        if not self.is_valid(target_row, target_col):
                            continue
                        elif in_gaze_box(my_row, my_col, my_gaze_action, target_row, target_col, num_rows, num_cols):
                            continue
                        else:
                            valid_recipients.append((target_row, target_col))
                        
                    if len(valid_recipients) == 0:
                        # poof, the mass disappears
                        # it'll get fixed during normalization
                        continue
                    # else, diffuse the mass
                    parcel = self.belief[row, col] / len(valid_recipients)
                    for recipient_row, recipient_col in valid_recipients:
                        new_belief[recipient_row, recipient_col] += parcel
            
            # Normalize to probability distribution. Is this really necessary?
            self.belief = new_belief / np.sum(new_belief)
        else:
            print("SEE OPPONENT")
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

    # def obs_prob(self, observation, gaze_action, row, col):
    #     """
    #     O(o | a,s’)
    #     row, col is s'
    #     """
    #     my_row, my_col, opp_row, opp_col = observation

    #     if opp_row == row and opp_col == col:
    #         # we observe the opponent at s'
    #         return 1.0

