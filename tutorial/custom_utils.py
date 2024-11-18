from enum import Enum

import numpy as np

#gaze is a 5x5 square with agent at a corner
GAZE_DISTANCE = 5

class MovementActions(Enum):
    DO_NOTHING = 0
    N = 1
    E = 2
    S = 3
    W = 4

"""
v0: quadrants
"""
class GazeActions(Enum):
    NE = 0
    SE = 1
    SW = 2
    NW = 3

# class GazeActions(Enum):
#     N = 0
#     NE = 1
#     E = 2
#     SE = 3
#     S = 4
#     SW = 5
#     W = 6
#     NW = 7

class Agent:
    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        self.gaze = None

    def __str__(self):
        return f"{self.name} at ({self.x}, {self.y})"
    
    def get_gaze_mask(self, rows, cols):
        """
        Returns a binary numpy array with 1 if cell is in agent's gaze, 0 otherwise.
        Array dimension is (num_y_cells, num_x_cells). <- important!
        """
        gaze_mask = np.zeros((rows, cols))
        
        if self.gaze == GazeActions.SE:
            top_left = (self.y, self.x)
        elif self.gaze == GazeActions.SW:
            top_left = (self.y, self.x - (GAZE_DISTANCE - 1))
        elif self.gaze == GazeActions.NW:
            top_left = (self.y - (GAZE_DISTANCE - 1), self.x - (GAZE_DISTANCE - 1))
        elif self.gaze == GazeActions.NE:
            top_left = (self.y - (GAZE_DISTANCE - 1), self.x)
        else:
            raise ValueError(f"Unknown gaze direction: {self.gaze}")
        
        for row in range(GAZE_DISTANCE):
            cur_row = top_left[0] + row
            if cur_row < 0 or cur_row >= rows:
                continue
            for col in range(GAZE_DISTANCE):
                cur_col = top_left[1] + col
                if cur_col < 0 or cur_col >= cols:
                    continue
                gaze_mask[cur_row, cur_col] = 1
        return gaze_mask

def index_to_action(index):
    """
    0 (<MovementActions.DO_NOTHING: 0>, <GazeActions.N: 0>)
    1 (<MovementActions.DO_NOTHING: 0>, <GazeActions.NE: 1>)
    2 (<MovementActions.DO_NOTHING: 0>, <GazeActions.E: 2>)
    3 (<MovementActions.DO_NOTHING: 0>, <GazeActions.SE: 3>)
    4 (<MovementActions.DO_NOTHING: 0>, <GazeActions.S: 4>)
    5 (<MovementActions.DO_NOTHING: 0>, <GazeActions.SW: 5>)
    6 (<MovementActions.DO_NOTHING: 0>, <GazeActions.W: 6>)
    7 (<MovementActions.DO_NOTHING: 0>, <GazeActions.NW: 7>)
    8 (<MovementActions.N: 1>, <GazeActions.N: 0>)
    etc.
    """
    return MovementActions(index // len(GazeActions)), GazeActions(index % len(GazeActions))
