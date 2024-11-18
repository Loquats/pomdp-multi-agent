from enum import Enum
import functools

import numpy as np

#gaze is a 5x5 square with agent at a corner
GAZE_DISTANCE = 5
NUM_SEEING_STEPS_TO_WIN = 3
WIN_REWARD = 100
SEE_REWARD = 1
DEFAULT_REWARD = 0

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
        self.row = None
        self.col = None
        self.gaze = None
        self.num_steps_seeing = 0

    def __str__(self):
        return f"{self.name} at ({self.row}, {self.col})"
        
@functools.lru_cache(maxsize=128)
def get_gaze_mask(row, col, gaze, rows, cols):
    """
    Returns a binary numpy array with 1 if cell is in agent's gaze, 0 otherwise.
    Array dimension is (num_y_cells, num_x_cells). <- important!
    """
    gaze_mask = np.zeros((rows, cols))
    
    if gaze == GazeActions.SE:
        top_left = (row, col)
    elif gaze == GazeActions.SW:
        top_left = (row, col - (GAZE_DISTANCE - 1))
    elif gaze == GazeActions.NW:
        top_left = (row - (GAZE_DISTANCE - 1), col - (GAZE_DISTANCE - 1))
    elif gaze == GazeActions.NE:
        top_left = (row - (GAZE_DISTANCE - 1), col)
    else:
        raise ValueError(f"Unknown gaze direction: {gaze}")
    
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
    
def is_visible(gaze_mask, row, col):
    return gaze_mask[row, col] == 1

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
