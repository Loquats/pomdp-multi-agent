from enum import Enum
import functools

import numpy as np

# GAZE_DISTANCE = 4 means gaze is a 5x5 square with agent at a corner
GAZE_DISTANCE = 4
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
    
def get_top_left(gaze, row, col):
    if gaze == GazeActions.SE:
        return (row, col)
    elif gaze == GazeActions.SW:
        return (row, col - GAZE_DISTANCE)
    elif gaze == GazeActions.NW:
        return (row - GAZE_DISTANCE, col - GAZE_DISTANCE)
    elif gaze == GazeActions.NE:
        return (row - GAZE_DISTANCE, col)
    else:
        raise ValueError(f"Unknown gaze direction: {gaze}")
    
def get_gaze_bounds(gaze, row, col):
    """
    Returns the bounds of the gaze area as a tuple of 4 integers:
    (min_row, min_col, max_row, max_col)
    The max_row and max_col are inclusive.
    """
    min_row, min_col = get_top_left(gaze, row, col)
    return (min_row, min_col, min_row + GAZE_DISTANCE, min_col + GAZE_DISTANCE)

@functools.lru_cache(maxsize=128)
def get_gaze_mask(row, col, gaze, rows, cols):
    """
    Returns a binary numpy array with 1 if cell is in agent's gaze, 0 otherwise.
    Array dimension is (num_y_cells, num_x_cells). <- important!
    """
    gaze_mask = np.zeros((rows, cols))
    
    top_left = get_top_left(gaze, row, col)
    
    for row in range(GAZE_DISTANCE+1):
        cur_row = top_left[0] + row
        if cur_row < 0 or cur_row >= rows:
            continue
        for col in range(GAZE_DISTANCE+1):
            cur_col = top_left[1] + col
            if cur_col < 0 or cur_col >= cols:
                continue
            gaze_mask[cur_row, cur_col] = 1
    return gaze_mask
    
def is_visible(gaze_mask, row, col):
    # deprecated because inefficient
    return gaze_mask[row, col] == 1

def in_gaze_box(my_row, my_col, my_gaze_action, target_row, target_col, num_rows, num_cols):    
    min_row, min_col, max_row, max_col = get_gaze_bounds(my_gaze_action, my_row, my_col)
    return min_row <= target_row <= max_row and min_col <= target_col <= max_col

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
