from enum import Enum
import functools
from dataclasses import dataclass

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

class GazeActions(Enum):
    NE = 0
    SE = 1
    SW = 2
    NW = 3

@dataclass(frozen=True)
class State:
    my_row: int
    my_col: int
    opp_row: int
    opp_col: int

@dataclass(frozen=True)
class Action:
    move: MovementActions
    gaze: GazeActions

@dataclass(frozen=True)
class Observation:
    my_row: int
    my_col: int
    opp_row: int
    opp_col: int

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
    """
    Intentionally return out-of-bounds values (eg. negative values), which get_gaze_bounds relies on
    """
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
    
def get_gaze_bounds(gaze, row, col, num_rows, num_cols):
    """
    Returns the bounds of the gaze area as a tuple of 4 integers:
    (min_row, min_col, max_row, max_col)
    The max_row and max_col are inclusive.
    """
    # print("input to get_gaze_bounds", gaze, row, col, num_rows, num_cols)
    top_left_row, top_left_col = get_top_left(gaze, row, col)
    
    min_row = max(top_left_row, 0)
    min_col = max(top_left_col, 0)
    max_row = min(top_left_row + GAZE_DISTANCE, num_rows-1)
    max_col = min(top_left_col + GAZE_DISTANCE, num_cols-1)
    return (min_row, min_col, max_row, max_col)

@functools.lru_cache(maxsize=128)
def get_gaze_mask(row, col, gaze, rows, cols, pos_value=1, neg_value=0):
    """
    Returns a binary numpy array with 1 if cell is in agent's gaze, 0 otherwise.
    Array dimension is (num_y_cells, num_x_cells). <- important!
    """
    gaze_mask = np.full((rows, cols), neg_value)
    
    top_left = get_top_left(gaze, row, col)
    
    for row in range(GAZE_DISTANCE+1):
        cur_row = top_left[0] + row
        if cur_row < 0 or cur_row >= rows:
            continue
        for col in range(GAZE_DISTANCE+1):
            cur_col = top_left[1] + col
            if cur_col < 0 or cur_col >= cols:
                continue
            gaze_mask[cur_row, cur_col] = pos_value
    return gaze_mask
    
def is_visible(gaze_mask, row, col):
    """
    deprecated because inefficient
    TODO: replace this with in_gaze_box, which does the same thing!
    """
    return gaze_mask[row, col] == 1

def in_gaze_box(my_row, my_col, my_gaze_action, target_row, target_col, num_rows, num_cols):    
    min_row, min_col, max_row, max_col = get_gaze_bounds(my_gaze_action, my_row, my_col, num_rows, num_cols)
    is_in_gaze_box = min_row <= target_row <= max_row and min_col <= target_col <= max_col
    # print("gaze bounds", min_row, min_col, max_row, max_col, is_in_gaze_box)
    return is_in_gaze_box

def index_to_action(index):
    """
    0 (<MovementActions.DO_NOTHING: 0>, <GazeActions.NE: 0>)
    1 (<MovementActions.DO_NOTHING: 0>, <GazeActions.SE: 1>)
    2 (<MovementActions.DO_NOTHING: 0>, <GazeActions.SW: 2>)
    3 (<MovementActions.DO_NOTHING: 0>, <GazeActions.NW: 3>)
    4 (<MovementActions.N: 1>, <GazeActions.N: 0>)
    etc.
    """
    move = MovementActions(index // len(GazeActions))
    gaze = GazeActions(index % len(GazeActions))
    return Action(move, gaze)

def action_to_index(action):
    return action.move.value * len(GazeActions) + action.gaze.value
