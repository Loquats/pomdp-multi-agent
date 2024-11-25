import functools
import random
from copy import copy
import os
from enum import Enum

import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.utils import seeding
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector


from pettingzoo import ParallelEnv
from gridworld.grid import Grid

from custom_utils import *
from text_grid import TextGrid

RENDER_FPS = 1

# def make_env(render_mode=None):
#     """
#     The env function often wraps the environment in wrappers by default.
#     You can find full documentation for these methods
#     elsewhere in the developer documentation.
#     """
#     # internal_render_mode = render_mode if render_mode != "ansi" else "human"
#     # env = raw_env(render_mode=internal_render_mode)

#     env = CustomActionMaskedEnvironment(render_mode=render_mode)

#     # This wrapper is only for environments which print results to the terminal
#     if render_mode == "ansi":
#         env = wrappers.CaptureStdoutWrapper(env)
#     # this wrapper helps error handling for discrete action spaces
#     # env = wrappers.AssertOutOfBoundsWrapper(env) # only works for AEC

#     # Provides a wide vareity of helpful user errors
#     # Strongly recommended
#     # env = wrappers.OrderEnforcingWrapper(env)

#     return env

class MarkovGameEnvironment(ParallelEnv):

    def __init__(self, fully_observable: bool, render_mode="none", max_timesteps=1000):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.fully_observable = fully_observable
        assert render_mode in ["pygame", "text", "none"]
        self.render_mode = render_mode
        # print(f"init env with render_mode: {self.render_mode}")
        self.timestep = None
        self.max_timesteps = max_timesteps

        self.you = Agent("you")
        self.opp = Agent("opp")
        self.agent_names = [self.you.name, self.opp.name]
        self.agents = [self.you, self.opp]

        self.num_cols = 20
        self.num_rows = 10

        if "pygame" in self.render_mode:
            cell_px = 30
            self.grid = Grid(self.num_cols, self.num_rows, cell_px, cell_px, title="custom game", margin=1)
        else:
            self.grid = TextGrid(self.num_cols, self.num_rows)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        if seed is not None:
            self._seed(seed=seed)

        self.timestep = 0

        self.you.row = 0
        self.you.col = 0
        self.you.gaze = GazeActions.SE

        self.opp.row = self.num_rows - 1
        self.opp.col = self.num_cols - 1
        self.opp.gaze = GazeActions.NW

        if self.fully_observable:
            observations = self.get_full_observations()
        else:
            observations = self.get_partial_observations()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def get_full_observations(self):
        """
        !!! the order is always (this_agent_observations..., other_agent_observations...)
        """
        observations = {
            "you": (self.you.row, self.you.col, self.opp.row, self.opp.col),
            "opp": (self.opp.row, self.opp.col, self.you.row, self.you.col),
        }
        return observations
    
    def get_partial_observations(self):
        """
        !!! the order is always (this_agent_observations..., other_agent_observations...)
        """
        if in_gaze_box(self.you.row, self.you.col, self.you.gaze, self.opp.row, self.opp.col, self.num_rows, self.num_cols):
            you_observation = (self.you.row, self.you.col, self.opp.row, self.opp.col)
        else:
            you_observation = (self.you.row, self.you.col, -1, -1)

        if in_gaze_box(self.opp.row, self.opp.col, self.opp.gaze, self.you.row, self.you.col, self.num_rows, self.num_cols):
            opp_observation = (self.opp.row, self.opp.col, self.you.row, self.you.col)
        else:
            opp_observation = (self.opp.row, self.opp.col, -1, -1)

        observations = {
            "you": you_observation,
            "opp": opp_observation,
        }
        return observations

    def move(self, agent, direction: int):
        direction = MovementActions(direction)
        if direction == MovementActions.N and agent.row > 0:
            agent.row -= 1
        elif direction == MovementActions.E and agent.col < self.num_cols - 1:
            agent.col += 1
        elif direction == MovementActions.S and agent.row < self.num_rows - 1:
            agent.row += 1
        elif direction == MovementActions.W and agent.col > 0:
            agent.col -= 1

    def gaze(self, agent, direction: int):
        direction = GazeActions(direction)
        agent.gaze = direction

    def get_rewards(self, agent, other_agent):
        """
        1 if you can see the opponent
        you do not know if the opponent can see you
        100 if you have seen the opponent for 3 consecutive steps
        """
        gaze_mask = get_gaze_mask(agent.row, agent.col, agent.gaze, self.num_rows, self.num_cols)

        if is_visible(gaze_mask, other_agent.row, other_agent.col):
            agent.num_steps_seeing += 1
            if agent.num_steps_seeing == NUM_SEEING_STEPS_TO_WIN:
                return WIN_REWARD, True
            else:
                return SEE_REWARD, False
        else:
            agent.num_steps_seeing = 0
            return DEFAULT_REWARD, False

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        for agent in self.agents:
            movement_action, gaze_action = index_to_action(actions[agent.name])
            self.move(agent, movement_action)
            self.gaze(agent, gaze_action)

        you_reward, you_win = self.get_rewards(self.you, self.opp)
        opp_reward, opp_win = self.get_rewards(self.opp, self.you)

        rewards = {self.you.name: you_reward, self.opp.name: opp_reward}
        terminations = {a.name: you_win or opp_win for a in self.agents}
        if you_win or opp_win:
            self.agents = []
            self.agent_names = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a.name: False for a in self.agents}
        if self.timestep > self.max_timesteps:
            rewards = {a.name: DEFAULT_REWARD for a in self.agents}
            truncations = {a.name: True for a in self.agents}
            self.agents = []
            self.agent_names = []

        if self.fully_observable:
            observations = self.get_full_observations()
        else:
            observations = self.get_partial_observations()
        
        infos = {
            self.you.name: {"win": you_win},
            self.opp.name: {"win": opp_win},
        }

        self.update_grid()
        self.render()

        self.timestep += 1 # always do this last
        return observations, rewards, terminations, truncations, infos

    def update_grid(self):
        self.grid.clear_all()
        self.draw_gazes()
        self.grid[self.you.col, self.you.row] = "Y"
        self.grid[self.opp.col, self.opp.row] = "A"

    def draw_gazes(self):
        you_gaze_mask = get_gaze_mask(self.you.row, self.you.col, self.you.gaze, self.num_rows, self.num_cols)
        opp_gaze_mask = get_gaze_mask(self.opp.row, self.opp.col, self.opp.gaze, self.num_rows, self.num_cols)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if you_gaze_mask[row, col] == 1 and opp_gaze_mask[row, col] == 1:
                    self.grid[col, row] = "x"
                elif you_gaze_mask[row, col] == 1:
                    self.grid[col, row] = "y"
                elif opp_gaze_mask[row, col] == 1:
                    self.grid[col, row] = "a"

    def render(self):
        if "text" in self.render_mode:
            self.grid.print()

        if "pygame" in self.render_mode:
            self.grid.redraw()
            self.grid.clock.tick(RENDER_FPS)

    def close(self):
        self.grid.done()

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=128)
    def observation_space(self, agent):
        """
        TODO: wtf is this?
        """
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7 - 1] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=128)
    def action_space(self, agent):
        return Discrete(len(MovementActions) * len(GazeActions))
