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

def make_env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    # internal_render_mode = render_mode if render_mode != "ansi" else "human"
    # env = raw_env(render_mode=internal_render_mode)

    env = CustomActionMaskedEnvironment(render_mode=render_mode)

    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env) # only works for AEC

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)

    return env

class CustomActionMaskedEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
        "render_fps": 10,
        "render_modes": ["human", "text", "human_text", "none"],
    }

    def __init__(self, render_mode="none"):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.render_mode = render_mode
        print(f"init env with render_mode: {self.render_mode}")
        self.timestep = None
        self.max_timesteps = 100

        self.you = Agent("you")
        self.opp = Agent("opp")
        self.agent_names = [self.you.name, self.opp.name]
        self.agents = [self.you, self.opp]

        self.num_cols = 20
        self.num_rows = 10
        cell_px = 30
        self.grid = Grid(self.num_cols, self.num_rows, cell_px, cell_px, title="custom game", margin=1)


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

        self.you.x = 0
        self.you.y = 0
        self.you.gaze = GazeActions.SE

        self.opp.x = self.num_cols - 1
        self.opp.y = self.num_rows - 1
        self.opp.gaze = GazeActions.NW

        observations = self.get_full_observations()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def get_full_observations(self):
        """
        TODO: should we observe the gaze?
        TODO: should we have action masks?
        """
        observation = (self.you.x, self.you.y, self.opp.x, self.opp.y)
        # observations = {
        #     "you": {"observation": observation, "action_mask": self.get_action_mask(self.you)},
        #     "opp": {"observation": observation, "action_mask": self.get_action_mask(self.opp)},
        # }
        observations = {
            "you": {"observation": observation},
            "opp": {"observation": observation},
        }
        return observations

    def move(self, agent, direction: int):
        direction = MovementActions(direction)
        if direction == MovementActions.N and agent.y > 0:
            agent.y -= 1
        elif direction == MovementActions.E and agent.x < self.num_cols - 1:
            agent.x += 1
        elif direction == MovementActions.S and agent.y < self.num_rows - 1:
            agent.y += 1
        elif direction == MovementActions.W and agent.x > 0:
            agent.x -= 1

    def gaze(self, agent, direction: int):
        direction = GazeActions(direction)
        agent.gaze = direction


    # def get_action_mask(self, agent):
    #     action_mask = np.ones(4, dtype=np.int8)
    #     if agent.x == 0:
    #         action_mask[3] = 0  # Block W movement
    #     elif agent.x == self.num_x_cells - 1:
    #         action_mask[1] = 0  # Block E movement
    #     if agent.y == 0:
    #         action_mask[0] = 0  # Block N movement
    #     elif agent.y == self.num_y_cells - 1:
    #         action_mask[2] = 0  # Block S movement
    #     return action_mask

    def get_reward(self, agent):
        pass

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - x and y coordinates
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

        # Check termination conditions
        terminations = {a.name: False for a in self.agents}
        rewards = {a.name: 0 for a in self.agents}
        # if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
        #     rewards = {"prisoner": -1, "guard": 1}
        #     terminations = {a.name: True for a in self.agents}
        #     self.agents = []

        # elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
        #     rewards = {"prisoner": 1, "guard": -1}
        #     terminations = {a: True for a in self.agents}
        #     self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a.name: False for a in self.agents}
        if self.timestep > self.max_timesteps:
            rewards = {a.name: 0 for a in self.agents}
            truncations = {a.name: True for a in self.agents}
            self.agents = []
        self.timestep += 1

        observations = self.get_full_observations()

        # Get dummy infos (not used in this example)
        infos = {a.name: {} for a in self.agents}

        self.grid.clear_all()
        self.draw_gazes()
        self.grid[self.you.x, self.you.y] = "Y"
        self.grid[self.opp.x, self.opp.y] = "A"

        self.render()

        return observations, rewards, terminations, truncations, infos

    def draw_gazes(self):
        you_gaze_mask = self.you.get_gaze_mask(self.num_rows, self.num_cols)
        opp_gaze_mask = self.opp.get_gaze_mask(self.num_rows, self.num_cols)

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
            grid = np.zeros((self.num_cols, self.num_rows))
            grid[self.you.y, self.you.x] = "Y"
            grid[self.opp.y, self.opp.x] = "O"
            print(f"{grid} \n")

        if "human" in self.render_mode:
            self.grid.redraw()
            self.grid.clock.tick(1)

    def close(self):
        self.grid.done()

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7 - 1] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(MovementActions) * len(GazeActions))
