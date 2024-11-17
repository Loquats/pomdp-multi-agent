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

class MoveDir(Enum):
    N = 0
    E = 1
    S = 2
    W = 3

class GazeDir(Enum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7

class Agent:
    def __init__(self, name):
        self.name = name
        self.x = None
        self.y = None
        self.gaze = None

    def __str__(self):
        return f"{self.name} at ({self.x}, {self.y})"

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
        self.timestep = None
        self.max_timesteps = 100

        self.you = Agent("you")
        self.opp = Agent("opp")
        self.agent_names = [self.you.name, self.opp.name]
        self.agents = [self.you, self.opp]

        self.num_x_cells = 20
        self.num_y_cells = 10
        cell_px = 30
        self.grid = Grid(self.num_x_cells, self.num_y_cells, cell_px, cell_px, title="custom game", margin=1)


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

        # self.opp.x = self.num_x_cells - 1
        # self.opp.y = self.num_y_cells - 1

        self.opp.x = 3
        self.opp.y = 5

        observations = self.get_full_observations()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def get_full_observations(self):
        observation = (self.you.x, self.you.y, self.opp.x, self.opp.y)
        observations = {
            "you": {"observation": observation, "action_mask": self.get_action_mask(self.you)},
            "opp": {"observation": observation, "action_mask": self.get_action_mask(self.opp)},
        }
        return observations

    def move(self, agent, direction: MoveDir):
        if direction == MoveDir.N and agent.y > 0:
            agent.y -= 1
        elif direction == MoveDir.E and agent.x < self.num_x_cells - 1:
            agent.x += 1
        elif direction == MoveDir.S and agent.y < self.num_y_cells - 1:
            agent.y += 1
        elif direction == MoveDir.W and agent.x > 0:
            agent.x -= 1

    def get_action_mask(self, agent):
        action_mask = np.ones(4, dtype=np.int8)
        if agent.x == 0:
            action_mask[3] = 0  # Block W movement
        elif agent.x == self.num_x_cells - 1:
            action_mask[1] = 0  # Block E movement
        if agent.y == 0:
            action_mask[0] = 0  # Block N movement
        elif agent.y == self.num_y_cells - 1:
            action_mask[2] = 0  # Block S movement
        return action_mask

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
        self.move(self.you, actions[self.you.name])
        self.move(self.opp, actions[self.opp.name])

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
        self.grid[self.you.x, self.you.y] = "Y"
        self.grid[self.opp.x, self.opp.y] = "O"
        self.render()

        return observations, rewards, terminations, truncations, infos


    def render(self):
        if "text" in self.render_mode:
            grid = np.zeros((self.num_x_cells, self.num_y_cells))
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
        return Discrete(4)