"""
my actual environment...
"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, max_cycles=25, continuous_actions=False, render_mode=None):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "andy_env"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_landmarks=1):
        world = World()
        world.dt = 1
        # add agents
        world.agents = [Agent() for _ in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size=0.2
            # boundary???
        return world

    def reset_world(self, world, np_random):
        """
        np_random is https://numpy.org/doc/stable/reference/random/generator.html
        """
        # random properties for agents
        # for i, agent in enumerate(world.agents):
        #     agent.color = np.array([0.25, 0.25, 0.25])

        world.agents[0].color = np.array([0.1, 0.9, 0.1]) # green
        world.agents[1].color = np.array([0.9, 0.1, 0.1]) # red

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75]) # grey?

        # set random initial states
        for agent in world.agents:
            # TODO: discretize
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = np_random.integers(0, 10, world.dim_p).astype(np.float64)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            print(f"{agent} initial pos: {agent.state.p_pos}")

        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_pos = np_random.integers(0, 10, world.dim_p).astype(np.float64)
            landmark.state.p_vel = np.zeros(world.dim_p)
            print(f"{landmark} initial pos: {landmark.state.p_pos}")

    def reward(self, agent, world):
        """
        TODO: gaze reward
        """
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        print(agent, agent.state.p_pos)
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            print(f"landmark pos: {entity.state.p_pos}")
        return np.concatenate([agent.state.p_vel] + entity_pos)
