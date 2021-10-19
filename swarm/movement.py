#!/bin/python3


import numpy as np


class Movement:
    """
    Generic movement class.
    """
    def __init__(self, agent):
        """
        Create a new movement.

        # Parameters
        - agent: Agent performing the movement.
        """
        self.agent = agent

        # distance to the current waypoint below which the next waypoint of the path is selected.
        self.wp_tolerance = 10

    def get_waypoint(self):
        """
        Get the next waypoint for the agent to move to.
        """
        pass


class Square(Movement):
    """
    Move on a square path.
    """
    def __init__(self, agent):
        """
        Create a new square path.

        # Parameters
        - agent: Agent performing the movement.
        """
        # initialize movement
        super().__init__(agent)

        # generate square path
        x1 = self.agent.model.space.center[0] / 2.0
        x2 = self.agent.model.space.center[0] * 3.0 / 2.0
        y1 = self.agent.model.space.center[1] / 2.0
        y2 = self.agent.model.space.center[1] * 3.0 / 2.0
        self.path = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

        # initial waypoint
        self.wp = 0

        # wait a few iterations before selecting next waypoint to be synchronized with all agents
        self.wp_delay = 2

    def get_waypoint(self):
        """
        Get the next waypoint of the path if the swarm is close or past the current one.

        # Returns
        numpy array: Two-dimensional point.
        """
        # compute flock centroid
        self.neighbors = self.agent.model.space.get_neighbors(self.agent.pos, self.agent.model.interaction_range, False)
        flock = np.mean(np.array([n.pos for n in self.neighbors+[self.agent]]), axis=0)

        # select next waypoint if flock is close enough to or past current waypoint
        if self.agent.model.space.get_distance(flock, self.path[self.wp]) < self.wp_tolerance:
            if self.wp_delay > 0:
                self.wp_delay -= 1
            else:
                self.wp_delay = 2
                self.wp += 1

        # repeat path
        if self.wp >= len(self.path):
            self.wp = self.wp % len(self.path)

        # return coordinates of waypoint
        return self.path[self.wp]


class Follow(Movement):
    """
    Follow the target being tracked.
    """
    def __init__(self, agent):
        """
        Create a new following movement.

        # Parameters
        - agent: Agent performing the movement.
        """
        # initialize movement
        super().__init__(agent)

    def get_waypoint(self):
        """
        Get the next waypoint for the agent to move to.

        # Returns
        numpy array: Two-dimensional point.
        """
        return self.agent.get_target()


class RandomDirection(Movement):
    """
    Move on a straight line until reaching the environment boundary. Then change direction randomly.
    """
    def __init__(self, agent):
        """
        Create a new random direction movement.

        # Parameters
        - agent: Agent performing the movement.
        """
        # initialize movement
        super().__init__(agent)

        # step size
        self.step = 10

        # initial direction
        self.dir = np.random.random() * 2 * np.pi

    def get_waypoint(self):
        """
        Get the next waypoint for the agent to move to.
        """
        # change direction at boundary
        while self.agent.model.space.out_of_bounds(self.agent.pos + [np.cos(self.dir) * self.step, np.sin(self.dir) * self.step]):
            self.dir = np.random.random() * 2 * np.pi

        return self.agent.pos + [np.cos(self.dir) * self.step, np.sin(self.dir) * self.step]


class RandomWalk(Movement):
    """
    Move with steps at fixed size in random direction.
    """
    def __init__(self, agent):
        """
        Create a new random walk movement.

        # Parameters
        - agent: Agent performing the movement.
        """
        # initialize movement
        super().__init__(agent)

        # step size
        self.step = 10

    def get_waypoint(self):
        """
        Get the next waypoint for the agent to move to.
        """
        dir = np.random.random() * 2 * np.pi
        return self.agent.pos + [np.cos(dir) * self.step, np.sin(dir) * self.step]
