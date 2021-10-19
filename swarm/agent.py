#!/bin/python3


import numpy as np

from mesa import Agent

import swarm.movement as mv
import swarm.behavior as bh
from swarm.target import Targets


class SwarmAgent(Agent):
    """
    Swarm agent performing coverage and tracking.
    """
    def __init__(self, unique_id, model):
        """
        Create a new agent.

        # Parameters
        - unique_id: Unique agent identifier.
        - model: Model to build upon.
        """
        # initialize agent model
        super().__init__(unique_id, model)

        # behavior state
        self.state = "coverage"

        # minimum number of time steps before a behavior can be switched
        self.state_steps_min = 15

        # number of time steps in current behavior
        self.state_steps = 0

        # neighbors in same state
        self.neighbors = []

        # create behaviors
        self.coverage = bh.Repulsion(self, mv.RandomDirection(self))
        self.tracking = bh.Flocking(self, mv.Follow(self), "ring")
        self.charging = bh.Repulsion(self, mv.Follow(self))

        # targets known to this agent
        self.targets = Targets(self)

        # agent velocity
        self.velocity = np.zeros(2)

        # remaining time steps
        self.battery = self.model.battery

        # percent of battery to keep as reserve [0,1]
        self.battery_min = 0.1

        # battery state of charge above which agent continues to work [0,1]
        self.battery_max = 0.9

        # time steps for recharging
        self.charging_steps = 10

        # charging point range (how close to be for recharging)
        self.charging_range = self.model.vision_range

    def step(self):
        """
        Execute one step of the simulation: move the agent.
        """
        # update swarm and targets
        self.update()

        # switch behavior
        self.behavior()

        # debug output
        if self.model.debug:
            targets = ["{0} ({1})".format(t, self.targets.targets[t].state) for t in self.targets.targets]
            print("Agent {0} {1}: {2}".format(self.unique_id, self.state, targets))

        # move the agent
        self.move()

    def update(self):
        """
        Update knowledge about targets and other agents.
        """
        # update battery
        self.update_battery()

        # time steps in current behavor
        self.state_steps += 1

        # get neighbors (in same state)
        self.all_neighbors = self.model.space.get_neighbors(self.pos, self.model.interaction_range, False)
        self.neighbors = [a for a in self.all_neighbors if a.state == self.state]

        # update information of targets within field of view
        self.targets.update(self.model.target_space.get_neighbors(self.pos, self.model.vision_range, False))

        # send target update to other agents
        for a in self.all_neighbors:
            a.targets.remote_update(a, self.targets.targets)

        # forget done targets
        self.targets.prune()

        # get target to be tracked
        self.targets.determine_target()

    def behavior(self):
        """
        Decide whether to perform tracking or coverage.
        """
        # need recharging
        if self.remaining_time() <= self.battery_min * self.model.battery:
            self.state = "charging"

        # currently tracking a target
        elif self.state == "tracking":
            # switch if target is done
            if self.targets.target is None:
                self.state = "coverage"
                self.state_steps = 0

            # switch if minimum number of agents exceeded
            elif len(self.neighbors) > self.model.min_agents and self.state_steps > self.state_steps_min:
                # probability to switch to coverage, increasing with number of neighbors
                p = 0.5 * np.log(1/self.model.min_agents * len(self.neighbors)) / np.log(1 + (self.model.max_agents - self.model.min_agents) / (2 * self.model.min_agents))
                p = min(p, 1.0)

                # switch to coverage with given probability
                if np.random.random() < p:
                    self.state = "coverage"
                    self.state_steps = 0

        # currently searching for targets
        elif self.state == "coverage":
            # track targets in field of view
            if self.targets.tracking:
                self.state = "tracking"

            # selected target is done, continue coverage
            elif self.targets.done_tracking:
                pass

            # another agent is tracking
            elif self.targets.help_needed and self.state_steps > self.state_steps_min:
                # switch to tracking with given probability
                if np.random.random() < self.tracking_probability():
                    self.state = "tracking"

        # currently recharging
        elif self.state == "charging":
            # switch to coverage if fully charged
            if self.battery >= self.battery_max * self.model.battery:
                self.state = "coverage"

        # unknown state, this should not happen
        else:
            print("Agent {0}: Unknown behavior state!".format(self.unique_id))

    def move(self):
        """
        Move the agent according to the current behavior.
        """

        # perform coverage
        if self.state == "coverage":
            self.velocity += self.coverage.velocity()

        # perform tracking
        elif self.state == "tracking":
            self.velocity += self.tracking.velocity()

        # recharge
        elif self.state == "charging":
            self.velocity += self.charging.velocity()

        # unknown state, this should not happen
        else:
            print("Agent {0}: Unknown behavior state {1}!".format(self.unique_id, self.state))

        # compute new position
        new_pos = self.pos + self.velocity

        # move agent to new position
        self.model.space.move_agent(self, new_pos)

    def update_battery(self):
        """
        Update the battery state of charge.
        """
        # reset battery
        if self.model.space.get_distance(self.pos, self.model.cp) <= self.charging_range and self.battery < self.model.battery:
            self.battery += self.model.battery / self.charging_steps
            if self.battery > self.model.battery:
                self.battery = self.model.battery

        # decrease battery
        else:
            self.battery -= 1

    def remaining_time(self):
        """
        Calculate the time this agent can still work before it has to go for recharging.

        # Returns
        int: Number of time steps.
        """
        return self.battery - self.model.space.get_distance(self.pos, self.model.cp)

    def get_target(self):
        """
        Get the target that the agent wants to move to.

        # Returns
        numpy array: Two-dimensional point.
        """
        # return the target to be tracked, make sure it is up-to-date by calling targets.determine_target()
        if self.state == "tracking":
            return self.targets.target.pos

        # return the charging point
        if self.state == "charging":
            return self.model.cp

        # unknown state, this should not happen
        print("Agent {0}: Unknown behavior state!".format(self.unique_id))
        return np.zeros(2)

    def tracking_probability(self):
        """
        Calculate the probability to switch from coverage to tracking. It is applicable for targets outside of own vision range to help other agents with tracking. This probability decreases exponentially with increasing target distance and increases with decreasing battery state of charge of the tracking agents.

        # Returns
        float: The tracking probability.
        """
        # distance of the target beyond own vision
        distance = self.model.space.get_distance(self.pos, self.targets.target.pos) - self.model.vision_range
        if distance < 0:
            print("Agent {0}: Cannot help tracking, target {1} already tracked!".format(self.unique_id, self.targets.target.unique_id))
            return 1

        # remaining time the target still needs to be tracked
        time_target = self.targets.target.track - self.targets.target.tracked

        # maximum remaining time the target can be tracked by any agent that currently tracks it
        if len(self.targets.target.agents) > 0:
            time_agents = max([a.remaining_time() for a in self.targets.target.agents])
        else:
            time_agents = 0

        # percent of time the target can still be tracked
        if 0 < time_target:
            time_perc = time_agents / time_target
        else:
            time_perc = 1

        # probability to help tracking
        return np.exp(distance * 2 * np.log(0.5) / self.model.help_range * time_perc)
