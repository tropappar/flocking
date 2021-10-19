#!/bin/python3


import numpy as np
import itertools as it

import mesa.time as t
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

from swarm.agent import SwarmAgent
from swarm.target import Target


def minimum_distance(model):
    """
    Compute the minimum distance between any agent pair.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    float: Minimum distance between any agent pair.
    """
    if len(model.density) < 1:
        return 0
    return min(model.density)

def maximum_distance(model):
    """
    Compute the maximum distance between any agent pair.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    float: Maximum distance between any agent pair.
    """
    if len(model.density) < 1:
        return 0
    return max(model.density)

def average_distance(model):
    """
    Compute the minimum distance between any agent pair.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    float: Average distance between agents.
    """
    if len(model.density) < 1:
        return 0
    return np.mean(model.density)

def collisions(model):
    """
    Compute the number of agents being too close.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Number of agents below minimum allowed distance.
    """
    return len([d for d in model.density if d < model.min_dist])

def messages_distributed(model):
    """
    Compute the number of messages exchanged in the network with distributed communication.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Number of agents.
    """
    return model.schedule.get_agent_count()

def messages_centralized(model):
    """
    Compute the number of messages exchanged in the network with centralized communication.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Number of hypothetical messages exchanged with centralized communication.
    """
    return model.schedule.get_agent_count() + sum([2 for d in model.density if d <= model.interaction_range])

def minimum_battery(model):
    """
    Compute minimum battery level in swarm.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Battery level in steps.
    """
    return min(model.batteries)

def maximum_battery(model):
    """
    Compute maximum battery level in swarm.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Battery level in steps.
    """
    return max(model.batteries)

def average_battery(model):
    """
    Compute average battery level in swarm.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Battery level in steps.
    """
    return np.mean(model.batteries)

def minimum_targets(model):
    """
    Compute minimum number of targets known to any agent.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Number of targets.
    """
    return min(model.targets)

def maximum_targets(model):
    """
    Compute maximum number of targets known to any agent.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Number of targets.
    """
    return max(model.targets)

def average_targets(model):
    """
    Compute average number of targets known to any agent.

    # Parameters
    - model: Model to access the agent data.

    # Returns
    int: Number of targets.
    """
    return np.mean(model.targets)


class Swarm(Model):
    """
    Swarm of agents performing a search mission.
    """
    def __init__(self, debug, population, targets, interaction_range, show_interaction_range, vision_range, show_vision_range, help_range, min_agents, max_agents, tracking_steps, battery):
        """
        Create a new swarm model.

        # Parameters
        - debug: Print debug output to the console.
        - population: Number of agents in the swarm.
        - targets: Number of targets.
        - interaction_range: Radius in which agents interact.
        - show_interaction_range: Visualize interaction range.
        - vision_range: Radius in which targets are detected.
        - show_vision_range: Visualize vision range.
        - help_range: Radius in which agents join others for tracking.
        - min_agents: Minimum number of agents that should track a target.
        - max_agents: Maximum number of agents that should track a target.
        - tracking_steps: Time steps which a target has to be tracked until it disappears.
        - battery: Time steps that an agent can work until it has to recharge.
        """
        # minimum allowed distance between agents before a collision occurs (for statistics)
        self.min_dist = 1

        # set parameters
        self.debug = debug
        self.interaction_range = interaction_range
        self.show_interaction_range = show_interaction_range
        self.vision_range = vision_range
        self.show_vision_range = show_vision_range
        self.help_range = help_range
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.tracking_steps = tracking_steps
        self.battery = battery

        # environment of agents and targets
        self.space = ContinuousSpace(100, 100, torus=False)
        self.target_space = ContinuousSpace(100, 100, torus=False)

        # data collection for plots
        self.datacollector = DataCollector(model_reporters={"Minimum Distance":minimum_distance, "Maximum Distance":maximum_distance, "Average Distance":average_distance, "Collisions":collisions, "Messags Distributed":messages_distributed, "Messags Centralized":messages_centralized, "Minimum Battery":minimum_battery, "Maximum Battery":maximum_battery, "Average Battery":average_battery, "Minimum Targets":minimum_targets, "Maximum Targets":maximum_targets, "Average Targets":average_targets})

        # execute agents
        # scheduler for agents and targets
        self.schedule = t.RandomActivation(self) # sequential, random order
        self.target_schedule = t.BaseScheduler(self) # sequential

        # place targets
        self.make_targets(targets)

        # place agents
        self.make_agents(population)

        # place charging point at center
        self.cp = self.space.center

        # pairwise distances
        self.agent_distances()

        # battery state of agents
        self.agent_batteries()

        # number of targets known to agents
        self.agent_targets()

        # run model
        self.running = True

        # collect initial data sample
        self.datacollector.collect(self)

    def step(self):
        """
        Execute one step of the simulation: move all agents.
        """
        # execute agent behavior
        self.schedule.step()

        # update targets
        self.target_schedule.step()
        self.update_targets()

        # gather statistics
        self.agent_distances()
        self.agent_batteries()
        self.agent_targets()
        self.datacollector.collect(self)

    def agent_distances(self):
        """
        Compute the pairwise distances between all agents.
        """
        self.density = [self.space.get_distance(pair[0].pos, pair[1].pos) for pair in it.combinations(self.schedule.agent_buffer(), 2)]

    def agent_batteries(self):
        """
        Create a list of battery levels in the swarm.
        """
        self.batteries = [a.battery for a in self.schedule.agent_buffer()]

    def agent_targets(self):
        """
        Create a list with number of known targets of each agent.
        """
        self.targets = [len(a.targets.targets) for a in self.schedule.agent_buffer()]

    def make_agents(self, population):
        """
        Create agents and place them at the center of the environment.

        # Parameters
        - population: Number of agents to create.
        """
        s = np.floor(np.sqrt(population))
        for i in range(population):
            x = self.space.center[0] - self.min_dist * s                        + 2 * self.min_dist * (i % s)
            y = self.space.center[1] - self.min_dist * np.floor(population / s) + 2 * self.min_dist * np.floor(i / s)
            pos = np.array((x, y))
            agent = SwarmAgent(i, self)
            self.space.place_agent(agent, pos)
            self.schedule.add(agent)

    def make_targets(self, num_targets):
        """
        Create targets and place them randomly in the environment.

        # Parameters
        - num_targets: Number of targets to create.
        """
        for self.target_id in range(num_targets):
            x = np.random.random() * self.target_space.width
            y = np.random.random() * self.target_space.height
            pos = np.array((x, y))
            target = Target(self.target_id, self)
            self.target_space.place_agent(target, pos)
            self.target_schedule.add(target)

    def update_targets(self):
        """
        Remove targets that have been tracked long enough and add new targets.
        """
        # remove completed targets
        removed = 0
        for t in self.target_schedule.agents:
            if t.state == "done":
                self.target_schedule.remove(t)
                self.target_space.remove_agent(t)
                removed += 1

        # add the same number of new targets
        for a in range(removed):
            self.target_id += 1
            x = np.random.random() * self.target_space.width
            y = np.random.random() * self.target_space.height
            pos = np.array((x, y))
            target = Target(self.target_id, self)
            self.target_space.place_agent(target, pos)
            self.target_schedule.add(target)
