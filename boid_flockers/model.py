import numpy as np
import itertools as it

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from .boid import Boid

def minimum_distance(model):
    '''
    Compute the minimum distance between any agent pair.
    '''
    return min(model.density)

def maximum_distance(model):
    '''
    Compute the minimum distance between any agent pair.
    '''
    return max(model.density)

def average_distance(model):
    '''
    Compute the minimum distance between any agent pair.
    '''
    return np.mean(model.density)

def collisions(model):
    '''
    Compute the number of agents being too close.
    '''
    return len([d for d in model.density if d < model.min_dist])

class BoidFlockers(Model):
    '''
    A Mesa implementation of flocking agents inspired by https://doi.org/10.1109/IROS.2014.6943105.
    '''

    def __init__(self, population, width, height, vision, min_dist, flock_vel, accel_time, equi_dist, repulse_max, repulse_spring, align_frict, align_slope, align_min, wall_decay, wall_frict, wp_tolerance):
        '''
        Create a new Flockers model.

        Args:
            population: Number of agents.
            width, height: Size of the space.
            vision: How far around should each agents look for its neighbors.
            min_dist: Minimum allowed distance between agents before a collision occurs. This is only used for statistics.
        '''
        # set parameters
        self.population = population
        self.space = ContinuousSpace(width, height, torus=False)
        self.vision = vision
        self.min_dist = min_dist
        self.params = dict(flock_vel=flock_vel, accel_time=accel_time, equi_dist=equi_dist, repulse_max=repulse_max, repulse_spring=repulse_spring, align_frict=align_frict, align_slope=align_slope, align_min=align_min, wall_decay=wall_decay, wall_frict=wall_frict, wp_tolerance=wp_tolerance)

        # data collection for plots
        self.datacollector = DataCollector(model_reporters={"Minimum Distance":minimum_distance, "Maximum Distance":maximum_distance, "Average Distance":average_distance, "Collisions":collisions})

        # execute agents sequentially in a random order
        self.schedule = RandomActivation(self)

        # place agents
        self.make_agents()

        # pairwise distances
        self.agent_distances()

        # run model
        self.running = True

        # collect initial data sample
        self.datacollector.collect(self)

    def agent_distances(self):
        '''
        Compute the pairwise distances between all agents.
        '''
        self.density = [self.space.get_distance(pair[0].pos, pair[1].pos) for pair in it.combinations(self.schedule.agent_buffer(), 2)]

    def make_agents(self):
        '''
        Create self.population agents, with random positions and starting headings.
        '''
        for i in range(self.population):
            x = self.space.center[0] / 2.0 - self.population + 2 * i
            y = self.space.center[1] / 2.0
            pos = np.array((x, y))
            velocity = np.array([0,1.0])
            boid = Boid(i, self, pos, velocity, self.vision, **self.params)
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
        '''
        Execute one step of the simulation.
        '''
        self.schedule.step()
        self.agent_distances()
        self.datacollector.collect(self)
