#!/bin/python3

import numpy as np
import itertools as it

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

from flocking.agent import Boid

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

def messages_distributed(model):
    '''
    Compute the number of messages exchanged in the network with distributed communication.
    '''
    return model.population

def messages_centralized(model):
    '''
    Compute the number of messages exchanged in the network with centralized communication.
    '''
    return model.population + sum([2 for d in model.density if d <= model.vision])

class BoidFlockers(Model):
    '''
    A Mesa implementation of flocking agents inspired by https://doi.org/10.1109/IROS.2014.6943105.
    '''

    def __init__(self, formation, population, width, height, vision, min_dist, flock_vel, accel_time, equi_dist, repulse_max, repulse_spring, align_frict, align_slope, align_min, wall_decay, wall_frict, form_shape, form_track, form_decay, wp_tolerance):
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
        self.center = np.array((np.round(width/2.0), np.round(height/2.0)))
        self.size = np.array((width, height))
        self.grid = MultiGrid(width, height, torus=False)
        self.vision = vision
        self.min_dist = min_dist
        self.params = dict(formation=formation,population=population,flock_vel=flock_vel, accel_time=accel_time, equi_dist=equi_dist, repulse_max=repulse_max, repulse_spring=repulse_spring, align_frict=align_frict, align_slope=align_slope, align_min=align_min, wall_decay=wall_decay, wall_frict=wall_frict, form_shape=form_shape, form_track=form_track, form_decay=form_decay, wp_tolerance=wp_tolerance)

        # data collection for plots
        self.datacollector = DataCollector(model_reporters={"Minimum Distance":minimum_distance, "Maximum Distance":maximum_distance, "Average Distance":average_distance, "Collisions":collisions, "Messags Distributed":messages_distributed, "Messags Centralized":messages_centralized})

        # execute agents sequentially in a random order
        self.schedule = RandomActivation(self)

        # place agents
        self.make_agents()

        # pairwise distances
        self.agent_distances()

        # run model
        self.running = True

        # collect initial data sample
        # self.datacollector.collect(self)

    def agent_distances(self):
        '''
        Compute the pairwise distances between all agents.
        '''
        self.density = [self.get_distance(pair[0].pos, pair[1].pos) for pair in it.combinations(self.schedule.agent_buffer(), 2)]

    def make_agents(self):
        '''
        Create self.population agents and place it at the center of the environment.
        '''
        s = np.floor(np.sqrt(self.population))
        for i in range(self.population):
            x = int(np.round(self.center[0] - self.min_dist * s                             + 2 * self.min_dist * (i % s)))
            y = int(np.round(self.center[1] - self.min_dist * np.floor(self.population / s) + 2 * self.min_dist * np.floor(i / s)))
            pos = (x, y)
            velocity = np.array([0,1.0])
            boid = Boid(i, self, pos, velocity, self.vision, **self.params)
            self.grid.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self):
        '''
        Execute one step of the simulation.
        '''
        self.schedule.step()
        self.agent_distances()
        # self.datacollector.collect(self)

    def get_distance(self, p1, p2, method="manhattan"):
        '''
        Compute the distance between two points in space.

        Args:
            p1: Coordinate tuple of the first point.
            p2: Coordinate tuple of the second point.
            method: The metric for how to compute the distance, can be euclidean or manhattan.

        Returns:
            The distance as floating point value.
        '''
        # convert tuples to numpy array
        one = np.array(p1)
        two = np.array(p2)

        # compute distance
        if method == "euclidean":
            distance = np.hypot(one, two)
        elif method == "manhattan":
            distance = abs(two[0] - one[0]) + abs(two[1] - one[1])
        else:
            print("Unknown distance function!")
            distance = 0

        return distance

    def get_heading(self, p1, p2):
        '''
        Compute the heading vector between two points, accounting for toroidal space.

        Args:
            p1: Coordinate tuple of the first point.
            p2: Coordinate tuple of the second point.
        '''
        # convert tuples to numpy array
        one = np.array(p1)
        two = np.array(p2)

        # account for toroidal space
        if self.grid.torus:
            one = (one - self.center) % self.size
            two = (two - self.center) % self.size
        
        # compute heading
        heading = two - one

        return heading
