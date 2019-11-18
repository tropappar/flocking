import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from .boid import Boid


class BoidFlockers(Model):
    '''
    A Mesa implementation of flocking agents inspired by https://doi.org/10.1109/IROS.2014.6943105.
    '''

    def __init__(self, population, width, height, vision,
                 flock_vel=      1.0,
                 accel_time=     10.0,
                 equi_dist=      1.0,
                 repulse_max=    0.1,
                 repulse_spring= 0.1,
                 align_frict=    1.0,
                 align_slope=    1.0,
                 align_min=      1.0,
                 wall_decay=     5.0,
                 wall_frict=     1.0,
                 wp_tolerance=   10.0):
        '''
        Create a new Flockers model.

        Args:
            population: Number of agents
            width, height: Size of the space.
            vision: How far around should each agents look for its neighbors
        '''
        self.population = population
        self.vision = vision
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.params = dict(flock_vel=flock_vel, accel_time=accel_time, equi_dist=equi_dist, repulse_max=repulse_max, repulse_spring=repulse_spring, align_frict=align_frict, align_slope=align_slope, align_min=align_min, wall_decay=wall_decay, wall_frict=wall_frict, wp_tolerance=wp_tolerance)
        self.make_agents()
        self.running = True

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
        self.schedule.step()
