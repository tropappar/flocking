import numpy as np
from timeit import default_timer as timer

from mesa import Agent


class Boid(Agent):
    '''
    A Boid-style flocker agent.

    The agent follows four behaviors to flock:
        Repulsion: avoiding getting too close to any other agent.
        Alignment: try to fly in the same direction as the neighbors.
        Formation: adjust velocity to stay in a predefined formation.
        Wall: repell from walls.
    '''
    def __init__(self, unique_id, model, pos, velocity, vision, formation, population, flock_vel, accel_time, equi_dist, repulse_max, repulse_spring, align_frict, align_slope, align_min, wall_decay, wall_frict, form_shape, form_track, form_decay, wp_tolerance):
        '''
        Create a new flocker agent.

        Args:
            unique_id: Unique agent identifier.
            model: The model to build upon.
            pos: Starting position.
            velocity: Initial velocity.
            vision: Radius to look around for nearby agents.
            formation: Formation of the agents.
            population: Number of agents.
            flock_vel: Target velocity of the flock (v_flock).
            accel_time: Characteristic time needed by the agent to reach the target velocity (tau).
            equi_dist: Equilibrium distance between agents (r_0).
            repulse_max: Maximum repulsion between agents (r_1).
            repulse_spring: Repulsion spring constant of half-spring (D).
            align_frict: Velocity alignment viscous friction coefficient (C_frict).
            align_slope: Constant slope around equilibrium distance (r_2).
            align_min: Minimum alignment between agents (r_1).
            wall_decay: Softness of wall as decay width (d).
            wall_frict: Bounding area viscous friction coefficient (C_shill).
            form_shape: Strength of the shape forming velocity component (beta).
            form_track: Strength of the tracking velocity component (alpha).
            form_decay: Softness of shape (d).
            wp_tolerance: The distance to the current waypoint below which the next waypoint of the path is selected.

        '''
        # initiale parent model
        super().__init__(unique_id, model)

        # read parameters
        self.pos = np.array(pos)
        self.velocity = velocity
        self.vision = vision
        self.formation = formation
        self.population = population
        self.flock_vel = flock_vel
        self.accel_time = accel_time
        self.equi_dist = equi_dist
        self.repulse_max = repulse_max
        self.repulse_spring = repulse_spring
        self.align_frict = align_frict
        self.align_slope = align_slope
        self.align_min = align_min
        self.wall_decay = wall_decay
        self.wall_frict = wall_frict
        self.form_shape = form_shape
        self.form_track = form_track
        self.form_decay = form_decay
        self.wp_tolerance = wp_tolerance

        # area coordinates
        self.coords = np.array([[self.model.space.x_min,self.model.space.y_min], [self.model.space.x_max,self.model.space.y_min], [self.model.space.x_max,self.model.space.y_max], [self.model.space.x_min,self.model.space.y_max]])

        # generate coverage path
        self.path = self.coverage_path()


    def step(self):
        '''
        Get the agent's neighbors, compute the new position, and move accordingly.
        Assumption: 1 step = 1 s
        '''
        self.neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)

        self.center = np.mean(np.array([n.pos for n in self.neighbors+[self]]), axis=0)

        # select next waypoint if flock is close enough to or past current waypoint
        self.coverage_waypoint()

        # compute velocity for different formations
        vel = 0
        if self.formation == "Grid":
            vel = self.formation_grid()
        elif self.formation == "Ring":
            vel = 0
        elif self.formation == "Line":
            vel = 0
        elif self.formation == "Star":
            vel = 0
        else:
            vel = self.flocking()

        # total velocity for flocking of agents
        self.velocity += 1 / self.accel_time * (vel - self.velocity) + (self.repulsion() + self.alignment() + self.wall())

        # compute new position
        new_pos = self.pos + self.velocity

        # move agent to new position
        self.model.space.move_agent(self, new_pos)

    def alignment(self):
        '''
        Compute acceleration to align velocities between agents.
        '''
        dist = self.equi_dist - self.align_slope
        return self.align_frict * sum([(n.velocity - self.velocity) / max(self.model.space.get_distance(self.pos, n.pos) - dist, self.align_min) ** 2 for n in self.neighbors])

    def coverage_path(self):
        '''
        Generate a path that allows the flock to sweep the area.
        '''
        x1 = self.model.space.center[0] / 2.0
        x2 = self.model.space.center[0] * 3.0 / 2.0
        y1 = self.model.space.center[1] / 2.0
        y2 = self.model.space.center[1] * 3.0 / 2.0

        # current waypoint
        self.wp = 0

        # wait a few iterations before selecting next waypoint to be synchronized with all agents
        self.wp_delay = 2

        return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

    def coverage_velocity(self):
        '''
        Compute velocity to reach the next waypoint of the coverage path.
        '''
        # return vector pointing towards next waypoint
        return self.model.space.get_heading(self.pos, self.path[self.wp])

    def coverage_waypoint(self):
        '''
        Select the next waypoint of the coverage path if the swarm is close or past the current one.
        '''
        # compute flock centroid
        flock = np.mean(np.array([n.pos for n in self.neighbors+[self]]), axis=0)

        # select next waypoint if flock is close enough to or past current waypoint
        if self.model.space.get_distance(flock, self.path[self.wp]) < self.wp_tolerance:
            if self.wp_delay > 0:
                self.wp_delay -= 1
            else:
                self.wp_delay = 2
                self.wp += 1

        # repeat path
        if self.wp >= len(self.path):
            self.wp = self.wp % len(self.path)

    def dist_bound(self):
        '''
        Compute the distance of the area bound to the coordinate system origin. A rectangular area is assumed. Returns the distance between origin and the point on the area bound where the line from origin through the current agent pose intersects.
        '''
        # distance of point from origin
        dist = 0.0

        # coordinates of line segment from center to pose
        p1 = self.model.space.center
        p2 = self.pos

        # find boundary that yields closest intersection point (i.e. the correct boundary)
        for i in range(0,len(self.coords)):
            # coordinates of boundary
            p3 = self.coords[i]
            p4 = self.coords[(i+1) % len(self.coords)]

            # compute point based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
            p = np.zeros(2)
            if (p1[0] - p2[0]) * (p3[1] - p4[1]) != (p1[1] - p2[1]) * (p3[0] - p4[0]): # only for non-parallel lines
                p[0] = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
                p[1] = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))

            # found closer point
            if self.model.space.get_distance(self.model.space.center, p) < dist or dist == 0.0:
                dist = self.model.space.get_distance(self.model.space.center, p)

        return dist

    def flocking(self):
        '''
        Compute velocity that allows agents to stay in the flock.
        '''
        velocity = self.coverage_velocity()
        return self.flock_vel * velocity / np.linalg.norm(velocity)

    def formation_vel(self, pos, dist):
        '''
        Compute velocity to achieve a generic formation.

        Args:
            pos: The position which the agent tries to reach.
            dist: The distance which the agent keeps from pos.
        '''
        shp_mag = np.linalg.norm(pos - self.pos)

        # compute shape velocity
        v_shp = np.zeros(2)
        if shp_mag > 0.01:
            tf_shp = self.transfer(shp_mag, dist, self.form_decay)
            v_shp = self.form_shape * self.flock_vel * tf_shp * (pos - self.pos) / shp_mag

        # compute distance between center of mass and target
        x_com = self.path[self.wp] - self.center
        com_mag = np.linalg.norm(x_com)

        # compute target tracking velocity
        v_trg = np.zeros(2)
        if com_mag > 0.01:
            tf_track = self.transfer(com_mag, dist, self.form_decay)
            v_trg = self.form_track * self.flock_vel * tf_track * x_com / com_mag

        # combine velocities
        v_formation = v_shp + v_trg
        vel_mag = np.linalg.norm(v_formation)
        if vel_mag > self.flock_vel:
            v_formation *= self.flock_vel / vel_mag

        return v_formation


    def formation_grid(self):
        '''
        Compute velocity that allows agents to form a grid.
        '''
        # circle packing, use function fitted from data available at http://hydra.nat.uni-magdeburg.de/packing/cci/cci.html
        dist = self.equi_dist / 2 * 0.8135 * self.population**-0.4775 - self.equi_dist / 2

        return self.formation_vel(self.center, dist)

    def potentials(self):
        '''
        Compute pair potentials between neighbors.
        '''
        for n in self.neighbors:
            dist = self.model.space.get_distance(self.pos, n.pos)
            if dist < self.equi_dist:
                yield min(self.repulse_max, self.equi_dist - dist) * self.model.space.get_heading(self.pos, n.pos) / dist

    def repulsion(self):
        '''
        Compute acceleration from repulsive forces between agents.

        Returns:
            The acceleration that lets the agents repulse from each other.
        '''
        return -self.repulse_spring * sum(list(self.potentials()))

    def transfer(self, x, r, d):
        '''
        A transfer function to smooth the accelerations and velocities.

        Args:
            x: The x value of the transfer function, e.g., position of agent.
            r: The x value below which the transfer function is zero.
            d: The width of x values in which the transfer function is active. For values above r+d, the transfer function is one, i.e., maximal.

        Returns: A value between zero and one which transfers smoothly using the sine function.
        '''
        if x <= r:
            return 0

        elif r + d <= x:
            return 1

        else:
            return 0.5 * np.sin(np.pi / d * (x - r) - np.pi / 2) + 0.5

    def wall(self):
        '''
        Compute acceleration from repulsive forces of bounding virtual walls, i.e., environment boundaries.
        '''
        # distance to center
        center_dist = self.model.space.center - self.pos

        # distance to wall
        wall_dist = self.dist_bound()

        # compute velocity requirement due to bounding area
        if np.linalg.norm(center_dist) == 0:
            v_wall = np.zeros(2)
        else:
            v_wall = self.flock_vel * center_dist / np.linalg.norm(center_dist) - self.velocity

        # compute transfer function for smooth movement
        tf = self.transfer(np.linalg.norm(center_dist), wall_dist-self.wall_decay, self.wall_decay) # subtract decay in order to stay within area bounds

        # total acceleration due to bounding area
        a_wall = self.wall_frict * tf * v_wall

        return a_wall
