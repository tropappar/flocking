#!/bin/python3


import numpy as np


class Behavior():
    """
    Generic behavior class.
    """
    def __init__(self, agent, movement):
        """
        Create a generic behavior.

        # Parameters
        - agent: Agent object that allows exchanging information.
        - movement: Movement model underlying the behavior.
        """
        # store parameters
        self.agent = agent

        # movement pattern
        self.movement = movement


class Repulsion(Behavior):
    """
    Behavior in which agents simply repulse from each other.
    """
    def __init__(self, agent, movement):
        """
        Create a repulsion behavior.

        # Parameters
        - agent: Agent object that allows exchanging information.
        - movement: Movement model underlying the behavior.
        """
        # initialize behavior model
        super().__init__(agent, movement)

        # configure repulsion behavior
        self.dist_critical = 2.5 # critical distance between agents, maximum repulsion
        self.dist_avoid = 5.0    # avoidance distance, repulsion active
        self.vel_avoid = 1.0     # target velocity magnitude during repulsion
        self.repulse_shape = 2    # function defining the repulsion (1: linear (default), 2: sine)

        # next waypoint to reach
        self.waypoint = self.agent.pos

        # environment boundaries
        self.coords = np.array([[self.agent.model.space.x_min,self.agent.model.space.y_min], [self.agent.model.space.x_max,self.agent.model.space.y_min], [self.agent.model.space.x_max,self.agent.model.space.y_max], [self.agent.model.space.x_min,self.agent.model.space.y_max]])

    def direction(self):
        """
        Compute direction from the agent to its desired goal waypoint.

        # Returns
        numpy array: Normalized, two-dimensional direction vector.
        """
        wp = self.movement.get_waypoint()
        dist = self.agent.model.space.get_distance(self.agent.pos, wp)
        return self.agent.model.space.get_heading(self.agent.pos, wp) / dist

    def repulsion(self):
        """
        Compute vector for repulsion from all other agents.

        # Returns
        - numpy array: Two-dimensional vector that lets the agents repulse from each other.
        - int: Number of agents repulsed from.
        """
        rep = []
        self.caution = False

        # look for close by neighbors
        for n in self.agent.all_neighbors:
            # distance and heading of neighbor
            dist = self.agent.model.space.get_distance(self.agent.pos, n.pos)
            dir = self.agent.model.space.get_heading(self.agent.pos, n.pos) / dist

            # maximum avoidance for very near neighbors
            if dist < self.dist_critical:
                rep.append(-dir)

            # sine based avoidance
            elif dist < self.dist_avoid:
                if (self.repulse_shape == 2):
                    rep.append(-dir * (0.5 - 0.5 * np.sin(np.pi / (self.dist_avoid - self.dist_critical) * (dist - 0.5 * (self.dist_avoid + self.dist_critical)))))
                else:
                    rep.append(-dir * (1 - (dist - self.dist_critical) / (self.dist_avoid - self.dist_critical)))

        # compute total repulsion
        agents = len(rep)
        rep = sum(rep)

        return rep, agents

    def velocity(self):
        """
        Compute total velocity of agent.

        # Returns
        numpy array: Two-dimensional velocity.
        """
        # waypoint to move to
        direction = self.direction()

        # repulsion from other agents
        repulsion_vec, neighbors = self.repulsion()

        # no repulsion
        if neighbors <= 0:
            return direction - self.agent.velocity

        # repulsion magnitude
        repulsion_mag = np.linalg.norm(repulsion_vec)

        # resulting velocity
        return (1 - repulsion_mag / neighbors) * (direction + repulsion_vec) - self.agent.velocity


class Flocking(Behavior):
    """
    Flocking behavior based on https://doi.org/10.1109/IROS.2014.6943105.
    """
    def __init__(self, agent, movement, formation):
        """
        Create a flocking behavior.

        # Parameters
        - agent: Agent object that allows exchanging information.
        - movement: Movement model underlying the behavior.
        - formation: Formation in which the agents move. Possibilities are
          * flock
          * grid
          * ring
          * line
          * star
        """
        # initialize behavior model
        super().__init__(agent, movement)

        # configure flocking behavior
        self.flock_vel = 1.0      # target velocity of the flock (v_flock)
        self.accel_time = 1.0     # characteristic time needed by the agent to reach the target velocity (tau)
        self.crit_dist = 2.5      # critical distance between agents, maximum repulsion
        self.equi_dist = 5.0      # equilibrium distance between agents (r_0)
        self.repulse_max = 0.5    # maximum acceleration due to repulsion between agents
        self.repulse_shape = 2    # function defining the repulsion (1: linear (default), 2: sine)
        self.align_frict = 0.3    # velocity alignment viscous friction coefficient (C_frict)
        self.align_slope = 30.0   # constant slope around equilibrium distance (r_2)
        self.align_min = 0.1      # minimum alignment between agents (r_1)
        self.wall_decay = 10.0    # softness of environment boundary as decay width (d)
        self.wall_frict = 1.0     # environment boundary viscous friction coefficient (C_shill)
        self.form_shape = 1.0     # strength of the shape forming velocity component (beta).
        self.form_track = 1.0     # strength of the tracking velocity component (alpha)
        self.form_decay = 10.0    # softness of shape (d)

        # formation of the flock
        self.formation = formation

        # next waypoint to reach
        self.waypoint = self.agent.pos

        # environment boundaries
        self.coords = np.array([[self.agent.model.space.x_min,self.agent.model.space.y_min], [self.agent.model.space.x_max,self.agent.model.space.y_min], [self.agent.model.space.x_max,self.agent.model.space.y_max], [self.agent.model.space.x_min,self.agent.model.space.y_max]])

    def alignment(self):
        """
        Compute acceleration to align velocities between agents.

        # Returns
        numpy array: Two-dimensional acceleration vector.
        """
        dist = self.equi_dist - self.align_slope
        return self.align_frict * sum([(n.velocity - self.agent.velocity) / max(self.agent.model.space.get_distance(self.agent.pos, n.pos) - dist, self.align_min) ** 2 for n in self.agent.neighbors])

    def dist_bound(self):
        """
        Compute the distance of the area bound to the coordinate system origin. Returns the distance between origin and the point on the area bound where the line from origin through the current agent pose intersects.

        # Returns
        float: Distance to area boundary.
        """
        # point on boundary
        bound = np.zeros(2)

        # distance of point from origin
        dist = 0.0

        # coordinates of line segment from center to pose
        p1 = self.agent.model.space.center
        p2 = self.agent.pos

        # find boundary that yields closest intersection point to the agent (i.e. the correct boundary)
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
            if self.agent.model.space.get_distance(p2, p) < dist or dist == 0.0:
                dist = self.agent.model.space.get_distance(p2, p)
                bound = p

        return self.agent.model.space.get_distance(p1, bound)

    def formation_flock(self):
        """
        Compute velocity that allows agents to stay in the flock.

        # Returns
        numpy array: Two-dimensional flocking velocity.
        """
        velocity = self.agent.model.space.get_heading(self.agent.pos, self.waypoint)
        return self.flock_vel * velocity / np.linalg.norm(velocity)

    def formation_grid(self):
        """
        Compute velocity that allows agents to form a grid.

        # Returns
        numpy array: Two-dimensional formation velocity.
        """
        dist = 0
        if len(self.agent.neighbors):
            # circle packing, use function fitted from data available at http://hydra.nat.uni-magdeburg.de/packing/cci/cci.html
            dist = self.equi_dist / 2 * 0.8135 * len(self.agent.neighbors)**-0.4775 - self.equi_dist / 2

        return self.formation_vel(self.waypoint, dist)

    def formation_ring(self):
        """
        Compute velocity that allows agents to form a circle.

        # Returns
        numpy array: Two-dimensional formation velocity.
        """
        # no neighbors, go directly to waypoint
        if len(self.agent.neighbors) == 0:
            pos = self.waypoint

        # one neighbor, each goes on one side of waypoint
        elif len(self.agent.neighbors) == 1:
            # specify ring
            center = self.waypoint
            radius = self.equi_dist / 2.0

            # direction of neighbor
            dir = self.agent.model.space.get_heading(center, self.agent.neighbors[0].pos) / self.agent.model.space.get_distance(center, self.agent.neighbors[0].pos)

            # go to opposite side
            pos = center - dir * radius

        # more neighbors, form ring around waypoint
        else:
            # specify ring
            center = self.waypoint
            radius = self.equi_dist / 2.0 / np.sin(np.pi / len(self.agent.neighbors))

            # two neighboring agents on ring
            d1 = -1
            d2 = -1
            n1 = None
            n2 = None
            # find closest neighbors
            for n in self.agent.neighbors:
                if d1 == -1 or self.agent.model.space.get_distance(self.agent.pos, n.pos) < d1:
                    d2 = d1
                    d1 = self.agent.model.space.get_distance(self.agent.pos, n.pos)
                    n2 = n1
                    n1 = n
                elif d2 == -1 or self.agent.model.space.get_distance(self.agent.pos, n.pos) < d2:
                    d2 = self.agent.model.space.get_distance(self.agent.pos, n.pos)
                    n2 = n

            # angle bisector: https://stackoverflow.com/questions/43435055/numerically-stable-angle-bisector-algorithm
            bis = None
            # unit vectors from center to neighbors
            n1_v = self.agent.model.space.get_heading(center, n1.pos) / self.agent.model.space.get_distance(center, n1.pos)
            n2_v = self.agent.model.space.get_heading(center, n2.pos) / self.agent.model.space.get_distance(center, n2.pos)
            # dot product of vectors
            dot = np.dot(n1_v, n2_v)
            # obtuse angle needs special handling for numeric precision
            if dot < 0:
                # difference of neighbor vectors
                bis_t = self.agent.model.space.get_heading(n2_v, n1_v) / self.agent.model.space.get_distance(n2_v, n1_v)
                # rotate n2_v bei 90°
                n2_v_r = [n2_v[1], -n2_v[0]]
                # determine direction
                dir = np.dot(n1_v, n2_v_r)
                # rotate by 90°
                if dir < 0:
                    bis = np.array([bis_t[1], -bis_t[0]])
                # rotate by 270°
                else:
                    bis = np.array([-bis_t[1], bis_t[0]])

            # acute angle
            else:
                # normalized sum of neighbor vectors
                bis = np.array(n1_v + n2_v) / np.linalg.norm(n1_v + n2_v)

            # intersect angle bisector and ring
            pos = center + bis * radius

        return self.formation_vel(pos, 0)

    def formation_vel(self, pos, dist):
        """
        Compute velocity to achieve a generic formation.

        # Parameters
        - pos: Position which the agent tries to reach.
        - dist: Distance which the agent keeps from pos.

        # Returns
        numpy array: Two-dimensional formation velocity.
        """
        shp_mag = np.linalg.norm(pos - self.agent.pos)

        # compute shape velocity
        v_shp = np.zeros(2)
        if shp_mag > 0.01:
            tf_shp = self.transfer(shp_mag, dist, self.form_decay)
            v_shp = self.form_shape * self.flock_vel * tf_shp * (pos - self.agent.pos) / shp_mag

        # compute distance between target and geometric mean of swarm
        x_com = self.movement.get_waypoint() - np.mean(np.array([n.pos for n in self.agent.neighbors+[self.agent]]), axis=0)
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

    def repulsion(self):
        """
        Compute acceleration from repulsive forces between agents.

        # Returns
        numpy array: Two-dimensional acceleration that lets the agents repulse from each other.
        """
        rep = []
        self.caution = False

        # look for close by neighbors
        for n in self.agent.all_neighbors:
            # distance and heading of neighbor
            dist = self.agent.model.space.get_distance(self.agent.pos, n.pos)
            dir = self.agent.model.space.get_heading(self.agent.pos, n.pos) / dist

            # maximum avoidance for very near neighbors
            if dist < self.crit_dist:
                rep.append(-dir)

            # sine based avoidance
            elif dist < self.equi_dist:
                if (self.repulse_shape == 2):
                    rep.append(-dir * (0.5 - 0.5 * np.sin(np.pi / (self.equi_dist - self.crit_dist) * (dist - 0.5 * (self.equi_dist + self.crit_dist)))))
                else:
                    rep.append(-dir * (1 - (dist - self.crit_dist) / (self.equi_dist - self.crit_dist)))

        # normalize to maximum repulsive acceleration
        rep = sum(rep)
        mag = np.linalg.norm(rep)

        if mag > 0:
            rep *= self.repulse_max / mag

        return rep

    def transfer(self, x, r, d):
        """
        Transfer function to smooth accelerations and velocities.

        # Parameters
        - x: Value for which to compute transfer function, e.g., position of agent.
        - r: Value below which the transfer function is zero.
        - d: Width of transfer interval. For values above r+d, the transfer function is one, i.e., maximal.

        # Returns
        float: Value between zero and one which transfers smoothly using the sine function.
        """
        if x <= r:
            return 0

        elif r + d <= x:
            return 1

        else:
            return 0.5 * np.sin(np.pi / d * (x - r) - np.pi / 2) + 0.5

    def velocity(self):
        """
        Compute total velocity of agent.

        # Returns
        numpy array: Two-dimensional velocity.
        """
        # waypoint to move to
        self.waypoint = self.movement.get_waypoint()

        # compute velocity for different formations
        vel = 0
        if self.formation == "grid":
            vel = self.formation_grid()
        elif self.formation == "ring":
            vel = self.formation_ring()
        elif self.formation == "line":
            pass
        elif self.formation == "star":
            pass
        else:
            vel = self.formation_flock()

        # total velocity for flocking
        return 1 / self.accel_time * (vel - self.agent.velocity) + (self.repulsion() + self.alignment() + self.wall())

    def wall(self):
        """
        Compute acceleration from repulsive forces of bounding virtual walls, i.e., environment boundaries.

        # Returns
        numpy array: Two-dimensional acceleration to stay within environment.
        """
        # distance to center
        center_dir = self.agent.model.space.center - self.agent.pos
        center_dist = np.linalg.norm(center_dir)

        # distance to wall
        wall_dist = self.dist_bound()

        # compute velocity requirement due to bounding area
        if center_dist == 0:
            v_wall = np.zeros(2)
        else:
            v_wall = self.flock_vel * center_dir / center_dist - self.agent.velocity

        # compute transfer function for smooth movement
        tf = self.transfer(center_dist, wall_dist-self.wall_decay, self.wall_decay) # subtract decay in order to stay within area bounds

        # total acceleration due to bounding area
        a_wall = self.wall_frict * tf * v_wall

        return a_wall
