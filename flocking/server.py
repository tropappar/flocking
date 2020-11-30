#!/bin/python3

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from flocking.SimpleContinuousModule import SimpleCanvas
from flocking.model import BoidFlockers


def boid_draw(agent):
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Red"}
    #return {"Shape": "arrowHead", "scale": 2, "heading_x": agent.velocity[0], "heading_y": agent.velocity[1], "Filled": "true", "Color": "Red"}

model_params = {
    "formation":      UserSettableParameter("choice", value="Grid", choices=["Flock", "Grid", "Ring", "Line", "Star"], name="formation", description="Formation in which the agents move."),
    "population":     UserSettableParameter("slider", value=30,  min_value=1,    max_value=256, step=1,    name="population size", description="Number of agents in the flock."),
    "width":          100,
    "height":         100,
    "vision":         UserSettableParameter("slider", value=50,  min_value=1,    max_value=150, step=1,    name="interaction range", description="The range within which agents interact."),
    "min_dist":       UserSettableParameter("slider", value=1,   min_value=0.1,  max_value=10,  step=0.1,  name="minimum distance", description="Minimum allowed distance between agents before a collision occurs. This is only used for statistics."),
    "flock_vel":      UserSettableParameter("slider", value=1,   min_value=0.1,  max_value=10,  step=0.1,  name="v<sub>flock</sub>: flock velocity", description="Target velocity of the flock."),
    "accel_time":     UserSettableParameter("slider", value=1,   min_value=0.1,  max_value=10,  step=0.01, name="&tau;: acceleration time", description="Characteristic time needed by the agent to reach the target velocity."),
    "equi_dist":      UserSettableParameter("slider", value=5,   min_value=0.1,  max_value=10,  step=0.1,  name="r<sub>0</sub>: distance", description="Equilibrium distance between agents."),
    "repulse_max":    UserSettableParameter("slider", value=10,  min_value=0.01, max_value=50,  step=0.01, name="r<sub>1</sub>: max repulsion", description="Maximum repulsion between agents."),
    "repulse_spring": UserSettableParameter("slider", value=0.5, min_value=0.01, max_value=10,  step=0.01, name="D: repulsion spring", description="Repulsion spring constant of half-spring."),
    "align_frict":    UserSettableParameter("slider", value=0.3, min_value=0.1,  max_value=10,  step=0.1,  name="C<sub>frict</sub>: alignment friction", description="Velocity alignment viscous friction coefficient."),
    "align_slope":    UserSettableParameter("slider", value=30,  min_value=0.1,  max_value=50,  step=0.1,  name="r<sub>2</sub>: alignment slope", description="Constant slope around equilibrium distance."),
    "align_min":      UserSettableParameter("slider", value=0.1, min_value=0.1,  max_value=10,  step=0.1,  name="r<sub>1</sub>: min alignment", description="Minimum alignment between agents."),
    "wall_decay":     UserSettableParameter("slider", value=10,  min_value=1,    max_value=100, step=1,    name="d: wall decay", description="Softness of wall as decay width."),
    "wall_frict":     UserSettableParameter("slider", value=1,   min_value=0.1,  max_value=10,  step=0.1,  name="C<sub>shill</sub>: wall friction", description="Bounding area viscous friction coefficient."),
    "form_shape":     UserSettableParameter("slider", value=1,   min_value=0.1,  max_value=1,   step=0.1,  name="&beta;: shape percentage", description="Strength of the shape forming velocity component."),
    "form_track":     UserSettableParameter("slider", value=1,   min_value=0.1,  max_value=1,   step=0.1,  name="&alpha;: tracking percentage", description="Strength of the tracking velocity component."),
    "form_decay":     UserSettableParameter("slider", value=10,  min_value=0.1,  max_value=10,  step=0.1,  name="d: shape decay", description="Softness of shape."),
    "wp_tolerance":   UserSettableParameter("slider", value=10,  min_value=1,    max_value=100, step=1,    name="waypoint tolerance", description="The distance to the current waypoint below which the next waypoint of the path is selected."),
}

boid_canvas = SimpleCanvas(boid_draw, 750, 750)

boid_messages = ChartModule([{"Label":"Messags Distributed", "Color":"Green"}, {"Label":"Messags Centralized", "Color":"Red"}])
boid_distance = ChartModule([{"Label":"Minimum Distance", "Color":"Red"}, {"Label":"Maximum Distance", "Color":"Green"}, {"Label":"Average Distance", "Color":"Black"}])
boid_collisions = ChartModule([{"Label":"Collisions", "Color":"Red"}])

server = ModularServer(BoidFlockers, [boid_canvas, boid_collisions, boid_distance, boid_messages], "Flocking", model_params)
