from mesa.visualization.ModularVisualization import ModularServer

from .model import BoidFlockers
from .SimpleContinuousModule import SimpleCanvas
from mesa.visualization.UserParam import UserSettableParameter


def boid_draw(agent):
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Red"}
    #return {"Shape": "arrowHead", "scale": 2, "heading_x": agent.velocity[0], "heading_y": agent.velocity[1], "Filled": "true", "Color": "Red"}


boid_canvas = SimpleCanvas(boid_draw, 750, 750)

model_params = {
    "population":     5,
    "width":          100,
    "height":         100,
    "vision":         150,
    "flock_vel":      UserSettableParameter("slider", value=1, min_value=0.1, max_value=10, step=0.1, name="v<sub>flock</sub>: flock velocity", description="Target velocity of the flock."),
    "accel_time":     UserSettableParameter("slider", value=10, min_value=1, max_value=100, step=1, name="&tau;: acceleration time", description="Characteristic time needed by the agent to reach the target velocity."),
    "equi_dist":      UserSettableParameter("slider", value=1, min_value=0.1, max_value=10, step=0.1, name="r<sub>0</sub>: distance", description="Equilibrium distance between agents."),
    "repulse_max":    UserSettableParameter("slider", value=0.1, min_value=0.01, max_value=1, step=0.01, name="r<sub>1</sub>: max repulsion", description="Maximum repulsion between agents."),
    "repulse_spring": UserSettableParameter("slider", value=0.1, min_value=0.01, max_value=1, step=0.01, name="D: repulsion spring", description="Repulsion spring constant of half-spring."),
    "align_frict":    UserSettableParameter("slider", value=1, min_value=0.1, max_value=10, step=0.1, name="C<sub>frict</sub>: alignment friction", description="Velocity alignment viscous friction coefficient."),
    "align_slope":    UserSettableParameter("slider", value=1, min_value=0.1, max_value=10, step=0.1, name="r<sub>2</sub>: alignment slope", description="Constant slope around equilibrium distance."),
    "align_min":      UserSettableParameter("slider", value=1, min_value=0.1, max_value=10, step=0.1, name="r<sub>1</sub>: min alignment", description="Minimum alignment between agents."),
    "wall_decay":     UserSettableParameter("slider", value=10, min_value=1, max_value=100, step=1, name="d: wall decay", description="Softness of wall as decay width."),
    "wall_frict":     UserSettableParameter("slider", value=1, min_value=0.1, max_value=10, step=0.1, name="C<sub>shill</sub>: wall friction", description="Bounding area viscous friction coefficient."),
    "wp_tolerance":   UserSettableParameter("slider", value=10, min_value=1, max_value=100, step=1, name="waypoint tolerance", description="The distance to the current waypoint below which the next waypoint of the path is selected."),
}

server = ModularServer(BoidFlockers, [boid_canvas], "Flocking", model_params)
