#!/bin/python3


from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from swarm.ContinuousModule import ContinuousCanvas
from swarm.model import Swarm
from swarm.agent import SwarmAgent
from swarm.target import Target


def draw_agent(agent, range=""):
    """
    Draw an agent in the canvas.

    # Parameters
    - agent: Agent to draw.
    """
    if agent is None:
        return

    # default values
    portrayal = {"Filled": "true", "Color": "Black"}
    portrayal["r"] = 0.25 / (agent.model.space.x_max - agent.model.space.x_min)
    portrayal["x"] = (agent.pos[0] - agent.model.space.x_min) / (agent.model.space.x_max - agent.model.space.x_min)
    portrayal["y"] = (agent.pos[1] - agent.model.space.y_min) / (agent.model.space.y_max - agent.model.space.y_min)

    # draw agent
    if type(agent) is SwarmAgent:
        if range == "interaction":
            portrayal["Color"] = "rgba(204, 0, 0, 0.1)"
            portrayal["r"] = agent.model.interaction_range / (agent.model.space.x_max - agent.model.space.x_min)
        elif range == "vision":
            if agent.targets.tracking:
                portrayal["Color"] = "rgba(0, 204, 0, 0.1)"
            else:
                portrayal["Color"] = "rgba(0, 0, 0, 0.1)"
            portrayal["r"] = agent.model.vision_range / (agent.model.space.x_max - agent.model.space.x_min)
        elif agent.state == "tracking":
            portrayal["Color"] = "Red"

    # draw target
    elif type(agent) is Target:
        portrayal["Color"] = "Green"

    return portrayal


# model parameters
params = {
    "debug":                  UserSettableParameter("checkbox", value=False,                                     name="debug output", description="Print debug output to the console."),
    "population":             UserSettableParameter("slider",   value=9,   min_value=1,  max_value=100,  step=1, name="population size", description="Number of agents in the swarm."),
    "targets":                UserSettableParameter("slider",   value=20,  min_value=1,  max_value=50,   step=1, name="targets", description="Number of targets."),
    "interaction_range":      UserSettableParameter("slider",   value=10,  min_value=1,  max_value=141,  step=1, name="interaction range", description="Radius in which agents interact."),
    "show_interaction_range": UserSettableParameter("checkbox", value=False,                                     name="interaction range", description="Visualize interaction range."),
    "vision_range":           UserSettableParameter("slider",   value=4,   min_value=1,  max_value=10,   step=1, name="vision range", description="Radius in which targets are detected."),
    "show_vision_range":      UserSettableParameter("checkbox", value=True,                                      name="vision range", description="Visualize vision range."),
    "help_range":             UserSettableParameter("slider",   value=3,   min_value=0,  max_value=10,   step=1, name="help range", description="Radius in which agents join others for tracking."),
    "min_agents":             UserSettableParameter("slider",   value=3,   min_value=1,  max_value=5,    step=1, name="minimum tracking agents", description="Minimum number of agents that should track a target."),
    "max_agents":             UserSettableParameter("slider",   value=7,   min_value=3,  max_value=15,   step=1, name="maximum tracking agents", description="Maximum number of agents that should track a target."),
    "tracking_steps":         UserSettableParameter("slider",   value=50,  min_value=0,  max_value=100,  step=5, name="tracking steps", description="Time steps that a target has to be tracked until it disappears."),
    "battery":                UserSettableParameter("slider",   value=250, min_value=50, max_value=1000, step=5, name="battery size", description="Time steps that an agent can work until it has to recharge."),
}

# environment of the agents
canvas = ContinuousCanvas(draw_agent, 750, 750)

# plots
messages = ChartModule([{"Label":"Messags Distributed", "Color":"Green"}, {"Label":"Messags Centralized", "Color":"Red"}])
targets = ChartModule([{"Label":"Minimum Targets", "Color":"Red"}, {"Label":"Maximum Targets", "Color":"Green"}, {"Label":"Average Targets", "Color":"Black"}])
battery = ChartModule([{"Label":"Minimum Battery", "Color":"Red"}, {"Label":"Maximum Battery", "Color":"Green"}, {"Label":"Average Battery", "Color":"Black"}])
distance = ChartModule([{"Label":"Minimum Distance", "Color":"Red"}, {"Label":"Maximum Distance", "Color":"Green"}, {"Label":"Average Distance", "Color":"Black"}])
collisions = ChartModule([{"Label":"Collisions", "Color":"Red"}])

# simulation server
server = ModularServer(Swarm, [canvas, targets, battery, collisions, distance, messages], "5G Playground: Communication in Swarms", params)
