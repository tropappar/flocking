#!/bin/python3

from mesa.visualization.ModularVisualization import VisualizationElement


class ContinuousCanvas(VisualizationElement):
    """
    Canvas for continuous space.
    """
    local_includes = ["swarm/continuous_canvas.js"]

    def __init__(self, portrayal_method, canvas_height=500, canvas_width=500):
        '''
        Instantiate a new canvas.

        # Parameters
        - portrayal_method_agents: Method for drawing agents.
        - portrayal_method_targets: Method for drawing targets.
        - canvas_height: Height of canvas.
        - canvas_width: Width of canvas.
        '''
        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = ("new Continuous_Module({0}, {1})".format(self.canvas_width, self.canvas_height))
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        """
        Draw agents and targets.

        # Parameters
        - model: Model containing agents and targets.

        # Returns
        list: JSON visualization data.
        """
        space_state = []

        # draw agents
        for agent in model.schedule.agents:
            if model.show_interaction_range:
                space_state.append(self.portrayal_method(agent, "interaction"))
            if model.show_vision_range:
                space_state.append(self.portrayal_method(agent, "vision"))
            space_state.append(self.portrayal_method(agent))

        # draw targets
        for target in model.target_schedule.agents:
            space_state.append(self.portrayal_method(target))

        return space_state
