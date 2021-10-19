#!/bin/python3


from copy import copy

from mesa import Agent


class Target(Agent):
    """
    A target agent object for two purposes:
    1. It is used by the Mesa space module and placed in the environment.
    2. It is used by the agents to keep track of existing agents.
    """
    def __init__(self, unique_id, model):
        """
        Create a new target.

        # Parameters
        - unique_id: Unique target identifier.
        - model: Model to build upon.
        """
        # initialize agent model
        super().__init__(unique_id, model)

        # initially targets are unknown
        self.state = "unknown"

        # target information update time stamp
        self.stamp = 0

        # tracking agents
        self.agents = set()

        # time steps that the target has been tracked
        self.tracked = 0

        # time steps that a target has to be tracked until done
        self.track = self.model.tracking_steps

    def step(self):
        """
        Update targets.
        """
        # count number of time steps the target has been tracked
        if len(self.agents):
            self.tracked += 1
        else:
            self.tracked = 0

        # tracked long enough, vanish
        if self.tracked > self.track:
            self.state = "done"
            # inform tracking agents
            for a in self.agents:
                a.targets.targets[self.unique_id].state = "done"


class Targets():
    """
    A collection of targets managed by an agent.
    """
    def __init__(self, agent):
        """
        Create a new target collection.

        # Parameters
        - agent: Agent that keeps track of the targets.
        """
        # the agent owning this collection
        self.agent = agent

        # local dict of targets
        self.targets = {}

        # time steps without updates after which a target is lost
        self.timeout = 0

        # whether there is a target being tracked by this agent
        self.tracking = False

        # whether there is a target not being tracked by this agent
        self.help_needed = False

        # whether done tracking a target
        self.done_tracking = False

        # the target being tracked by this agent, call self.determine_target() to update
        self.target = None

    def determine_target(self):
        """
        Determine the target to track.
        - If this agent is tracking multiple targets, select closest.
        - If this agent is not tracking any target, select the next closest known target.
        - If no targets are known, return None.
        """
        # select new target
        target = None

        # currently tracking a target
        if self.tracking:
            # get all targets being tracked
            targets = [self.targets[t] for t in self.targets if self.targets[t].state == "tracked"]

            # there are multiple targets being tracked, select one
            if len(targets) > 1:
                # select closest target
                dist = -1
                for t in targets:
                    if dist == -1 or self.agent.model.space.get_distance(self.agent.pos, t.pos) < dist:
                        target = t
                        dist = self.agent.model.space.get_distance(self.agent.pos, t.pos)

                if self.agent.model.debug:
                    ids = [str(t.unique_id) for t in targets]
                    print("Agent {0}: Currently tracking multiple targets {1}, selecting target {2}".format(self.agent.unique_id, ",".join(ids), target.unique_id))

            # there is only one target being tracked
            elif len(targets) == 1:
                target = targets[0]
                if self.agent.model.debug:
                    print("Agent {0}: Currently tracking target {1} at ({2:.2f},{3:.2f})".format(self.agent.unique_id, target.unique_id, target.pos[0], target.pos[1]))

            # there is no target being tracked, this should not happen
            else:
                print("Agent {0}: Currently not tracking any target!".format(self.agent.unique_id))

        # not tracking
        else:
            # just done tracking, force transition to coverage
            if self.done_tracking:
                pass

            # known targets available
            elif self.help_needed:
                # get all known targets
                targets = [self.targets[t] for t in self.targets if self.targets[t].state == "known"]

                # select closest target
                dist = -1
                for t in targets:
                    if dist == -1 or self.agent.model.space.get_distance(self.agent.pos, t.pos) < dist:
                        target = t
                        dist = self.agent.model.space.get_distance(self.agent.pos, t.pos)

                # something went wrong, this should not happen
                if target is None:
                    print("Agent {0}: No target known, cannot perform tracking!".format(self.agent.unique_id))

        # allow global target object to keep track of tracking agents
        if self.target is not None and self.target.unique_id in self.agent.model.target_schedule._agents and self.agent in self.agent.model.target_schedule._agents[self.target.unique_id].agents:
            self.agent.model.target_schedule._agents[self.target.unique_id].agents.remove(self.agent)
        if target is not None and target.unique_id in self.agent.model.target_schedule._agents and self.agent not in self.agent.model.target_schedule._agents[target.unique_id].agents:
            self.agent.model.target_schedule._agents[target.unique_id].agents.add(self.agent)

        # store selected target
        self.target = target

    def remote_update(self, agent, targets):
        """
        Update information about targets in the local dictionary.

        # Parameters
        - agent: Sender of the update.
        - targets: Dictionary of target objects received from another agent.
        """
        # process received targets
        for t in targets:
            # add new target
            if t not in self.targets:
                self.targets[t] = Target(t, targets[t].model)
                self.targets[t].pos = targets[t].pos

            # update existing target
            if targets[t].state == "known":
                if len(targets[t].agents) < self.agent.model.max_agents:
                    self.help_needed = True
                if self.targets[t].state in ["unknown"]:
                    self.targets[t].state = "known"

            if targets[t].state == "tracked":
                if len(targets[t].agents) < self.agent.model.max_agents:
                    self.help_needed = True
                if self.targets[t].state in ["unknown"]:
                    self.targets[t].state = "known"

            if targets[t].state == "done":
                self.targets[t].state = "done"

    def update(self, targets=[]):
        """
        Update information about targets in the local dictionary.

        # Parameters
        - targets: List of target objects detected by this agent. Default empty.
        """
        # process detected targets
        for t in targets:
            # add new target
            if t.unique_id not in self.targets:
                self.targets[t.unique_id] = copy(t)

            # update target time stamp
            self.targets[t.unique_id].stamp = t.model.schedule.time

            # update target state
            if self.targets[t.unique_id].state in ["unknown", "known"]:
                self.targets[t.unique_id].state = "tracked"
                self.targets[t.unique_id].agents.add(self.agent)
                if self.agent.model.debug:
                    print("Agent {0}: Found target {1}".format(self.agent.unique_id, t.unique_id))

        # currently tracked target is done
        self.done_tracking = False
        if self.target and self.target.state == "done":
            if self.agent.model.debug:
                print("Agent {0}: Done tracking target {1}".format(self.agent.unique_id, self.target.unique_id))
            self.done_tracking = True

        # process existing targets
        self.tracking = False
        self.help_needed = False
        for id, target in self.targets.items():
            # target lost if no updates received within timeout
            if target.state == "tracked" and target.stamp + self.timeout < target.model.schedule.time:
                target.state = "known"
                if self.agent in target.agents:
                    target.agents.remove(self.agent)
                if self.agent.model.debug:
                    print("Agent {0}: Lost target {1}".format(self.agent.unique_id, id))

            # there is at least one target being tracked by this agent
            if target.state == "tracked":
                self.tracking = True

            # there is at least one known target not being tracked by this agent
            if target.state == "known":
                # the target is already done
                if self.agent.model.space.get_distance(self.agent.pos, target.pos) < self.agent.model.vision_range:
                    target.state = "done"
                elif len(target.agents) < self.agent.model.max_agents:
                    self.help_needed = True

    def prune(self):
        """
        Clean up list of targets by removing done targets for performance reasons.
        """
        self.targets = {id:target for id,target in self.targets.items() if target.state != "done"}
