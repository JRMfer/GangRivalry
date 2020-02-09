#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains several representation of agents for simulation regarding
gang rivalry in Hollenbeck. The first agent is just a simple Gang representation
containing primary information. The other agents represent (individual) members
of a specific gang. The main difference between these agents is their mobility
algorithm.
"""

# Import built-in libraries
import math
import random

# Import libraries
import numpy as np
from mesa import Agent

class Gang(object):
    def __init__(self, number, coords, size):
        """
        Gang represents a node with a id ,
        size of members and coordinates

        Input:
            number: id of gang
            coords: coordinates set space gang
            size: members in gang
        Output:
            Node representation of Gang with id
        """

        self.number = number
        self.coords = coords
        self.size = size


class BM(Agent):
    """
    Representation of an agent that walks according a Brownian Motion algorithm.
    """
    def __init__(self, unique_id, model, pos, number, beta):
        """
        Initialize each agent with a unique id, model, the position of their
        home location, gang number and beta (probabilty to cross one boundary)
        """

        super().__init__(unique_id, model)
        self.pos = pos
        self.number = number
        self.beta = beta

    def random_move(self):
        """
        Human mobility algorithm (Brownian Motion)
        """

        # Start algorithm
        while True:

            # Unpack current position, and determine new position
            x, y = self.pos
            x1 = x + np.random.normal()
            x2 = y + np.random.normal()
            new_pos = (x1, x2)

            # Checks if new position is in area. If not algorithm starts over
            if not self.model.area.out_of_bounds(new_pos):

                # Get the corresponding new region
                new_pos_int = (int(x1), int(x2))
                new_region = self.model.config.areas[new_pos_int]

                # If new region is a boundary then algrotihm starts over
                if new_region == 24:
                    continue

                # Determines the amount of boundaries the agent has to cross
                old_region = self.model.config.areas[(int(x), int(y))]
                borders_crossed = self.model.config.boundaries[old_region,
                                                                new_region]

                # Checks if agent wants to cross that many boundaries.
                # If so, make move
                if random.uniform(0, 1) < self.beta ** borders_crossed:
                    self.model.area.move_agent(self, new_pos)

                # End algorithm
                break

class GangMemberB(BM):
    def __init__(self, unique_id, model, pos, number, beta, vision):
        super().__init__(unique_id, model, pos, number, beta)
        self.vision = vision

    def step(self):
        """
        Moves the gang member according a Brownian Motion algorithm.
        """
        self.random_move()
        self.check_rivals()

    def check_rivals(self):
        """
        Check if any rivals are within the vision of the agent. If so,
        update rivalry matrix.
        """
        poss_agents = self.model.area.get_neighbors(self.pos, self.vision,
                                                    include_center=True)

        for agent in poss_agents:
            if self.number != agent.number:
                self.model.update_rivalry(self, agent)


class SBLN(Agent):
    """
    An agent that moves according a Semi-Biased Levy walk (flight) model. This
    agent determines its bias based on all the set space of each gang and on
    the current status of the rivalry matrix.
    """

    def __init__(self, unique_id, model, pos, number, min_jump,
                    weight_home, bounded_pareto, kappa, beta):
        """
        Each agent is initialized at its home location (set space) and
        the number of gang, a minimum jump length, a certaing weight constant
        toward the home location and certain constant values for drawing random
        numbers from a Von Mises distribution (bounded_pareto and kappa) and a
        probabilty for crossing boundaries.
        """

        super().__init__(unique_id, model)
        self.pos = pos
        self.number = number
        self.min_jump = min_jump
        self.weight_home = weight_home
        self.bounded_pareto = bounded_pareto
        self.kappa = kappa
        self.beta = beta

    def move_SBLN(self):
        """
        Semi-biased Levy walking algorithm
        """

        while True:

            # First determines maximum possible jump lenght
            x, y = self.pos
            road_dens = self.model.config.road_dens[int(y), int(x)]
            H = ((1 - road_dens) * self.model.upper_max_jump
                 + self.model.lower_max_jump)

            # Inverse random sampling bounded pareto distribution (jump length)
            L, k = self.min_jump, self.bounded_pareto
            U = random.uniform(0, 1)
            jump = ((-(U * H ** k - U * L ** k - H ** k) / (H ** k * L ** k))
                    ** (- 1 / k))

            # Copies gang information for coordinates set spaces
            gang_info = self.model.config.gang_info.copy()
            bias_home_x, bias_home_y = self.bias_home(gang_info)
            bias_rivals_x, bias_rivals_y = self.bias_rivals(gang_info)

            # Determines overall bias
            bias_x = bias_home_x + bias_rivals_x
            bias_y = bias_home_y + bias_rivals_y
            bias = 0
            if bias_x:
                bias = math.atan(bias_y / bias_x)
            else:
                bias = math.pi / 2

            # Determine  angle direction movement agent
            angle_bias = random.vonmisesvariate(bias, self.kappa)
            change_x = jump * math.cos(angle_bias)
            change_y = jump * math.sin(angle_bias)
            old_region = self.model.config.areas[(int(x), int(y))]

            # Determine the new postion
            x += change_x
            y += change_y

            # Ensure new postion is inside area
            if not self.model.area.out_of_bounds((x, y)):

                # Get new region, if new region is boundary then algorithms starts over
                new_region = self.model.config.areas[(int(x), int(y))]
                if new_region == 24:
                    continue

                # Determine the amount of boundaries the agent needs to cross,
                # and the chance the agent will make this move
                boundaries = self.model.config.boundaries[
                                    old_region,
                                    new_region
                                    ]
                chance = self.beta ** boundaries

                # Checks if agent makes the move.
                if random.uniform(0, 1) < chance:
                    self.model.area.move_agent(self, (x, y))

                # End algorithm
                break

    def bias_home(self, gang_info):
        """
        Determines the bias towards the member's own set space.
        Input:
            gang_info = list of Gang objects
        Output:
            A tuple containing the bias in the x and y coordinates towards home.
        """

        # Find norm "home vector"
        x_home, y_home = gang_info[self.number].coords
        bias_home_x = x_home - self.pos[0]
        bias_home_y = y_home - self.pos[1]
        norm_home = math.sqrt(bias_home_x * bias_home_x +
                              bias_home_y * bias_home_y)

        # Determine bias towards home
        if norm_home:
            bias_home_x /= norm_home
            bias_home_y /= norm_home
            rules_home = self.weight_home * norm_home
            bias_home_x *= rules_home
            bias_home_y *= rules_home
        else:
            bias_home_x, bias_home_y = 0, 0

        # Remove own gang for the rest of the calculations
        gang_info.pop(self.number)

        return bias_home_x, bias_home_y

    def bias_rivals(self, gang_info):
        """
        Determines the bias towards the rivals' set spaces.
        Input:
            gang_info = list of Gang objects
        Output:
            A tuple containing the bias in the x and y coordinates towards the rivals.
        """

        # Determine the total interaction of member's gang
        bias_rivals_x, bias_rivals_y = 0.0, 0.0
        contact_all_rivals = self.model.rivalry_matrix[self.number, :].sum()

        # Starts calculations for bias towards all rivals
        for gang in gang_info.values():

            # Determines the amount of interaction of this member with another gang
            contact_rival = self.model.rivalry_matrix[self.number,
                                                      gang.number]

            # Determine the normilzed vectory pointing to rival's set space
            rival_x = gang.coords[0] - self.pos[0]
            rival_y = gang.coords[1] - self.pos[1]
            norm_rival = math.sqrt(rival_x * rival_x + rival_y * rival_y)

            # if there has been any contact then determine bias towards this specific
            # gang
            if contact_all_rivals and norm_rival:
                weight_bias = (-(contact_rival / contact_all_rivals)
                               / norm_rival)
                bias_rivals_x += weight_bias * rival_x / norm_rival
                bias_rivals_y += weight_bias * rival_y / norm_rival

        return bias_rivals_x, bias_rivals_y


class GangMember_SBLN(SBLN):
    def __init__(self, unique_id, model, pos, number,
                 min_jump, weight_home, bounded_pareto, kappa, vision, beta):
        super().__init__(unique_id, model, pos, number,
                         min_jump, weight_home, bounded_pareto, kappa, beta)
        self.vision = vision

    def step(self):
        """
        Moves the gangster according a semi biased Levy model.
        """
        self.move_SBLN()
        self.check_rivals()

    def check_rivals(self):
        """
        After moving checks for rivals
        and if found updates rivalry matrix.
        """

        poss_agents = self.model.area.get_neighbors(self.pos, self.vision,
                                                    include_center=True)

        for agent in poss_agents:
            if self.number != agent.number:
                self.model.update_rivalry(self, agent)


class GRAV(SBLN):
    """
    An agent that moves according a gracity model. This agent determines its bias
    based not only on all the set space of each gang and on the current status
    of the rivalry matrix, but also on the size of each gang.
    """

    def __init__(self, unique_id, model, pos, number,
                 min_jump, weight_home, bounded_pareto, kappa, beta):
        super().__init__(unique_id, model, pos, number,
                         min_jump, weight_home, bounded_pareto, kappa, beta)
        self.norm_size = self.model.norm_gang_size[number]

    def move_SBLN(self):
        """
        Gravity walking algorithm
        """

        while True:

            # First determines maximum possible jump lenght
            x, y = self.pos
            road_dens = self.model.config.road_dens[int(y), int(x)]
            H = ((1 - road_dens) * self.model.upper_max_jump
                 + self.model.lower_max_jump)

            # Inverse random sampling bounded pareto distribution (jump length)
            L, k = self.min_jump, self.bounded_pareto
            U = random.uniform(0, 1)
            jump = ((-(U * H ** k - U * L ** k - H ** k) / (H ** k * L ** k))
                    ** (- 1 / k))

            # Copies gang information for coordinates set spaces
            gang_info = self.model.config.gang_info.copy()
            bias_home_x, bias_home_y = self.bias_home(gang_info)
            bias_rivals_x, bias_rivals_y = self.bias_rivals(gang_info)

            # Determines bias agent
            bias_x = bias_home_x + bias_rivals_x
            bias_y = bias_home_y + bias_rivals_y
            bias = 0
            if bias_x:
                bias = math.atan(bias_y / bias_x)
            else:
                bias = math.pi / 2

            # Determine  angle direction movement agent
            angle_bias = random.vonmisesvariate(bias, self.kappa)
            change_x = jump * math.cos(angle_bias)
            change_y = jump * math.sin(angle_bias)
            old_region = self.model.config.areas[(int(x), int(y))]

            # Determines new position of agent
            x += change_x
            y += change_y

            # Ensures new position is valid
            if not self.model.area.out_of_bounds((x, y)):

                # Retrieve new region and ensures it is not a boundary.
                # If boundary algorithm starts over
                new_region = self.model.config.areas[(int(x), int(y))]
                if new_region == 24:
                    continue

                # Determines the amount of boundaries to cross and the
                # corresponding chance an agent will make this move.
                boundaries = self.model.config.boundaries[old_region,
                                                            new_region]
                chance = self.beta ** boundaries

                # Checks if agent makes move.
                if random.uniform(0, 1) < chance:
                    self.model.area.move_agent(self, (x, y))

                # End algorithm
                break

    def bias_home(self, gang_info):
        """
        Determines the bias towards the member's own set space.
        Input:
            gang_info = list of Gang objects
        Output:
            A tuple containing the bias in the x and y coordinates towards home.
        """

        # Find norm "home vector"
        x_home, y_home = gang_info[self.number].coords
        bias_home_x = x_home - self.pos[0]
        bias_home_y = y_home - self.pos[1]
        norm_home = math.sqrt(bias_home_x * bias_home_x +
                              bias_home_y * bias_home_y)

        # Determine bias towards home
        if norm_home:
            bias_home_x /= norm_home
            bias_home_y /= norm_home
            rules_home = self.weight_home * (2 - self.norm_size) * norm_home
            bias_home_x *= rules_home
            bias_home_y *= rules_home
        else:
            bias_home_x, bias_home_y = 0, 0

        # Remove own gang for the rest of the calculations
        gang_info.pop(self.number)

        return bias_home_x, bias_home_y

    def bias_rivals(self, gang_info):
        """
        Determines the bias towards the rivals' set spaces.
        Input:
            gang_info = list of Gang objects
        Output:
            A tuple containing the bias in the x and y coordinates towards the rivals.
        """

        # Determines total interaction of member's gang
        bias_rivals_x, bias_rivals_y = 0.0, 0.0
        contact_all_rivals = self.model.rivalry_matrix[self.number, :].sum()

        # Start calculation
        gang_inf = list(gang_info.values())
        for i in range(len(gang_inf)):

            # Determines the normalized vector pointing towards a specific gang
            contact_rival = self.model.rivalry_matrix[self.number, i]
            rival_x = gang_inf[i].coords[0] - self.pos[0]
            rival_y = gang_inf[i].coords[1] - self.pos[1]
            norm_rival = math.sqrt(rival_x * rival_x + rival_y * rival_y)

            # If contact and not already on set space of rival then determine bias
            if contact_all_rivals and norm_rival:
                weight_bias = (-(contact_rival / contact_all_rivals)
                               / norm_rival) * (1 + self.model.norm_gang_size[i])
                bias_rivals_x += weight_bias * rival_x / norm_rival
                bias_rivals_y += weight_bias * rival_y / norm_rival

        return bias_rivals_x, bias_rivals_y

class GangMemberG(GRAV):
    def __init__(self, unique_id, model, pos, gang,
                 min_jump, weight_home, bounded_pareto, kappa, vision, beta):
        super().__init__(unique_id, model, pos, gang,
                         min_jump, weight_home, bounded_pareto, kappa, beta)
        self.vision = vision

    def step(self):
        """
        Moves the gangster according a semi biased Levy model.
        """
        self.move_SBLN()
        self.check_rivals()

    def check_rivals(self):
        """
        After moving checks for rivals
        and if found updates rivalry matrix.
        """

        poss_agents = self.model.area.get_neighbors(self.pos, self.vision,
                                                    include_center=True)

        for agent in poss_agents:
            if self.number != agent.number:
                self.model.update_rivalry(self, agent)
