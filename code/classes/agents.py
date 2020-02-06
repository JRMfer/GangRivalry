# Agents
# Agents-based modeling
# University of Amsterdam
# Julien Fer, Inge Bieger, Jorien Lokker, Jasper, Heleen
#
# This script contains the functionality to represent gang members.

from mesa import Agent
import random
import math


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
        the Gang's id 
        """

        super().__init__(unique_id, model)
        self.pos = pos
        self.pos_int = (int(pos[0]), int(pos[1]))
        self.number = number
        self.min_jump = min_jump
        self.weight_home = weight_home
        self.bounded_pareto = bounded_pareto
        self.kappa = kappa
        self.beta = beta

    def move_SBLN(self):
        """
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
            x += change_x
            y += change_y

            if not self.model.area.out_of_bounds((x, y)):
                new_region = self.model.config.areas[(int(x), int(y))]
                if new_region == 24:
                    continue

                boundaries = self.model.config.boundaries[
                                    old_region, 
                                    new_region
                                    ]
                chance = self.beta ** boundaries
                
                if random.uniform(0, 1) < chance:
                    self.model.area.move_agent(self, (x, y))
                    # self.model.grid.move_agent(self, (int(x), int(y)))
                break

    def bias_home(self, gang_info):
        """
        """
        # Find norm "home vector"
        x_home, y_home = gang_info[self.number].coords
        bias_home_x = x_home - self.pos[0]
        bias_home_y = y_home - self.pos[1]
        norm_home = math.sqrt(bias_home_x * bias_home_x +
                              bias_home_y * bias_home_y)

        # Determine bias toawards home
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
        """
        bias_rivals_x, bias_rivals_y = 0.0, 0.0
        contact_all_rivals = self.model.rivalry_matrix[self.number, :].sum()
        for gang in gang_info.values():
            contact_rival = self.model.rivalry_matrix[self.number,
                                                      gang.number]
            rival_x = gang.coords[0] - self.pos[0]
            rival_y = gang.coords[1] - self.pos[1]
            norm_rival = math.sqrt(rival_x * rival_x + rival_y * rival_y)

            if contact_all_rivals and norm_rival:
                weight_bias = (-(contact_rival / contact_all_rivals)
                               / norm_rival)
                bias_rivals_x += weight_bias * rival_x / norm_rival
                bias_rivals_y += weight_bias * rival_y / norm_rival

        return bias_rivals_x, bias_rivals_y


class GangMember_SBLN(SBLN):
    def __init__(self, unique_id, model, pos, gang,
                 min_jump, weight_home, bounded_pareto, kappa, vision, beta):
        super().__init__(unique_id, model, pos, gang,
                         min_jump, weight_home, bounded_pareto, kappa, beta)
        self.interactions = 0
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
                self.interactions += 1
                self.model.update_rivalry(self, agent)
