from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from .agents import Gang, GangMemberB, GangMember_SBLN, GangMemberG
from .schedule import OneRandomActivation
from code.helpers.helpers import accuracy_graph, shape_metrics

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

class GangRivalry(Model):
    """
    """

    def __init__(self, config, algorithm):
        super().__init__()

        self.config = config
        self.width = self.config.road_dens.shape[1] - 1
        self.height = self.config.road_dens.shape[0] - 1
        self.min_jump = self.config.parameters["min_jump"]
        self.lower_max_jump = self.config.parameters["lower_max_jump"]
        self.upper_max_jump = self.config.parameters["upper_max_jump"]
        self.weight_home = self.config.parameters["weight_home"]
        self.bounded_pareto = self.config.parameters["bounded_pareto"]
        self.kappa = self.config.parameters["kappa"]
        self.beta = self.config.parameters["beta"]
        self.vision = self.config.parameters["vision"]
        self.area = ContinuousSpace(self.width, self.height, False)
        self.threshold = self.config.parameters["threshold"]
        self.algorithm = algorithm

        gang_sizes = [gang.size for gang in self.config.gang_info.values()]
        self.min_gang = min(gang_sizes)
        self.max_gang = max(gang_sizes)
        self.norm_gang_size = [
            (i - self.min_gang) / (self.max_gang - self.min_gang) 
            for i in gang_sizes
            ]

        self.schedule = OneRandomActivation(self)
        self.init_population()

        self.rivalry_matrix = np.zeros(
                (self.config.total_gangs, 
                self.config.total_gangs)
                )
        self.create_graph()
        self.datacollector = DataCollector(
            model_reporters={
                "Accuracy": accuracy_graph,
                "Shape": shape_metrics},

            agent_reporters={"Agent interaction": "interactions"}
        )
        self.running = True
        self.datacollector.collect(self)

    def init_population(self):
        """
        """
        for gang in self.config.gang_info.values():
            for _ in range(gang.size):
                self.new_agent(gang.coords, gang.number)

    def new_agent(self, pos, name):
        """
        """
        agent = None

        if self.algorithm == "BM":
            agent = GangMemberB(self.next_id(), self, pos, 
                                name, self.beta, self.vision)

        elif self.algorithm == "SBLN":
            agent = GangMember_SBLN(self.next_id(), self, pos, name, 
                                    self.min_jump, self.weight_home, 
                                    self.bounded_pareto, self.kappa, 
                                    self.vision, self.beta)
        
        elif self.algorithm == "GRAV":
            agent = GangMemberG(self.next_id(), self, pos, name, 
                                self.min_jump, self.weight_home, 
                                self.bounded_pareto, self.kappa, 
                                self.vision, self.beta)

        self.area.place_agent(agent, pos)
        self.schedule.add(agent)

    def update_rivalry(self, agent1, agent2):
        """
        """
        self.rivalry_matrix[agent1.number, agent2.number] += 1
        self.rivalry_matrix[agent2.number, agent1.number] += 1

    def create_graph(self):
        gr = nx.Graph()
        for gang in self.config.gang_info.values():
            gr.add_node(gang.number, pos=gang.coords)
        self.gr = gr

    def make_graph(self):
        shape = self.config.total_gangs
        edges = set(self.gr.edges).copy()
        self.gr.remove_edges_from(edges)
        rivalry_strength = np.zeros((shape, shape))

        for i in range(shape):
            total_interactions = np.sum(self.rivalry_matrix[i])
            for j in range(shape):
                if total_interactions:
                    rival_strength = self.rivalry_matrix[i][j] / \
                        total_interactions
                    if rival_strength > self.threshold:
                        self.gr.add_edge(i, j)

    def step(self):
        self.schedule.step()

    def run_model(self, step_count=200):
        for i in tqdm(range(step_count)):     
            self.step()

            if i % 1000 == 0:
                self.make_graph()
                self.datacollector.collect(self)

        return self.rivalry_matrix
