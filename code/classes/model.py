from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from .agents import Gang, GangMember_SBLN
from .schedule import OneRandomActivation
from code.helpers.helpers import (get_total_interactions, accuracy_graph,
                                  shape_metrics, get_rivalry_matrix)

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

        self.schedule = OneRandomActivation(self)
        self.init_population()

        self.rivalry_matrix = np.zeros(
                (self.config.total_gangs, 
                self.config.total_gangs)
                )
        self.create_graph()
        self.datacollector = DataCollector(
            model_reporters={
                "Interaction": get_total_interactions,
                "Accuracy": accuracy_graph,
                "Shape": shape_metrics,
                "Rivalry": get_rivalry_matrix},

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

        if self.algorithm == "SBLN":
            x, y = pos
            agent = GangMember_SBLN(self.next_id(), self, pos, name, 
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

# if __name__ == "__main__":
#     road_dens = load_road_density(ROAD_TXT)
#     road_dens = road_dens[::-1]
#     areas = load_areas(AREAS, BOUNDS)
#     height, width = road_dens.shape[0] - 1, road_dens.shape[1] - 1
#     gangs = load_gangs(GANG_INFO)
#     boundaries = load_region_matrix(REGIONS)
#     observed_graph, all_gr = load_connectivity_matrix(OBSERVED_NETWORK, gangs)
#     model = GangRivalry(observed_graph, all_gr, boundaries, road_dens, areas, 
#                         xmax=width, ymax=height,gang_info=gangs)


#     folder = "simulations_SBLN/"
#     os.makedirs(folder, exist_ok=True)
#     model.run_model(step_count=2000000)
#     data = model.datacollector.get_model_vars_dataframe()
#     data.to_csv(folder + "run6.csv")
#     print(data)
#     # start = time.time()
#     # model.run_model(step_count=10000)
#     # print("Model ran in {} seconds".format(time.time() - start))

#     # data = model.datacollector.get_model_vars_dataframe()
#     # print(data.head())
#     # data.plot()
#     # plt.show()
