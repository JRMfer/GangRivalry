from mesa import Model
from mesa.space import ContinuousSpace
from schedule import OneRandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import numpy as np
from agents import Gang, GangMember
from helpers.helpers import load_colors
import time
import networkx as nx
from tqdm import tqdm

AREAS = "../images/area_no_boundaries.jpg"
ROAD_TXT = "../data/hollenbeckRoadDensity.txt"
GANG_INFO = "../data/gang_information_correct.csv"
OBSERVED_NETWORK = "../data/Connectivity_matrix_observed_network.xlsx"
COLORS = "colors.txt"
REGIONS = "../data/num_bords.csv"

BOUNDS = [
    ([60, 50, 120], [70, 60, 130]),  # 15, 1
    ([0, 100, 0], [10, 115, 10]),  # 20, 2
    ([0, 0, 170], [10, 5, 185]),  # 21, 3
    ([120, 70, 0], [135, 80, 10]),  # 11, 4
    ([0, 105, 120], [5, 115, 135]),  # 22, 5
    ([0, 45, 120], [5, 55, 135]),  # 14, 6
    ([230, 230, 230], [255, 255, 255]),  # 23, 7
    ([50, 60, 240], [60, 70, 255]),  # 2, 8
    ([240, 0, 170], [255, 10, 185]),  # 24, 9
    ([200, 0, 240], [235, 20, 255]),  # 4, 10
    ([70, 235, 120], [90, 255, 150]),  # 3, 11
    ([0, 100, 250], [10, 110, 255]),  # 16, 12
    ([140, 0, 160], [155, 5, 175]),  # 13, 13
    ([240, 30, 0], [255, 45, 10]),  # 12, 14
    ([0, 200, 240], [10, 240, 255]),  # 6, 15
    ([230, 230, 0], [255, 255, 20]),  # 5, 16
    ([0, 0, 110], [10, 10, 130]),  # 7, 17
    ([220, 170, 245], [255, 190, 255]),  # 9, 18
    ([80, 100, 110], [90, 120, 120]),  # 19, 19
    ([60, 50, 0], [70, 60, 10]),  # 1, 20
    ([35, 110, 20], [65, 140, 50]),  # 10, 21
    ([105, 0, 235], [120, 10, 255]),  # 8, 22
    ([0, 240, 60], [5, 255, 80]),  # 17, 23
    ([0, 240, 175], [5, 255, 190]),  # 18, 24
    ([0, 0, 0], [20, 20, 20])  # bords
]

class GangRivalry(Model):
    """
    """

    def __init__(
            self, observed_graph, all_graph, 
            boundaries, road_density, areas, 
            xmax=100, ymax=100, min_jump=0.1, 
            lower_max_jump=100, upper_max_jump=200, vision=3,
            weight_home=1, bounded_pareto=1.1, beta=0.2, 
            kappa=3.5, gang_info={}, threshold=0.04
            ):
        super().__init__()

        self.width = xmax
        self.height = ymax
        self.min_jump = min_jump
        self.lower_max_jump = lower_max_jump
        self.upper_max_jump = upper_max_jump
        self.weight_home = weight_home
        self.bounded_pareto = bounded_pareto
        self.kappa = kappa
        self.gang_info = gang_info
        self.total_gangs = len(self.gang_info)
        self.boundaries = boundaries
        self.beta = beta
        self.vision = vision
        self.all_graph = all_graph
        self.observed_graph = observed_graph
        self.area = ContinuousSpace(self.width, self.height, False)

        self.schedule = OneRandomActivation(self)
        colors = load_colors(COLORS)
        self.colors = ["#{:02x}{:02x}{:02x}"
                    .format(color[0], color[1], color[2]) for color in colors]
        self.init_population()

        self.rivalry_matrix = np.zeros((self.total_gangs, self.total_gangs))
        self.road_density = road_density
        self.areas = areas
        self.threshold = threshold
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
        for gang in self.gang_info.values():
            for _ in range(gang.size):
                self.new_agent(gang.coords, gang.number)

    def new_agent(self, pos, name):
        """
        """
        x, y = pos
        agent = GangMember(self.next_id(), self, pos, name, self.min_jump,
                           self.weight_home, self.bounded_pareto,
                           self.kappa, self.vision, self.beta)

        self.area.place_agent(agent, pos)
        # self.grid.place_agent(agent, (int(x), int(y)))
        self.schedule.add(agent)

    def update_rivalry(self, agent1, agent2):
        """
        """
        self.rivalry_matrix[agent1.number, agent2.number] += 1
        self.rivalry_matrix[agent2.number, agent1.number] += 1

    def create_graph(self):
        gr = nx.Graph()

        for gang in self.gang_info.values():
            gr.add_node(gang.number, pos=gang.coords)
        self.gr = gr

    def make_graph(self):
        shape = self.total_gangs
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
