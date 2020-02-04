from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from model import GangRivalry
from helpers import *
import random

GANG_INFO = "../data/gang_information_correct.csv"
OBSERVED_NETWORK = "../data/Connectivity_matrix_observed_network.xlsx"
ROAD_TXT = "../data/hollenbeckRoadDensity.txt"



colors = load_colors("colors.txt")
colors_hex = ["#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2]) 
                for color in colors]

    
# You can change this to whatever ou want. Make sure to make the different types
# of agents distinguishable
def agent_portrayal(agent):

    portrayal = {"Shape": "circle", 
                "Color": colors_hex[agent.number], 
                "Filled": "true", 
                "Layer": 0, 
                "r": 0.8}

    return portrayal

road_dens = np.array(load_road_density(ROAD_TXT))
road_dens = road_dens[::-1]
areas = load_areas(AREAS)
xmax, ymax = road_dens.shape[0], road_dens.shape[1]
gangs = load_gangs(GANG_INFO)
observed_graph, all_gr = load_connectivity_matrix(OBSERVED_NETWORK, gangs)

# Create a grid of 20 by 20 cells, and display it as 500 by 500 pixels
# grid = CanvasGrid(agent_portrayal, 500, 500, 500, 500)
grid = CanvasGrid(agent_portrayal, 100, 100, 500, 500)
chart = ChartModule([{"Label": "Interaction",
                      "Color": "green"}],
                    data_collector_name='datacollector')

# Create the server, and pass the grid and the graph
server = ModularServer(GangRivalry,
                       [grid, chart],
                       "Gang Rivalry Model",
                       {
                        "all_graph": all_gr,
                        "observed_graph": observed_graph,
                        "xmax": xmax,
                        "ymax": ymax,
                        "road_density": road_dens,
                        "areas": areas,
                        "gang_info": gangs
                        })

server.port = 8521