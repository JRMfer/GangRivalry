# add current structure to path
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(directory, "code"))
sys.path.append(os.path.join(directory, "data"))
sys.path.append(os.path.join(directory, "code", "classes"))
sys.path.append(os.path.join(directory, "code", "helpers"))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import GangRivalry
from helpers.helpers import *

AREAS = "area_no_boundaries.jpg"
ROAD_TXT = "hollenbeckRoadDensity.txt"
GANG_INFO = "gang_information_correct.csv"
OBSERVED_NETWORK = "Connectivity_matrix_observed_network.xlsx"
COLORS = "colors.txt"
REGIONS = "num_bords.csv"

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

if __name__ == "__main__":
    pass
#     road_dens = helpers.load_road_density(ROAD_TXT)
#     road_dens = road_dens[::-1]
#     areas = helpers.load_areas(AREAS, BOUNDS)
#     height, width = road_dens.shape[0] - 1, road_dens.shape[1] - 1
#     gangs = helpers.load_gangs(GANG_INFO)
#     boundaries = helpers.load_region_matrix(REGIONS)
#     observed_graph, all_gr = helpers.load_connectivity_matrix(
#                                 OBSERVED_NETWORK, gangs
#                                 )
#     model = GangRivalry(
#         observed_graph, all_gr, 
#         boundaries, road_density=road_dens,
#         xmax=width, ymax=height,  
#         areas=areas, gang_info = gangs
#         )

#     # start = time.time()
#     # model.run_model(step_count=1000)
#     # print("Model ran in {} seconds".format(time.time() - start))

#     # data = model.datacollector.get_model_vars_dataframe()
#     # data.plot()
#     # plt.show()

#     fixed_params = {
#         "observed_graph": observed_graph,
#         "all_graph": all_gr,
#         "boundaries": boundaries,
#         "xmax": width,
#         "ymax": height,
#         "road_density": road_dens,
#         "areas": areas,
#         "min_jump": 0.1,
#         "weight_home": 1,
#         "bounded_pareto": 1.1,
#         "kappa": 3.5,
#         "vision": 3,
#         "beta": 0.2,
#         "threshold": 0.04
#     }

#     # variable_params = {
#     #     "bounded_pareto": np.arange(1, 2, 0.1),
#     #     "kappa": np.arange(1.5, 4, 0.5)
#     # }
#     # variable_params = {}

#     iterations = 5
#     batch_run = FixedBatchRunner(
#         GangRivalry,
#         fixed_parameters=fixed_params,
#         iterations=5,
#         max_steps=1000,
#         model_reporters={"Interaction": helpers.get_total_interactions, 
#             "Accuracy": helpers.accuracy_graph, 
#             "Shape": helpers.shape_metrics
#             # "Rivalry": helpers.get_rivalry_matrix
#             }
#     )

#     try:
#         start = time.time()
#         batch_run.run_all()
#         print("BATCH RUN ran in {} minutes".format((time.time() - start) / 60))
#         run_data = batch_run.get_model_vars_dataframe()
#         folder = "../results_simulation/"
#         os.makedirs(folder, exist_ok=True)
#         batch_run.to_csv(folder + "SBLN_{}iterations_batchrun0.csv")
#     except KeyboardInterrupt:
#         folder = "../results_simulation/"
#         os.makedirs(folder, exist_ok=True)
#         batch_run.to_csv(folder + "SBLN_{}iterations_batchrun0.csv")

#     print(run_data.head())
#     # plt.scatter(run_data.bounded_pareto, run_data.Interaction)
#     # plt.show()
#     # plt.scatter(run_data.kappa, run_data.Interaction)
#     # plt.show()
