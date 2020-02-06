# add current structure to path
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(DIRECTORY, "data")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Correct usage: python main.py input_data.txt")

    filenames = get_filenames(DATA_DIRECTORY, sys.argv[1])
    config = Configuration(filenames)
    # road_dens = load_road_density(os.path.join(DATA_DIRECTORY, ROAD_TXT))
    # print(road_dens)
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
