# import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames, check_cmd_args
from code.visualization.visualizer import (
    plot_network, plot_networks, plot_metrics
    )

if __name__ == "__main__":

    if len(sys.argv) != 8:
        sys.exit("Correct usage: python main.py algorithm " \
                    "simulations iterations iterations_check " \
                    "input_data.txt " \
                    "user_name start_id_number")

    _, algorithm, simulations, iterations, iter_check, input_data, user_name, num = sys.argv

    check_cmd_args(algorithm, simulations, iterations, iter_check, input_data, num)

    simulations, iterations = int(simulations), int(iterations)
    iter_check, num = int(iter_check), int(num)
    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")
    
    results_folder = os.path.join(directory, f"results_{user_name}")
    results_algorithms = os.path.join(results_folder, algorithm)
    os.makedirs(results_algorithms, exist_ok=True)

    filenames = get_filenames(data_directory, input_data)
    config = Configuration(filenames)

    # run simulations for given walking method
    for sim in tqdm(range(simulations)):
        model = GangRivalry(config, algorithm, iter_check)
        rivalry_mat = model.run_model(step_count=iterations)
        np.save(os.path.join(results_algorithms, 
                    f"rivalry_matrix_sim{num + sim}"), rivalry_mat)
        data = model.datacollector.get_model_vars_dataframe()
        data.to_csv(os.path.join(results_algorithms,
                                 f"datacollector_sim{num + sim}.csv"))

    plot_metrics(algorithm, simulations, user_name)
    plot_network(config.road_dens, config.observed_gr, user_name, 
                    "observed_network")
    plot_network(config.road_dens, config.gtg_gr, user_name, "GTG")
    plot_networks(algorithm, simulations, config, user_name)
