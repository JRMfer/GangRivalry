# import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames, is_correct_integer
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
    
    algorithms = ["BM", "SBLN", "GRAV"]
    if algorithm not in algorithms:
        sys.exit("Algorithm not found. \nChoices: BM/SBLN/GRAV")

    if not is_correct_integer(simulations, 0, 1000):
        sys.exit("The amount of simulations should be an integer " \
                    "between 0 and 1000")
    
    if not is_correct_integer(iterations, 0, 1e7):
        sys.exit("The amount of iterations should be an integer " \
                    "between 0 and 10 million")

    simulations, iterations = int(simulations), int(iterations)

    if not is_correct_integer(iter_check, 0, iterations):
        sys.exit("Iterations check should be positive but smaller " \
                    "than max amount of iterations")
    
    if not os.path.exists(input_data):
        sys.exit("Could not find input data (txt file)")

    if not is_correct_integer(num):
        sys.exit("Start id number should be an integer.")

    iter_check, num = int(iter_check), int(num)
    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")
    
    results_folder = os.path.join(directory, f"results_{user_name}")
    results_algorithms = os.path.join(results_folder, algorithm)
    os.makedirs(results_algorithms, exist_ok=True)

    filenames = get_filenames(data_directory, input_data)
    config = Configuration(filenames)

    # # run simulations for given walking method
    # for sim in tqdm(range(simulations)):
    #     model = GangRivalry(config, algorithm, iter_check)
    #     rivalry_mat = model.run_model(step_count=iterations)
    #     np.save(os.path.join(results_algorithms, 
    #                 f"rivalry_matrix_sim{num + sim}"), rivalry_mat)
    #     data = model.datacollector.get_model_vars_dataframe()
    #     data.to_csv(os.path.join(results_algorithms,
    #                              f"datacollector_sim{num + sim}.csv"))
    plot_metrics(algorithm, 23, user_name)
    plot_network(config.road_dens, config.observed_gr, user_name, 
                    "observed_network")
    plot_network(config.road_dens, config.gtg_gr, user_name, "GTG")
    plot_networks(algorithm, 23, config, user_name, config.parameters["threshold"])
