#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains all the necessary in order to run simulations for a specific
human mobility algorithm. It will guarantee that the user correctly tries to run
this script and will save all results to specific folder.
"""

# Import built-in libraries
import os
import sys

# Import libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from own modules
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames, check_cmd_args
from code.visualization.visualizer import (
    plot_network, plot_networks, plot_metrics
    )

if __name__ == "__main__":

    # Ensures the right amount of cmd arguments are given
    if len(sys.argv) != 8:
        sys.exit("Correct usage: python main.py algorithm " \
                    "simulations iterations iterations_check " \
                    "input_data.txt " \
                    "user_name start_id_number")

    # Determines if arguments are valid.
    _, algorithm, simulations, iterations, iter_check, input_data, user_name, num = sys.argv
    check_cmd_args(algorithm, simulations, iterations, iter_check, input_data, num)
    simulations, iterations = int(simulations), int(iterations)
    iter_check, num = int(iter_check), int(num)

    # Create relative file paths and a folder for saving results
    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")
    results_folder = os.path.join(directory, f"results_{user_name}")
    results_algorithms = os.path.join(results_folder, algorithm)
    os.makedirs(results_algorithms, exist_ok=True)

    # Get filenames and create a configuration for the algorithm
    filenames = get_filenames(data_directory, input_data)
    config = Configuration(filenames)

    # Run simulations for given walking method and save results of each simulation
    for sim in tqdm(range(simulations)):
        model = GangRivalry(config, algorithm, iter_check)
        rivalry_mat = model.run_model(step_count=iterations)
        np.save(os.path.join(results_algorithms,
                    f"rivalry_matrix_sim{num + sim}"), rivalry_mat)
        data = model.datacollector.get_model_vars_dataframe()
        data.to_csv(os.path.join(results_algorithms,
                                 f"datacollector_sim{num + sim}.csv"))

    # Make figures of the development of the statistics and resulting networks
    plot_metrics(algorithm, simulations, user_name)
    plot_network(config.road_dens, config.observed_gr, user_name,
                    "observed_network")
    plot_network(config.road_dens, config.gtg_gr, user_name, "GTG")
    plot_networks(algorithm, simulations, config, user_name)
