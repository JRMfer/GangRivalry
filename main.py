# import modules
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames, is_correct_integer

if __name__ == "__main__":

    if len(sys.argv) != 6:
        sys.exit("Correct usage: python main.py algorithm " \
                    "simulations iterations input_data.txt " \
                    "user_name")

    _, algorithm, simulations, iterations, input_data, user_name = sys.argv

    algorithms = ["BM", "SBLN", "gravity"]
    if algorithm not in algorithms:
        sys.exit("Algorithm not found. \nChoices: BM/SBLN/gravity")

    if not is_correct_integer(simulations, 0, 1000):
        sys.exit("The amount of simulations should be an integer " \
                    "between 0 and 1000")
    
    if not is_correct_integer(iterations, 0, 1e7):
        sys.exit("The amount of iterations should be an integer " \
                    "between 0 and 10 million")
    
    if not os.path.exists(input_data):
        sys.exit("Could not find input data (txt file)")

    simulations, iterations = int(simulations), int(iterations)
    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")
    
    results_folder = os.path.join(directory, f"results_{user_name}")
    results_algorithms = os.path.join(results_folder, algorithm)
    os.makedirs(results_algorithms, exist_ok=True)

    filenames = get_filenames(data_directory, input_data)
    config = Configuration(filenames)
    model = GangRivalry(config, algorithm)

    # run simulations for given walking method
    for sim in range(simulations):
        rivalry_mat = model.run_model(step_count=iterations)
        np.save(os.path.join(results_algorithms, f"rivalry_matrix_sim{sim}"), 
                    rivalry_mat)
        data = model.datacollector.get_model_vars_dataframe()
        data.to_csv(os.path.join(results_algorithms,
                                 f"datacollector_sim{sim}.csv"))
