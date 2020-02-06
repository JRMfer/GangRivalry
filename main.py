# import modules
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames

if __name__ == "__main__":
    algorithms = ["BM", "SBLN", "gravity"]
    if len(sys.argv) != 3:
        sys.exit("Correct usage: python main.py algorithm input_data.txt")

    algorithm = sys.argv[1]
    if algorithm not in algorithms:
        sys.exit("Algorithm not found. \nChoices: BM/SBLN/gravity")

    simulations = 0
    while simulations <= 0 or simulations > 1000:
        try:
            simulations = int(input("Enter the amount of simulations: "))
        except ValueError:
            print("Make sure to enter an integer between 0 and 1000")

    iterations = 0
    while iterations <= 0 or iterations > 1e7:
        try:
            iterations = int(input("Enter the amount of iterations per " \
                                        "simulation: "))
        except ValueError:
            print("Make sure to enter an integer between 0 and 10 million")

    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")

    results_folder = os.path.join(directory, "results_JULIEN")
    results_algorithms = os.path.join(results_folder, algorithm)
    os.makedirs(results_algorithms, exist_ok=True)

    filenames = get_filenames(data_directory, sys.argv[2])
    config = Configuration(filenames)
    model = GangRivalry(config, algorithm)

    # run simulations for given walking method
    for sim in range(simulations):
        rivalry_mat = model.run_model(step_count=iterations)
        np.save(os.path.join(results_algorithms, f"rivalry_matrix_sim{sim}"), 
                    rivalry_mat)
