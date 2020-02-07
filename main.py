# import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from code.classes.configuration import Configuration
from code.classes.model import GangRivalry
from code.helpers.helpers import get_filenames, is_correct_integer

if __name__ == "__main__":

    if len(sys.argv) != 7:
        sys.exit("Correct usage: python main.py algorithm " \
                    "simulations iterations input_data.txt " \
                    "user_name start_id_number")

    _, algorithm, simulations, iterations, input_data, user_name, num = sys.argv
    
    algorithms = ["BM", "SBLN", "GRAV"]
    if algorithm not in algorithms:
        sys.exit("Algorithm not found. \nChoices: BM/SBLN/GRAV")

    if not is_correct_integer(simulations, 0, 1000):
        sys.exit("The amount of simulations should be an integer " \
                    "between 0 and 1000")
    
    if not is_correct_integer(iterations, 0, 1e7):
        sys.exit("The amount of iterations should be an integer " \
                    "between 0 and 10 million")
    
    if not os.path.exists(input_data):
        sys.exit("Could not find input data (txt file)")

    if not is_correct_integer(num):
        sys.exit("Start id number should be an integer.")

    simulations, iterations = int(simulations), int(iterations)
    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")
    
    results_folder = os.path.join(directory, f"results_{user_name}")
    results_algorithms = os.path.join(results_folder, algorithm)
    os.makedirs(results_algorithms, exist_ok=True)

    filenames = get_filenames(data_directory, input_data)
    config = Configuration(filenames)

    # run simulations for given walking method
    for sim in tqdm(range(simulations)):
        model = GangRivalry(config, algorithm)
        rivalry_mat = model.run_model(step_count=iterations)
        np.save(os.path.join(results_algorithms, 
                    f"rivalry_matrix_sim{num + sim}"), rivalry_mat)
        data = model.datacollector.get_model_vars_dataframe()
        data.to_csv(os.path.join(results_algorithms,
                                 f"datacollector_sim{num + sim}.csv"))
