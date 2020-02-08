import os
import sys
from code.classes.configuration import Configuration
from code.helpers.helpers import get_filenames, plot_networks

if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Usage: python trial_script.py input_data.txt")
    
    _, input_data = sys.argv
    directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = os.path.join(directory, "data")
    filenames = get_filenames(data_directory, input_data)
    config = Configuration(filenames)
    plot_networks("GRAV", 1, config, "JULIEN", config.parameters["threshold"])
