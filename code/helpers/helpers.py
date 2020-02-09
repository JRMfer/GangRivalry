#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper function for a specifig Agent-based model 
to measure gang rivalry in Hollenbeck LA. It contains functionallity 
to ask for user input, read filenames from input file and contains functions
to keep track of certain varables during the simulations (Accuracy and Shape 
metrics.)
"""

# Import built-in modules
import os
import sys
from math import sqrt, inf

# Import libs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

def is_correct_integer(s, lower_bound=-inf, upper_bound=inf):
    """
    Check if argument is an integer and between certaiton (optional) bounds.

    Input:
        s = string to be checked
        lower_bound = min value of integer (if not give is negativ infinity)
        upper_bounds = max value of integer (if not given is infinity)

    Output:
        True if satisfies condition, otherwise False
    """

    # Checks if argument is an integer
    try:
        int(s)
    except ValueError:
        return False

    # Checks if integer is within certain bounds
    temp = int(s)
    return temp > lower_bound and temp <= upper_bound

def check_cmd_args(
    algorithm, simulations, iterations, iter_check, input_data, num
    ):
    """
    Checks if arguemetns are valid. If not notify user and exit program.

    Input:
        algorithm = algorithm for human mobility
        simulations = the amount of simulations to run for each model
        iterations = the amount of iterations to run each simulation
        iter_check = the number of iterations to collect echt time data
        input_data = text file containing the files that need to be loaded
        num = start id number for the files that are saved after each simulation

    Output:
        Nothing.
    """

    # Ensures algorithm (walking method )is valid. 
    algorithms = ["BM", "SBLN", "GRAV"]
    if algorithm not in algorithms:
        sys.exit("Algorithm not found. \nChoices: BM/SBLN/GRAV")

    # Ensures amount of simulations is valid.
    if not is_correct_integer(simulations, 0, 1000):
        sys.exit("The amount of simulations should be an integer "
                 "between 0 and 1000")

    # Ensures amount of iteration per simulation is valid.
    if not is_correct_integer(iterations, 0, 1e7):
        sys.exit("The amount of iterations should be an integer "
                 "between 0 and 10 million")

    simulations, iterations = int(simulations), int(iterations)

    # Ensure data collection moment is valid.
    if not is_correct_integer(iter_check, 0, iterations):
        sys.exit("Iterations check should be positive but smaller "
                 "than max amount of iterations")

    # Ensures input data is a file thac can be found.
    if not os.path.exists(input_data):
        sys.exit("Could not find input data (txt file)")

    # Ensures start id number is an integer
    if not is_correct_integer(num):
        sys.exit("Start id number should be an integer.")

def get_filenames(data_directory, filename):
    """
    Loads filenames from text file.

    Input:
        data_directory = relative file path to data folder
        filename = text file containing all the files that one needs to load

    Output:
        A dictionary with a key pointing to the relative file path 
        of the data files needed to run the model.
    """
    filenames = {}
    with open(filename, 'r') as f:
        line = f.readline().rstrip("\n")
        while line != '':
            line_split = line.split('=')
            filenames[line_split[0]] = os.path.join(
                                        data_directory, line_split[1])
            line = f.readline().rstrip("\n")

    return filenames

def accuracy_graph(model):
    """
    Calculates three different accuracy values of the current graph of the 
    model 1. Accuracy 2. F1 score 3. Matthews Correlation Coefficient (MCC)
    The calculations are mainly based on the current structure of the network
    and determines if the edges are correctly placed (True Postive), are 
    correctly not placed (True Negative), are wrongly placed (True Negative) or 
    are wrongly not placed (False negative) compared to the observed netwrok. 
    The function returns a tuple containing the values of the three statistics.
    For more information one is referred to the article 'Geographical 
    influences of an emerging network of gang rivalries' 
    (Rachel A. Hegemann et al., 2011)

    Input:
        model = Model object
    Output:
        Tuple containing the three accuracy values in the order described above.
    """

    # Set tracking variables to keep score of correctly placed edges 
    # (True Positive) or correclty missing edges (True Negative) and of wrong 
    # placed edges (False Positve) or wrong missing edges (False Negative)
    true_pos, true_neg = 0, 0
    false_pos, false_neg = 0, 0

    # Collect all edges of graph, all possible edges and of the observed network
    graph_edges = set(model.gr.edges)
    all_edges = set(model.config.all_gr.edges)
    observed_graph_edges = set(model.config.observed_gr.edges)

    # Determinse the amount of True Positive and False Postive edges
    non_existing_edges = all_edges ^ observed_graph_edges
    similar_edges = graph_edges & observed_graph_edges
    true_pos = len(similar_edges)
    false_edges = graph_edges & non_existing_edges
    false_pos = len(false_edges)

    # Determene the amount of True Negative and False Negative edges
    obs_remaining = observed_graph_edges ^ similar_edges
    false_neg = len(obs_remaining)
    lacking_edges = non_existing_edges ^ false_edges
    true_neg = len(lacking_edges)

    # Determine the accuracy variables
    ACC = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    F_1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
    try:
        MCC = ((true_pos * true_neg - false_pos * false_neg) / 
            (sqrt((true_pos + false_pos) * (true_pos + false_neg) 
                        * (true_neg + false_pos) * (true_neg + false_neg))))
    except ZeroDivisionError:
        MCC = 0

    # Returns tuple of accuracy values
    return ACC, F_1, MCC

def shape_metrics(model):
    """""
    Calculates three different shape metrics of the current graph of the model.
    Shape metrics: 1. Density 2. Variance of nodal degree 3. Centrality
    The calculations are mainly based on the degree statistics of the current
    graph

    For more information one is referred to the article 'Geographical
    influences of an emerging network of gang rivalries'
    (Rachel A. Hegemann et al., 2011)

    Input:
        model = Model object
    Output:
        Tuple containing the three shape metrics in the order described above.
    """

    # Determine total degree, average degree, max degree and density graph
    degrees = [degree[1] for degree in model.gr.degree]
    total_degree = sum(degrees)
    ave_degree = total_degree / model.config.total_gangs
    max_degree = max(degrees)
    graph_density = nx.density(model.gr)

    # Determine variance of nodal degree and centrality
    variance_degree, centrality = 0, 0
    for degree in degrees:
        variance_degree += ((degree - ave_degree) * (degree - ave_degree))
        centrality += max_degree - degree
    
    # Normailize variance of nodal degree and centrality
    variance_degree /= model.config.total_gangs
    centrality /= ((model.config.total_gangs - 1) * 
                    (model.config.total_gangs - 2))

    # Returns a tuple containging the three statistics
    return graph_density, variance_degree, centrality
