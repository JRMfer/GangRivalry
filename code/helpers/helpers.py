import os
import numpy as np
import pandas as pd
from math import sqrt, inf
from matplotlib import pyplot as plt
import networkx as nx

def is_correct_integer(s, lower_bound=-inf, upper_bound=inf):
    try:
        int(s)
    except ValueError:
        return False

    temp = int(s)
    return temp > lower_bound and temp <= upper_bound

def get_filenames(data_directory, filename):
    """
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


def get_total_interactions(model):
    """
    """
    return sum([
                agent.interactions for agent in model.schedule.agents
                ])

def get_rivalry_matrix(model):
    """
    """
    return model.rivalry_matrix

def accuracy_graph(model):
    """
    """

    true_pos, true_neg = 0, 0
    false_pos, false_neg = 0, 0
    graph_edges = set(model.gr.edges)
    all_edges = set(model.config.all_gr.edges)
    observed_graph_edges = set(model.config.observed_gr.edges)

    non_existing_edges = all_edges ^ observed_graph_edges
    similar_edges = graph_edges & observed_graph_edges
    true_pos = len(similar_edges)
    false_edges = graph_edges & non_existing_edges
    false_pos = len(false_edges)

    obs_remaining = observed_graph_edges ^ similar_edges
    false_neg = len(obs_remaining)
    lacking_edges = non_existing_edges ^ false_edges
    true_neg = len(lacking_edges)
    ACC = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    F_1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
    try:
        MCC = ((true_pos * true_neg - false_pos * false_neg) / 
            (sqrt((true_pos + false_pos) * (true_pos + false_neg) 
                        * (true_neg + false_pos) * (true_neg + false_neg))))
    except ZeroDivisionError:
        MCC = 0

    return ACC, F_1, MCC

def shape_metrics(model):
    """
    """

    degrees = [degree[1] for degree in model.gr.degree]
    total_degree = sum(degrees)
    ave_degree = total_degree / model.config.total_gangs
    graph_density = nx.density(model.gr)
    max_degree = max(degrees)

    variance_degree, centrality = 0, 0
    for degree in degrees:
        variance_degree += ((degree - ave_degree) * (degree - ave_degree))
        centrality += max_degree - degree

    variance_degree /= model.config.total_gangs
    centrality /= ((model.config.total_gangs - 1) * 
                    (model.config.total_gangs - 2))

    return graph_density, variance_degree, centrality
