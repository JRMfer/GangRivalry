import os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from code.classes.agents import Gang

def random_gangs(n, width, height):
    """
    Generates Gangs of fixed size 30 randomly located 
    inside the area.

    Input: 
        n = amoount of gangs, 
        width = width of area,
        height = height of area
    output:
        dictionary with keys an integer pointing to 
        a Gang object
    """

    gangs, size = {}, 30
    for i in range(n):
        coords = (np.random.randint(0, width), np.random.randint(0, height))
        gangs[i] = Gang(i, coords, size)
    return gangs

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

def plot_simulation(model):
    """
    """

    fig = plt.figure()
    plt.xlim(0, model.width)
    plt.ylim(0, model.height)
    for agent in model.schedule.agents:
        x, y = agent.pos
        plt.scatter(x, y, color=model.config.colors[agent.number], s=2)

    folder = "../results/"
    os.makedirs(folder, exist_ok=True)
    fig.savefig(folder + "step" + str(model.schedule.time) + ".png")
    plt.close()

def shape_metrics(model):
    """
    """

    degrees = [degree[1] for degree in model.gr.degree]
    total_degree = sum(degrees)
    ave_degree = total_degree / model.config.total_gangs
    graph_density = total_degree / (model.config.total_gangs * 
                                        (model.config.total_gangs - 1))
    max_degree = max(degrees)

    variance_degree, centrality = 0, 0
    for degree in degrees:
        variance_degree += ((degree - ave_degree) * (degree - ave_degree))
        centrality += max_degree - degree

    variance_degree /= model.config.total_gangs
    centrality /= ((model.config.total_gangs - 1) * 
                    (model.config.total_gangs - 2))

    return graph_density, variance_degree, centrality
