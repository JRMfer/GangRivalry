import cv2 as cv
import numpy as np
import pandas as pd
from math import sqrt
import os
from matplotlib import pyplot as plt
import networkx as nx
import csv
from agents import Gang

COLORS = "colors.txt"

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

def load_gangs(filename):
    """
    Loads gangs from a given csv file.

    Input: 
        filename = location of csv file
    Output: 
        dictionary with keys an integer pointing to 
        a Gang object
    """
    gangs = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gang = Gang(int(row["gang number"]), 
                        (float(row["x location"]), float(row["y"])), 
                        int(row["gang members"]))
            gangs[int(row["gang number"])] = gang

    return gangs

def load_road_density(filename):
    """
    Loads the road density from a txt file into 
    a 2D numpy array
    """
    road_dens = []
    with open(filename, 'r') as f:

        line = f.readline().rstrip(" \n")
        while line != '':
            split_line = line.split("    ")
            line_float = [float(x) for x in split_line]
            road_dens.append(line_float)
            line = f.readline().rstrip(" \n")

    return np.array(road_dens)

def load_areas(filename, bounds):
    """
    Determines which coordinates belong 
    to the differen regions of Hollenbeck.
    """
    areas = cv.imread(filename)

    # loop over the boundaries
    areadict = {}
    for l, (lower, upper) in enumerate(bounds):

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv.inRange(areas, lower, upper)
        for n, i in enumerate(mask):
            
            # find region for sequence of coordinates
            for m, j in enumerate(i):
                if j == 255:
                    areadict[(m, 690 - n)] = l

        cv.waitKey(0)
        cv.destroyAllWindows()

    width, height = 1036, 691
    x_coords = set(range(width))
    y_coords = set(range(height))
    all_coords = {(x, y) for x in x_coords for y in y_coords}
    missing_coords = all_coords ^ set(areadict.keys())
    # fig = plt.figure()
    # x = [i[0] for i in missing_coords]
    # y = [i[1] for i in missing_coords]
    # plt.scatter(x, y)
    # plt.show()
    for coord in missing_coords:
        areadict[coord] = 24
        
    return areadict

def load_region_matrix(filename):
    """
    Loads 
    """

    regions = []
    with open(filename, 'r') as csvfile:
        
        reader = csv.reader(csvfile)
        for row in reader:
            regions.append([int(float(x)) for x in row])

    return np.array(regions)

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

def load_connectivity_matrix(filename, gangs):
    """
    """

    matrix = pd.read_excel(filename)
    matrix = matrix.values
    total_matrix = np.ones((matrix.shape[0], matrix.shape[1]))
    rows, cols = np.where(matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()

    # Add nodes to graph
    for k, gang in gangs.items():
        gr.add_node(k, pos=gang.coords)

    gr.add_edges_from(edges)
    pos = nx.get_node_attributes(gr, "pos")
    # nx.draw(gr, pos)
    # plt.show()

    # # Also generate a graph with all possible edges for now?
    rows, cols = np.where(total_matrix == 1)
    all_edges = zip(rows.tolist(), cols.tolist())
    all_gr = nx.Graph()
    for k, gang in gangs.items():
        all_gr.add_node(k, pos=gang.coords)
    all_gr.add_edges_from(all_edges)
    all_pos = nx.get_node_attributes(all_gr, "pos")
    # nx.draw(all_gr, all_pos)
    # plt.show()

    return gr, all_gr

def accuracy_graph(model):
    """
    """

    true_pos, true_neg = 0, 0
    false_pos, false_neg = 0, 0
    graph_edges = set(model.gr.edges)
    all_edges = set(model.all_graph.edges)
    observed_graph_edges = set(model.observed_graph.edges)

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


def load_colors(filename):
    """
    Loads colors from textfile to indicate 
    the different gangs in the visualization
    """

    colors = []
    with open(filename, 'r') as f:
        line = f.readline().rstrip("\n")
        while line != '':
            split = line.split(',')
            rgb_color = (int(split[0]), int(split[1]), int(split[2]))
            colors.append(rgb_color)
            line = f.readline().rstrip("\n")

    return colors

def plot_simulation(model):
    """
    """

    fig = plt.figure()
    plt.xlim(0, model.width)
    plt.ylim(0, model.height)
    for agent in model.schedule.agents:
        x, y = agent.pos
        plt.scatter(x, y, color=model.colors[agent.number], s=2)

    folder = "../results/"
    os.makedirs(folder, exist_ok=True)
    fig.savefig(folder + "step" + str(model.schedule.time) + ".png")
    plt.close()

def shape_metrics(model):
    """
    """

    degrees = [degree[1] for degree in model.gr.degree]
    total_degree = sum(degrees)
    ave_degree = total_degree / model.total_gangs
    graph_density = total_degree / (model.total_gangs * (model.total_gangs - 1))
    max_degree = max(degrees)

    variance_degree, centrality = 0, 0
    for degree in degrees:
        variance_degree += ((degree - ave_degree) * (degree - ave_degree))
        centrality += max_degree - degree

    variance_degree /= model.total_gangs
    centrality /= ((model.total_gangs - 1) * (model.total_gangs - 2))

    return graph_density, variance_degree, centrality
