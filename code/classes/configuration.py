#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains all the functionallity needed
to load all input files in order to run the simulations.
"""

# Import built-in libraries
import re
import csv

# Import libraries
import cv2 as cv
import numpy as np
import pandas as pd
import networkx as nx

# Import from own module (Gang)
from .agents import Gang

class Configuration(object):
    """
    An object which contains all methods needed to load all files given by the user.
    Input:
        dictionary with as keys [GANG_INFO, ROAD_TXT, BOUNDS, AREAS, REGIONS,
        OBSERVED_NETWORK, GTG_NETWORK, COLORS, OPT_PARAMETERS] and each one of
        them pointing to the corresponding files.
    Output:
        An Configuration object with as attributes the parameters needed to run
        the simulation
    """

    def __init__(self, filenames):
        """
        Initialize each configuration with information about the gangs,
        road density of the area, a dictionary with as key a coordinate in the
        area and as value the region it is in, a matrix to represent the
        crossing probability from region i to j, the observed graph, a graph
        containing all the edges, geographical threshold graph,
        colors for the different gangs and the parameters for the human mobility
        algorithms.
        """

        # Load gang information
        self.gang_info = {}
        self.load_gangs(filenames["GANG_INFO"])
        self.total_gangs = len(self.gang_info)

        # Load road denisty
        self.road_dens = []
        self.load_road_density(filenames["ROAD_TXT"])

        # Load color scheme representing the bounds of regions and create
        # dictionary with as key a coordinate and value the region it is in.
        self.bounds = []
        self.load_bounds(filenames["BOUNDS"])
        self.areas = {}
        self.load_areas(filenames["AREAS"], self.bounds)

        # Load matrix for crossing probability regions
        self.boundaries = []
        self.load_region_matrix(filenames["REGIONS"])

        # Create observed graph and graph containing all edges
        self.observed_gr = nx.Graph()
        self.all_gr = nx.Graph()
        self.load_connectivity_matrix(
                    filenames["OBSERVED_NETWORK"],
                    self.gang_info
                    )

        # Create geographical threshold graph
        self.gtg_gr = nx.Graph()
        self.load_gtg_matrix(filenames["GTG_NETWORK"], self.gang_info)

        # Load colorscheme from text file
        self.colors = []
        self.load_colors(filenames["COLORS"])
        self.colors = ["#{:02x}{:02x}{:02x}"
                        .format(color[0], color[1], color[2])
                        for color in self.colors]

        # Load parameters from text file
        self.parameters = {}
        self.load_parameters(filenames["OPT_PARAMETERS"])

    def load_gangs(self, filename):
        """
        Loads gangs from a given csv file.
        Input:
            filename = location of csv file
        Output:
            dictionary with keys an integer pointing to
            a Gang object
        """

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gang = Gang(int(row["gang number"]),
                            (float(row["x location"]), float(row["y"])),
                            int(row["gang members"]))
                self.gang_info[int(row["gang number"])] = gang

    def load_road_density(self, filename):
        """
        Loads the road density.
        Input:
            filename = location of text file
        Output:
            2D numpy array with as each element the road density of the area
        """

        with open(filename, 'r') as f:
            line = f.readline().rstrip(" \n")
            while line != '':
                split_line = line.split("    ")
                line_float = [float(x) for x in split_line]
                self.road_dens.append(line_float)
                line = f.readline().rstrip(" \n")
        self.road_dens = np.array(self.road_dens)

    def load_bounds(self, filename):
        """
        Load colors representing the bounds of the different regions.
        Input:
            filename = location of text file
        Output:
            Changed attribute (self.bounds)
        """

        with open(filename, 'r') as f:

            line = f.readline().rstrip("\n")
            while line != '':
                line_no_whitespace = re.sub(r'\s+', '', line)
                line_split = line_no_whitespace.split('[')
                region_colors = []

                for rgb_triple in line_split:
                    rgb_split = rgb_triple.split(',')
                    region_colors.append([int(rgb) for rgb in rgb_split])
                self.bounds.append(tuple(region_colors))
                line = f.readline().rstrip("\n")

    def load_areas(self, filename, bounds):
        """
        Determines which coordinates belong
        to the different regions of Hollenbeck.
        Input:
            filename = location of JPEG file
            bounds =  text file
        Output:
            Changes dictionary attribute to one with as keys coordinates of the
            area and as values the corresponding regions it is in.
        """

        areas = cv.imread(filename)

        # Loop over the boundaries
        for l, (lower, upper) in enumerate(bounds):

            # Create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # Find the colors within the specified boundaries and apply
            # the mask
            mask = cv.inRange(areas, lower, upper)
            for n, i in enumerate(mask):

                # Find region for sequence of coordinates
                for m, j in enumerate(i):
                    if j == 255:
                        self.areas[(m, 690 - n)] = l

            cv.waitKey(0)
            cv.destroyAllWindows()

        # Create a list with all coordinates in area
        # and determine the missing coordinates
        width, height = 1036, 691
        x_coords = set(range(width))
        y_coords = set(range(height))
        all_coords = {(x, y) for x in x_coords for y in y_coords}
        missing_coords = all_coords ^ set(self.areas.keys())

        # Set missing coordates as the coordinates of a border (water/bridge etc..)
        for coord in missing_coords:
            self.areas[coord] = 24

    def load_region_matrix(self, filename):
        """
        Loads the crossing probablity matrix between different regions.
        Input:
            filename =  csv file
        Output:
            Changes attribute to 2D numpy array containing the min amount of
            boundaries one needs to cross in order to go from region i to j.
        """
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.boundaries.append([int(float(x)) for x in row])
        self.boundaries = np.array(self.boundaries)

    def load_connectivity_matrix(self, filename, gangs):
        """
        Loads and creates the observed network.
        Input:
            filename = location of Excel file
            gangs = list of Gang objects
        Output:
            Creates the observed network and a network containing all edges as
            an attribute of the object
        """

        # Load csv file, make adjency matrix and determine the edges
        matrix = pd.read_excel(filename)
        matrix = matrix.values
        total_matrix = np.ones((matrix.shape[0], matrix.shape[1]))
        rows, cols = np.where(matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())

        # Add nodes to graph
        for k, gang in gangs.items():
            self.observed_gr.add_node(k, pos=gang.coords)
        self.observed_gr.add_edges_from(edges)

        # Also generate a graph with all possible edges for now
        rows, cols = np.where(total_matrix == 1)
        all_edges = zip(rows.tolist(), cols.tolist())

        # Add nodes (gangs) to the graph with all possible edges
        for k, gang in gangs.items():
            self.all_gr.add_node(k, pos=gang.coords)
        self.all_gr.add_edges_from(all_edges)

    def load_gtg_matrix(self, filename, gangs):
        """
        Load the Geographical Threshold Graph.
        Input:
            filename = location of Excel file
            gangs = list of Gang objects
        Output:
            Creates the geographical threshold graph as an attribute of the object
        """
        matrix = pd.read_excel(filename)
        matrix = matrix.values
        total_matrix = np.ones((matrix.shape[0], matrix.shape[1]))
        rows, cols = np.where(matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())

        # Add nodes to graph
        for k, gang in gangs.items():
            self.gtg_gr.add_node(k, pos=gang.coords)
        self.gtg_gr.add_edges_from(edges)

    def load_colors(self, filename):
        """
        Loads colors from textfile to indicate
        the different gangs in the visualization
        Input:
            filename = location of text files
        Output:
            Creates a list the colors as an attribute of this object
        """

        with open(filename, 'r') as f:
            line = f.readline().rstrip("\n")
            while line != '':
                split = line.split(',')
                rgb_color = (int(split[0]), int(split[1]), int(split[2]))
                self.colors.append(rgb_color)
                line = f.readline().rstrip("\n")

    def load_parameters(self, filename):
        """
        Load parameters for the model.
        Input:
            filename = location of text file
        Output:
            Creates a dictionary with as key the parameter's name and as value
            the value of this paramater (attribute object)
        """

        with open(filename, 'r') as f:
            line = f.readline().rstrip("\n")
            while line != '':
                parameter, value = line.split('=')
                if value.isdigit():
                    self.parameters[parameter] = int(value)
                else:
                    self.parameters[parameter] = float(value)
                line = f.readline().rstrip("\n")
