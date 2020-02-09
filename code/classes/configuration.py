#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import re
import csv
import cv2 as cv
import numpy as np
import pandas as pd
import networkx as nx
from .agents import Gang


class Configuration(object):

    def __init__(self, filenames):
        """
        """
        
        self.gang_info = {}
        self.load_gangs(filenames["GANG_INFO"])
        self.total_gangs = len(self.gang_info)

        self.road_dens = []
        self.load_road_density(filenames["ROAD_TXT"])

        self.bounds = []
        self.load_bounds(filenames["BOUNDS"])
        self.areas = {}
        self.load_areas(filenames["AREAS"], self.bounds)

        self.boundaries = []
        self.load_region_matrix(filenames["REGIONS"])

        self.observed_gr = nx.Graph()
        self.all_gr = nx.Graph()
        self.load_connectivity_matrix(
                    filenames["OBSERVED_NETWORK"], 
                    self.gang_info
                    )

        self.gtg_gr = nx.Graph()
        self.load_gtg_matrix(filenames["GTG_NETWORK"], self.gang_info)

        self.colors = []
        self.load_colors(filenames["COLORS"])
        self.colors = ["#{:02x}{:02x}{:02x}"
                        .format(color[0], color[1], color[2]) 
                        for color in self.colors]

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
        Loads the road density from a txt file into 
        a 2D numpy array
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
        to the differen regions of Hollenbeck.
        """
        areas = cv.imread(filename)

        # loop over the boundaries
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
                        self.areas[(m, 690 - n)] = l

            cv.waitKey(0)
            cv.destroyAllWindows()

        width, height = 1036, 691
        x_coords = set(range(width))
        y_coords = set(range(height))
        all_coords = {(x, y) for x in x_coords for y in y_coords}
        missing_coords = all_coords ^ set(self.areas.keys())

        for coord in missing_coords:
            self.areas[coord] = 24

    def load_region_matrix(self, filename):
        """
        Loads 
        """
        with open(filename, 'r') as csvfile:

            reader = csv.reader(csvfile)
            for row in reader:
                self.boundaries.append([int(float(x)) for x in row])

        self.boundaries = np.array(self.boundaries)

    def load_connectivity_matrix(self, filename, gangs):
        """
        """

        matrix = pd.read_excel(filename)
        matrix = matrix.values
        total_matrix = np.ones((matrix.shape[0], matrix.shape[1]))
        rows, cols = np.where(matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())

        # Add nodes to graph
        for k, gang in gangs.items():
            self.observed_gr.add_node(k, pos=gang.coords)          
        self.observed_gr.add_edges_from(edges)

        # # Also generate a graph with all possible edges for now
        rows, cols = np.where(total_matrix == 1)
        all_edges = zip(rows.tolist(), cols.tolist())

        for k, gang in gangs.items():
            self.all_gr.add_node(k, pos=gang.coords)
        self.all_gr.add_edges_from(all_edges)

    def load_gtg_matrix(self, filename, gangs):
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
