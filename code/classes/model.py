#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This model represent a computational model to simulate multiple gangs with a 
certain amount of members in a certain area (Hollenbeck) that walk around 
according a given human mobility algrotihm (Brownian Motion, Semi-biased Levy 
walk and Gravity walk). It keeps track of the rivavlry network and its 
corresponding statistics.
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

# Import own modules
from .agents import Gang, GangMemberB, GangMember_SBLN, GangMemberG
from .schedule import OneRandomActivation
from code.helpers.helpers import accuracy_graph, shape_metrics

class GangRivalry(Model):
    """
    A computational model that contains all the parameters, tracking variables 
    and simulations fiction needed to simulate gang members walking around in 
    the area and to constuct a resulting rivalry network.
    """

    def __init__(self, config, algorithm, iter_check):
        """
        Initialize each model with a certaing configuration (parameters), algorithm and moment of data collection.

        Input:
            config = An object containing all the necassry parameters
            algorithm = string which determines the algorithm for human mobility
            iter_check = moment of data collection

        Output:
            An model object with which the user can run the simulations
        """

        super().__init__()
        self.config = config
        self.algorithm = algorithm
        self.iter_check = iter_check

        # Extract parameters from configuration obejct
        self.width = self.config.road_dens.shape[1] - 1
        self.height = self.config.road_dens.shape[0] - 1
        self.min_jump = self.config.parameters["min_jump"]
        self.lower_max_jump = self.config.parameters["lower_max_jump"]
        self.upper_max_jump = self.config.parameters["upper_max_jump"]
        self.weight_home = self.config.parameters["weight_home"]
        self.bounded_pareto = self.config.parameters["bounded_pareto"]
        self.kappa = self.config.parameters["kappa"]
        self.beta = self.config.parameters["beta"]
        self.vision = self.config.parameters["vision"]
        self.threshold = self.config.parameters["threshold"]

        # Create continuousSpace area for the given widht and height
        self.area = ContinuousSpace(self.width, self.height, False)

        # Determine normalized gang size (this is especially for gravity model)
        gang_sizes = [gang.size for gang in self.config.gang_info.values()]
        self.min_gang = min(gang_sizes)
        self.max_gang = max(gang_sizes)
        self.norm_gang_size = [
            (size - self.min_gang) / (self.max_gang - self.min_gang) 
            for size in gang_sizes
            ]

        # Set scheduler for model
        self.schedule = OneRandomActivation(self)

        # Distribute gang members at their home location
        self.init_population()

        # Initialize a rivalry matrix
        self.rivalry_matrix = np.zeros(
                (self.config.total_gangs, self.config.total_gangs)
                )

        # Initialize a graph
        self.create_graph()

        # Initialize datacollector
        self.datacollector = DataCollector(
            model_reporters={
                "Accuracy": accuracy_graph,
                "Shape": shape_metrics}
        )
        self.running = True
        self.datacollector.collect(self)

    def init_population(self):
        """
        Distribute gang members at their home location
        """
        for gang in self.config.gang_info.values():
            for _ in range(gang.size):
                self.new_agent(gang.coords, gang.number)

    def new_agent(self, pos, name):
        """
        Create new agent according to the given algorithm for human mobility, 
        add the agent to the area and to the scheduler.
        """
        agent = None

        if self.algorithm == "BM":
            agent = GangMemberB(self.next_id(), self, pos, 
                                name, self.beta, self.vision)

        elif self.algorithm == "SBLN":
            agent = GangMember_SBLN(self.next_id(), self, pos, name, 
                                    self.min_jump, self.weight_home, 
                                    self.bounded_pareto, self.kappa, 
                                    self.vision, self.beta)
        
        elif self.algorithm == "GRAV":
            agent = GangMemberG(self.next_id(), self, pos, name, 
                                self.min_jump, self.weight_home, 
                                self.bounded_pareto, self.kappa, 
                                self.vision, self.beta)

        self.area.place_agent(agent, pos)
        self.schedule.add(agent)

    def update_rivalry(self, agent1, agent2):
        """
        Update rivalry matrix for two given agents.
        """
        self.rivalry_matrix[agent1.number, agent2.number] += 1
        self.rivalry_matrix[agent2.number, agent1.number] += 1

    def create_graph(self):
        """
        Initialize a graph with the nodes being the home locations of the gangs.
        """
        gr = nx.Graph()
        for gang in self.config.gang_info.values():
            gr.add_node(gang.number, pos=gang.coords)
        self.gr = gr

    def make_graph(self):
        """
        Create the edges of the graph depending on the current status of the 
        rivalry matrix.
        """

        # Retrieve edges and remove them from the graph
        edges = set(self.gr.edges).copy()
        self.gr.remove_edges_from(edges)

        # Start algorithm to determine if edges between gangs should be placed
        shape = self.config.total_gangs
        for i in range(shape):
            total_interactions = np.sum(self.rivalry_matrix[i])
            for j in range(shape):

                # If relative rivalry is greater than threshold than add edge
                if total_interactions:
                    rival_strength = self.rivalry_matrix[i][j] / \
                        total_interactions
                    if rival_strength > self.threshold:
                        self.gr.add_edge(i, j)

    def step(self):
        """
        A step to perform at each iteration.
        """
        self.schedule.step()

    def run_model(self, step_count=200):
        """
        Run the model for a certain amount of iterations. Also updates the 
        state of the graph and collects data at specific collection moments.

        Input:
            step_count = amount of iterations (integer)
        Output:
            A numpy array containing the final state of the rivalry matrix.
        """
        for i in tqdm(range(step_count)):     
            self.step()
            if i % self.iter_check == 0:
                self.make_graph()
                self.datacollector.collect(self)
        return self.rivalry_matrix