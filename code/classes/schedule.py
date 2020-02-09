#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Representation of scheduler for a Mesa based model.
It randomly selects one agent and performs a step.
"""

import random
from mesa.time import RandomActivation

class OneRandomActivation(RandomActivation):
    def step(self):
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
            break
        self.steps += 1
        self.time += 1
