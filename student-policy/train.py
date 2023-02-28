#!/usr/bin/python

import numpy as np
import torch

observations = np.load("../chatbot-gym/observations.npy")
actions = np.load("../chatbot-gym/actions.npy")
print(observations)
print(actions)
