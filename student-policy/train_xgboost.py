#!/usr/bin/python

import numpy as np

import xgboost as xgb

observations = np.load("../chatbot-gym/observations.npy")
actions = np.load("../chatbot-gym/actions.npy")

print(observations.shape)
# No need for test set split because for that we can just regenerate totally new data.
SPLIT=3000
training_observations = observations[:SPLIT]
training_actions = actions[:SPLIT]
VALIDATION_SIZE = 1000
validation_observations = observations[SPLIT:SPLIT+VALIDATION_SIZE]
validation_actions = actions[SPLIT:SPLIT+VALIDATION_SIZE]

dtrain = xgb.DMatrix(training_observations, label=training_actions)
bst = xgb.train({}, dtrain, 10)
bst.save_model("xgboost.model")
