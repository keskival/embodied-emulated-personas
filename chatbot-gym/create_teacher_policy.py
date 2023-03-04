#!/usr/bin/python

import numpy as np
import json

run_name = "2023-03-03 21:17:53.874433"

with open(f"outputs/{run_name}/sequence.json", "r") as sequences_file:
  sequences = list(json.load(sequences_file))
  observations = list(map(lambda item: item[0]['obs'], sequences))
  actions = list(map(lambda item: item[1], sequences))
  np.save("observations.npy", np.asarray(observations))
  np.save("actions.npy", np.asarray(actions))
