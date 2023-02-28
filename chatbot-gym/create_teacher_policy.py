#!/usr/bin/python

import numpy as np
import json

run_name = "2023-02-28 13:46:18.495354"

with open(f"outputs/{run_name}/sequence.json", "r") as sequences_file:
  # There were extra appends to the sequences due to a bug.
  # These were just the same observations as arrays, skipping them.
  sequences = filter(lambda item: "obs" in item[0], json.load(sequences_file))
  observations = list(map(lambda item: item[0]['obs'], sequences))
  actions = list(map(lambda item: item[1], sequences))
  np.save("observations.npy", np.asarray(observations))
  np.save("actions.npy", np.asarray(actions))
