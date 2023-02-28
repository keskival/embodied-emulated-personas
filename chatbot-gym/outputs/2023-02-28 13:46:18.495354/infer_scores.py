#!/usr/bin/python

import gymnasium as gym
import json
import os
from datetime import datetime
from PIL import Image
import time

scores = []
score = 0

with open("sequence.json", "r") as sequences_file:
  sequences = json.load(sequences_file)
  for item in sequences:
    if ("terminated" in item[0]):
      # There were extra appends to the sequences.
      # These were just observations as arrays, skipping them.
      score += item[0]["reward"]
      if (item[0]["terminated"]):
        print(item)
        scores.append(score)
        score = 0
print(scores)
with open("fixed_scores.json", "w") as new_scores_file:
  json.dump(scores, new_scores_file)
