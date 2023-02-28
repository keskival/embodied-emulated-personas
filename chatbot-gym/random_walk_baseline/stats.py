#!/usr/bin/python

import numpy as np
import seaborn as sns
import json
from matplotlib import pyplot as plt

sns.set_theme()
run_name = "2023-02-28 12:13:47.146559"

with open(f"outputs/{run_name}/scores.json", "r") as scores_file:
  scores = json.load(scores_file)

sns.displot(scores)
plt.show()
