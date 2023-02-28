#!/usr/bin/python

import numpy as np
import seaborn as sns
import json
from matplotlib import pyplot as plt
import pandas as pd

sns.set_theme()

with open(f"fixed_scores.json", "r") as scores_file:
  scores = json.load(scores_file)

scores_df = pd.DataFrame(scores)
print("mean: ", scores_df.mean())

sns.displot(scores, bins=100)
plt.show()
