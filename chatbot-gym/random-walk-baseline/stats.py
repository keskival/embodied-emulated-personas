#!/usr/bin/python

import numpy as np
import seaborn as sns
import json
from matplotlib import pyplot as plt
import pandas as pd

sns.set_theme()
run_name = "2023-02-28 12:13:47.146559"

with open(f"outputs/{run_name}/scores.json", "r") as scores_file:
  scores = json.load(scores_file)

scores_df = pd.DataFrame(scores)
print("95th quantile: ", scores_df.quantile(0.95))
print("mean: ", scores_df.mean())
sns.displot(scores, bins=100)
plt.show()
