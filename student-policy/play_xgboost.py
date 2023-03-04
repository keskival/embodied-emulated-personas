#!/usr/bin/python

import json
import os
from datetime import datetime
from PIL import Image
import gymnasium as gym
import numpy as np
import xgboost as xgb

env = gym.make('CartPole-v1', render_mode='rgb_array')

observation, info = env.reset(seed=42)
states = []
actions = []
scores = []
score = 0

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

FRAMES = 100000

bst = xgb.Booster()
bst.load_model("xgboost.model")

# We'll run x steps with the same action because the environment is so slow.
STEPS_PER_ACTION = 2

states.append({
  "obs": observation.tolist(),
  "reward": 0,
  "terminated": False
})

print(observation, observation.shape)

for step in range(FRAMES):
  print(f"Step: {step}/{FRAMES}")
  # The model is an LSTM now, so we'll keep track of the last three observations.

  image = env.render()
  if (step < 1000):
    frame = Image.fromarray(image)
    frame.save(f"outputs/{run_name}/{step}.png")

  if step % STEPS_PER_ACTION == 0:
    action = bst.predict(xgb.DMatrix(np.expand_dims(observation, 0)))[0]
    if action < 0.5:
      action = 0
    else:
      action = 1
    print(action)
  actions.append(action)

  observation, reward, terminated, truncated, info = env.step(action)

  states.append({
    "obs": observation.tolist(),
    "reward": reward,
    "terminated": terminated
  })
  score = score + reward

  if terminated or truncated:
    scores.append(score)
    score = 0
    print("Episode ended.")
    observation, info = env.reset()
    states.append({
      "obs": observation.tolist(),
      "reward": 0,
      "terminated": False
    })
  if step % 1000 == 0:
    with open(f"outputs/{run_name}/sequence.json", "w") as sequence:
      sequence.write(json.dumps(list(zip(states, actions))))
    with open(f"outputs/{run_name}/scores.json", "w") as sequence:
      sequence.write(json.dumps(scores))
with open(f"outputs/{run_name}/sequence.json", "w") as sequence:
  sequence.write(json.dumps(list(zip(states, actions))))
with open(f"outputs/{run_name}/scores.json", "w") as sequence:
  sequence.write(json.dumps(scores))
print("Got scores: ", scores)
env.close()
