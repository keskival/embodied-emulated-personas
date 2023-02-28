#!/usr/bin/python

import gymnasium as gym
import json
import os
from datetime import datetime
import numpy as np

env = gym.make('CartPole-v1', render_mode='rgb_array')

observation, info = env.reset(seed=42)
states = []
actions = []
prompts = []
scores = []
score = 0

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

FRAMES = 100000

# We'll run x steps with the same action because the environment is so slow.
STEPS_PER_ACTION = 2

states.append({
  "obs": observation.tolist(),
  "reward": 0,
  "terminated": False
})

for step in range(FRAMES):
  if step % STEPS_PER_ACTION == 0:
    action = np.random.randint(0,2)
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

  with open(f"outputs/{run_name}/sequence.json", "w") as sequence:
    sequence.write(json.dumps(list(zip(states, actions))))
  with open(f"outputs/{run_name}/scores.json", "w") as sequence:
    sequence.write(json.dumps(scores))
print(scores)
env.close()
