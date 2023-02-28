#!/usr/bin/python

import torch
import json
import os
from datetime import datetime
from PIL import Image
import gymnasium as gym
from policy import Policy

env = gym.make('CartPole-v1', render_mode='rgb_array')

observation, info = env.reset(seed=42)
states = []
actions = []
scores = []
score = 0

run_name = datetime.now()
os.makedirs(f"outputs/{run_name}", exist_ok=True)

FRAMES = 100000

policy = Policy(4, 2)
policy.load_state_dict(torch.load("student-policy.pt"))
policy.eval()

# We'll run x steps with the same action because the environment is so slow.
STEPS_PER_ACTION = 2

states.append({
  "obs": observation.tolist(),
  "reward": 0,
  "terminated": False
})

for step in range(FRAMES):
  image = env.render()
  if (step < 1000):
    frame = Image.fromarray(image)
    frame.save(f"outputs/{run_name}/{step}.png")

  if step % STEPS_PER_ACTION == 0:
    action_probabilities = policy(torch.as_tensor(observation).unsqueeze(0))[0]
    action = torch.distributions.Categorical(action_probabilities).sample().item()
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
print("Got scores: ", scores)
env.close()
