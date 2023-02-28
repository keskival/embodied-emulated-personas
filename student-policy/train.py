#!/usr/bin/python

import numpy as np
import torch

from policy import Policy

observations = np.load("../chatbot-gym/observations.npy")
actions = np.load("../chatbot-gym/actions.npy")

policy = Policy(4, 2)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# It's a tiny network and a tiny problem so we don't need to do anything fancy.
STEPS = 100000
for step in range(STEPS):
  model_actions = policy(torch.as_tensor(observations, dtype=torch.float32))
  teacher_actions = torch.as_tensor(actions, dtype=torch.long)
  loss = torch.sum(torch.square(model_actions - torch.nn.functional.one_hot(teacher_actions, 2)))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print(f"Step: {step}/{STEPS}. Loss: {loss}")
torch.save(policy.state_dict(), "student-policy.pt")
