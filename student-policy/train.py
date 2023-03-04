#!/usr/bin/python

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from policy import Policy

observations = np.load("../chatbot-gym/observations.npy")
actions = np.load("../chatbot-gym/actions.npy")

print(observations.shape)
# No need for test set split because for that we can just regenerate totally new data.
SPLIT=3000
training_observations = observations[:SPLIT]
training_actions = actions[:SPLIT]
validation_observations = observations[SPLIT:]
validation_actions = actions[SPLIT:]

writer = SummaryWriter()

policy = Policy(4, 2)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

# It's a tiny network and a tiny problem so we don't need to do anything fancy.
STEPS = 10000
for step in range(STEPS):
  model_actions = policy(torch.as_tensor(training_observations, dtype=torch.float32))
  teacher_actions = torch.as_tensor(training_actions, dtype=torch.long)
  loss = torch.sum(torch.square(model_actions - torch.nn.functional.one_hot(teacher_actions, 2)))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  validation_model_actions = policy(torch.as_tensor(validation_observations, dtype=torch.float32))
  validation_teacher_actions = torch.as_tensor(validation_actions, dtype=torch.long)
  validation_loss = torch.sum(torch.square(validation_model_actions - torch.nn.functional.one_hot(validation_teacher_actions, 2)))

  writer.add_scalar("loss/train", loss, step)
  writer.add_scalar("loss/validation", validation_loss, step)

  print(f"Step: {step}/{STEPS}. Loss: {loss}. Validation loss: {validation_loss}")

  torch.save(policy.state_dict(), "student-policy.pt")
