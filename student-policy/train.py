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
VALIDATION_SIZE = 1000
validation_observations = observations[SPLIT:SPLIT+VALIDATION_SIZE]
validation_actions = actions[SPLIT:SPLIT+VALIDATION_SIZE]

writer = SummaryWriter()

policy = Policy(4, 2)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

NUMBER_OF_STEPS = 2

# It's a tiny network and a tiny problem so we don't need to do anything fancy
# except stop before it start overfitting.
STEPS = 100000
for step in range(STEPS):
  policy.train()
  training_observations = training_observations.reshape([int(SPLIT / NUMBER_OF_STEPS), NUMBER_OF_STEPS, -1])
  model_actions = policy(torch.as_tensor(training_observations, dtype=torch.float32))
  teacher_actions = torch.as_tensor(training_actions, dtype=torch.long)
  # Three time indices. The data is sequential. Occasionally this might hit a split in the episodes,
  # but we don't care as the chatbot generally also sees the history across episodes.
  teacher_actions = teacher_actions.reshape([int(SPLIT / NUMBER_OF_STEPS), NUMBER_OF_STEPS])
  loss = torch.sum(torch.square(model_actions - torch.nn.functional.one_hot(teacher_actions, 2)))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  policy.eval()
  validation_observations = validation_observations.reshape([int(VALIDATION_SIZE / NUMBER_OF_STEPS), NUMBER_OF_STEPS, -1])
  validation_model_actions = policy(torch.as_tensor(validation_observations, dtype=torch.float32))
  validation_teacher_actions = torch.as_tensor(validation_actions, dtype=torch.long)
  validation_teacher_actions = validation_teacher_actions.reshape([int(VALIDATION_SIZE / NUMBER_OF_STEPS), NUMBER_OF_STEPS])

  validation_loss = torch.sum(torch.square(validation_model_actions - torch.nn.functional.one_hot(validation_teacher_actions, 2)))

  writer.add_scalar("loss/train", loss, step)
  writer.add_scalar("loss/validation", validation_loss, step)

  print(f"Step: {step}/{STEPS}. Loss: {loss}. Validation loss: {validation_loss}")

  if step % 1000 == 0:
    torch.save(policy.state_dict(), "student-policy.pt")
  scheduler.step()
torch.save(policy.state_dict(), "student-policy.pt")

