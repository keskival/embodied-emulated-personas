import torch

class Policy(torch.nn.Module):

  def __init__(self, state_space, action_space):
    super(Policy, self).__init__()
    width = 64
    self.project_to_output = torch.nn.Sequential(
      torch.nn.Linear(state_space, width),
      torch.nn.Dropout(p = 1 - 4/64),
      torch.nn.ReLU(),
      torch.nn.Linear(width, action_space),
    )

  def forward(self, x):
    return torch.nn.functional.softmax(self.project_to_output(x), dim=2)
