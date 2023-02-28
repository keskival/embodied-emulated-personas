import torch

class Policy(torch.nn.Module):

  def __init__(self, state_space, action_space):
    super(Policy, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(state_space, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, action_space),
      torch.nn.Softmax()
    )

  def forward(self, x):    
    return self.model(x)
