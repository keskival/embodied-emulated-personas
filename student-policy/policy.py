import torch

class Policy(torch.nn.Module):

  def __init__(self, state_space, action_space):
    super(Policy, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(state_space, 16, bias=False),
      torch.nn.Dropout(),
      torch.nn.ReLU(),
      torch.nn.Linear(16, action_space, bias=False),
      torch.nn.Softmax()
    )

  def forward(self, x):    
    return self.model(x)
