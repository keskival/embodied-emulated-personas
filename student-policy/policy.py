import torch

class Policy(torch.nn.Module):

  def __init__(self, state_space, action_space):
    super(Policy, self).__init__()
    width = 16
    self.lstm = torch.nn.LSTM(input_size = state_space, hidden_size = width, batch_first = True)
    self.project_to_output = torch.nn.Sequential(
      torch.nn.Dropout(),
      torch.nn.Linear(width, action_space, bias=False)
    )

  def forward(self, x):
    return torch.nn.functional.softmax(self.project_to_output(self.lstm(x)[0]), dim=2)
