import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FullyConnectedNN(nn.Module):

  def __init__(self, input_number=100, output_number=10):
    super(FullyConnectedNN, self).__init__()
    self.fc1 = nn.Linear(input_number, 20)
    self.fc2 = nn.Linear(20, 200)
    self.fc3 = nn.Linear(200, output_number)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return F.log_softmax(x, dim=1)
