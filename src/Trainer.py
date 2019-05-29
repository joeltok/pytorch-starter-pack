import torch.nn as nn
import torch.optim as optim
from Networks import FullyConnectedNN

max_epochs = 10
lr = 0.001
momentum = 0.01

class Trainer():

  def __init__(self, net, training_generator):
    self.net = net
    self.training_generator = training_generator

    # specify criterion
    self.criterion = nn.MSELoss()

    # specify optimizer
    self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

  def run(self):
    print('Start Training')

    for epoch in range(max_epochs):
      running_loss = 0.0
      for i, data in enumerate(self.training_generator, 0):
        inputs, labels = data

        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
          print('[%d, %5d] loss: %0.3f' %
                (epoch + 1, i + 1, running_loss / 200))
          running_loss = 0.0
    print('Finished Training')
