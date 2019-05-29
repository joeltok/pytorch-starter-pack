import torch

class Measures():

  def __init__(self, net, test_generator):
    self.net = net
    self.test_generator = test_generator
    pass

  def check_accuracy(self):
    correct = 0
    total = 0
    with torch.no_grad():
      for data in self.test_generator:
        inputs, labels = data
        outputs = self.net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        _, expected  = torch.max(labels, 1)

        total += labels.size(0)
        correct += (predicted == expected).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network: {accuracy}%')
