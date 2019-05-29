import json
import random
import torch
from torch.utils.data import Dataset, DataLoader

class RadialScatterDataset(Dataset):

  def __init__(self, size, centers=[], fuzz=1.0):
    self.points = []

    for _ in range(size):
      idx = random.randint(0, len(centers)-1)

      center = centers[idx]
      x = center['x'] + random.uniform(-fuzz, fuzz)
      y = center['y'] + random.uniform(-fuzz, fuzz)

      one_hot_label = [0] * len(centers)
      one_hot_label[idx] = 1

      self.points.append({
        'one_hot_label': one_hot_label,
        'x': x,
        'y': y
      })

  def __len__(self):
    return len(self.points)

  def __getitem__(self, idx):
    x = self.points[idx]['x']
    y = self.points[idx]['y']
    one_hot_label = self.points[idx]['one_hot_label']

    input = torch.FloatTensor([x, y])
    one_hot_label = torch.FloatTensor(one_hot_label)

    return input, one_hot_label
