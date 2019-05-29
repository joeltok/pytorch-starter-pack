from torch.utils.data import DataLoader
from datasets.RadialScatterDataset import RadialScatterDataset

generator_params = {
  'batch_size': 64,
  'shuffle': True,
  'num_workers': 6
}

centers = [
  {
    'x': 1.0,
    'y': 1.0
  },
  {
    'x': 3.0,
    'y': 3.0
  },
  {
    'x': 5.0,
    'y': 5.0
  },
]

fuzz = 0.5

training_dataset = RadialScatterDataset(20000, centers, fuzz=fuzz)
training_generator = DataLoader(training_dataset, **generator_params)

test_dataset = RadialScatterDataset(5000, centers, fuzz=fuzz)
test_generator = DataLoader(test_dataset, **generator_params)

num_label_categories = len(centers)
