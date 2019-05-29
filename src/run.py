from DataLoaders import training_generator, test_generator, num_label_categories
from Networks import FullyConnectedNN
from Trainer import Trainer
from Measures import Measures

net = FullyConnectedNN(2, num_label_categories)
trainer = Trainer(net, training_generator)
measures = Measures(net, test_generator)

curr_accuracy = measures.check_accuracy()
print(f'Net starting accuracy {curr_accuracy}')

trainer.run()

new_accuracy = measures.check_accuracy()
print(f'Net ending accuracy {new_accuracy}')
