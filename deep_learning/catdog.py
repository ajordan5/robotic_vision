from custom_dataset import DataPrepper
from number_net import SimpleNet, NewTrainer, SimpleCNN, tlNet
from torchvision import models
import torch.nn as nn

# Attempt at training the cats and dogs NN with a CNN, doesn't work so well at the moment. 'transferlearn.py' has a successful implementation
data = DataPrepper("./deep_learning/catdog", 32)
#model = SimpleNet(image_size=224*224*3, num_outputs=2)
model = SimpleCNN()
trainer = NewTrainer(model, num_epochs=50, loss_function=nn.MSELoss())
trainer.train(data.train_data, data.test_data)