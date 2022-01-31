import torch
import torchvision
from torchvision import datasets, transforms
from number_net import SimpleNet, Trainer
import matplotlib.pyplot as plt


# Load dataset for handwritten numbers
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# Instantiate the NN and trainer objects
net = SimpleNet()
trainer = Trainer(net)

trainer.train(trainset, testset)

# How to save the trained model
path = "./deep_learning/numbers.pt"
torch.save(net.state_dict(), path)

# How to load the saved model
new_net = SimpleNet()
new_net.load_state_dict(torch.load(path))
new_net.eval()

# Test loaded model
with torch.no_grad():
    for data in testset:
        X, y = data
        output = new_net(X.view(-1,28*28))
        for idx, i in enumerate(output):
            print(torch.argmax(i), y[idx])
            plt.imshow(X[idx].view(28,28))
            plt.show()
            plt.waitforbuttonpress()
        break # Just check one batch




