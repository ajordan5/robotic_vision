import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch

class SimpleNet(nn.Module):
    def __init__(self, image_size=28*28, num_outputs=10, channels=1) -> None:
        super().__init__()
        self.input_size = image_size*channels
        # Build fully connected layer
        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_outputs)
    
    def forward(self, x):
        # Simple nn feedforward with relu to keep scaled between 0-1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class tlNet():
    """Transfer learning net with resnet50"""
    def __init__(self) -> None:
        self.resnet50 = models.resnet50(pretrained=True)
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Change the fully connected layers of resnet50
        fc_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
                            nn.Linear(fc_inputs, 256),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(256, 10), 
                            nn.LogSoftmax(dim=1) # For using NLLLoss()
                        )
        

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_size = 1
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        # Used to flatten out put of convolutional layers
        x = torch.randn(3,50,50).view(-1,3,50,50)
        self._to_linear = None
        self.convs(x)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


class Trainer():
    def __init__(self, net, num_epochs=2, loss_function = nn.CrossEntropyLoss()) -> None:
        self.num_epochs = num_epochs
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=0.0001)
        self.loss_function = loss_function
        self.loss_hist = []
        self.accuracy_hist = []

    def train(self, trainset, testset):
        for epoch in range(self.num_epochs):
            self.net.train()
            for data in trainset:                                   # `data` is a batch of data
                X, y = data                                         # X is the batch of features, y is the batch of targets.
                self.net.zero_grad()                                # sets gradients to 0 before loss calc. You will do this likely every step.
                output = self.net(X.view(-1,self.net.input_size))   # pass in the reshaped batch (recall they are 28x28 atm)
                loss = F.nll_loss(output, y)                        # calc and grab the loss value
                loss.backward()                                     # apply this loss backwards thru the network's parameters
                self.optimizer.step()                               # attempt to optimize weights to account for loss/gradients
            print("Loss for epoch #{}".format(epoch),loss)                                             # print loss. We hope loss (a measure of wrong-ness) declines!
            self.loss_hist.append(loss) 
            correct = 0
            total = 0

            with torch.no_grad():
                for data in testset:
                    X, y = data
                    output = self.net(X.view(-1,self.net.input_size))
                    #print(output)
                    for idx, i in enumerate(output):
                        #print(torch.argmax(i), y[idx])
                        if torch.argmax(i) == y[idx]:
                            correct += 1
                        total += 1
                        

            accuracy = round(correct/total, 3)
            print("Accuracy for epoch #{}: ".format(epoch), accuracy)
            self.accuracy_hist.append(accuracy)