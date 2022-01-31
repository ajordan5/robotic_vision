import torch.nn as nn
import torch
from torchvision import models
from torch import optim
import time
from custom_dataset import DataPrepper


def tl_resnet50(data_path, save_path, num_epochs, num_outputs=2, batch_size=32):
    """Function used to train a model with transfer learning by leveraging the inner layers of resnet50
    
    Input:
        data_path (string): path to data containing train and valid directories
        save_path (string): path where the model dictionary will be saved after training
        num_epochs (int): Desired number of epochs
        num_outputs (int): Number of possible outcomes from the neural net
        batch_size (int): size per batch that will be processed when training and validating the net
    """

    history = []
    # Setup the model
    resnet50 = models.resnet50(pretrained=True)
    data = DataPrepper(data_path, batch_size)
    # Freeze model parameters
    for param in resnet50.parameters():
        param.requires_grad = False
    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_outputs), 
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )
    # Use GPU if available
    if torch.cuda.is_available():
        device = 'cuda:0'
        resnet50 = resnet50.to(device)
    else:
        device = 'cpu'
    # Define Optimizer and Loss Function
    loss_func = nn.NLLLoss() # Negative Log Likelihood
    optimizer = optim.Adam(resnet50.parameters())

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, num_epochs))
        # Set to training mode
        resnet50.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(data.train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = resnet50(inputs)
            # Compute loss
            loss = loss_func(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            resnet50.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(data.valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the resnet50
                outputs = resnet50(inputs)
                # Compute loss
                loss = loss_func(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/data.train_data_size 
        avg_train_acc = train_acc/float(data.train_data_size)
        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/data.valid_data_size 
        avg_valid_acc = valid_acc/float(data.valid_data_size)
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        # stop training once you reach desired accuracy
        if avg_valid_acc > 0.94 and avg_train_acc > 0.92:
            print("[INFO] Stopping training and saving the current epoch")
            torch.save(resnet50.state_dict(), save_path)
            break
    # Function saves the last iteration if desired accuracy is never achieved
    torch.save(resnet50.state_dict(), save_path)

if __name__ == "__main__":
    # Run the transfer learning model with the cats and dogs data
    data_path = "./deep_learning/catdog"
    save_path = "./deep_learning/catdog/predictor.pt"
    tl_resnet50(data_path, save_path, 15)