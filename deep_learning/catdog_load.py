from cv2 import transform
from torchvision import models
import torch
import torch.nn as nn
import cv2
import glob
import random
from custom_dataset import DataPrepper
from PIL import Image

## Test the trained model for detecting cats and dogs 

resnet50 = models.resnet50(pretrained=True)
data = DataPrepper('./deep_learning/catdog', bs=32)
transformer = data.image_transforms['test']
# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2), 
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)
try:
    resnet50.load_state_dict(torch.load("./deep_learning/catdog/predictor.pt"))
except Exception:
    print("INFO: Didn't find a trained model dictionary. You need to train the model in catdog_transferlearn.py first. Make sure you did that.")
resnet50.eval()

# Load files for cat and dog pictures
cats = [file for file in glob.glob("./deep_learning/catdog/test/cat/*.jpg")]
dogs = [file for file in glob.glob("./deep_learning/catdog/test/dog/*.jpg")]
combined = cats+dogs
random.shuffle(combined)
answers = ['cat', 'dog']
for file in combined:
    image = Image.open(file)
    image = transformer(image)
    image = image.view(-1,3,224,224)
    output = resnet50(image)
    image = cv2.imread(file)
    print("I think this is an image of a {}".format(answers[torch.argmax(output)]))
    cv2.imshow("testing", image)
    cv2.waitKey()
    