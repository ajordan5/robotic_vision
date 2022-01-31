from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import os


class DataPrepper:
    """Prepare data for training of a neural network
    
    Inputs:
    
    path (string): string containing the path to organized data. Should contain a directory for 
                    train, valid and test with a folder for each class in all three directories
    bs (int): Desired batch size"""

    def __init__(self, path, bs) -> None:
        # Applying Transforms to the Data
        self.image_transforms = { 
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        }
        # Image transforms
        self.train_directory = os.path.join(path, "train")
        self.valid_directory = os.path.join(path, "valid")
        self.test_directory = os.path.join(path, "test")
        data = {
            'train': datasets.ImageFolder(root=self.train_directory, transform=self.image_transforms['train']),
            'valid': datasets.ImageFolder(root=self.valid_directory, transform=self.image_transforms['valid']),
            'test': datasets.ImageFolder(root=self.test_directory, transform=self.image_transforms['test'])
        }
        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.num_classes = len(os.listdir(self.train_directory))
        self.train_data_size = len(data['train'])
        self.valid_data_size = len(data['valid'])
        self.test_data_size = len(data['test'])
        # Create iterators for the Data loaded using DataLoader module
        self.train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
        self.valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
        self.test_data = DataLoader(data['test'], batch_size=bs, shuffle=True)
