import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def get_data_loaders(batch_size=64, data_dir='./data'):
    """
    Creates DataLoaders for FashionMNIST train and test sets.
    Applies normalization and basic augmentation.
    """
    # Define transforms
    # ToTensor converts [0, 255] to [0.0, 1.0]
    # Normalize scales to standard normal distribution (mean=0.5, std=0.5 for grayscale)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load training data
    train_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = train_dataset.classes
    
    return train_loader, test_loader, classes



def get_device():
    """Returns the appropriate device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")