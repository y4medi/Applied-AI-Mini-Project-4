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

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, save_path='./results/training_curves.png'):
    """
    Plots the training and validation loss and accuracy curves.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, save_path='./results/confusion_matrix.png'):
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def get_device():
    """Returns the appropriate device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")