import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionClassifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) for FashionMNIST classification.
    
    Architecture:
    - Conv2d (32 filters) -> BatchNorm -> ReLU -> MaxPool
    - Conv2d (64 filters) -> BatchNorm -> ReLU -> MaxPool
    - Flatten
    - Linear (Dense) -> ReLU -> Dropout
    - Linear (Output)
    """
    def __init__(self, num_classes=10):
        super(FashionClassifier, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully Connected Layers
        # Input image is 28x28. 
        # After pool1 (2x2): 14x14
        # After pool2 (2x2): 7x7
        # Flattened size: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x