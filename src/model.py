import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionClassifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) for FashionMNIST classification.
    
    Architecture can be customized:
    - Conv blocks are fixed (for now)
    - Classifier (Dense) layers are dynamic based on config
    """
    def __init__(self, num_classes=10, hidden_layers=[512], activation='relu', dropout_rate=0.5, use_bn=False):
        super(FashionClassifier, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Determine activation function
        self.activation = self._get_activation(activation)
        
        # Build Dynamic Classifier
        # Input image: 28x28 -> Pool1: 14x14 -> Pool2: 7x7
        # Flattened size: 64 * 7 * 7 = 3136
        input_dim = 64 * 7 * 7
        
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            
        # Final output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)

    def _get_activation(self, name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dynamic Classifier
        x = self.classifier(x)
        
        return x