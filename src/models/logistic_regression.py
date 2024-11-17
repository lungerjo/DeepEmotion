import torch.nn as nn
import torch
import torch.nn.functional as F


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights to zero to favor minimum norm solution
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.sigmoid(self.linear(x))
    
class DeepLogisticRegressionModel(nn.Module):
    hidden_dim = 500
    def __init__(self, input_dim, output_dim=1):
        super(DeepLogisticRegressionModel, self).__init__()
        self.l1 = nn.Linear(input_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply the linear transformation followed by the sigmoid activation
        x = x.flatten(start_dim=1)
        a1 = self.sigmoid(self.l1(x))
        return self.sigmoid(self.l2(a1)), a1
    
class Small3DCNNClassifier(nn.Module):
    def __init__(self, output_dim):
        super(Small3DCNNClassifier, self).__init__()
        # First 3D convolutional layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after convolution and pooling
        # Assuming two pooling operations with kernel_size=2 and stride=2
        self.flattened_dim = 64 * (132 // 4) * (175 // 4) * (48 // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # Add a channel dimension for Conv3d input if necessary
        if x.ndim == 4:  # Check if input is [b, 132, 175, 48]
            x = x.unsqueeze(1)  # Add channel dimension, making shape [b, 1, 132, 175, 48]
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # Flatten for the fully connected layers
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

