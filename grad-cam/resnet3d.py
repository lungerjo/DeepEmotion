import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet3D(nn.Module):
    """ 
    Simple 3d ResNet for experimentation
    """
    def __init__(self, base_channels=32):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base_channels*2 * 8 * 8 * 8, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1)

        self.gradients = None # place_holder for now

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        """
        This is the feature extractor that returns the activations 
        for the Grad-CAM method.
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

    def forward(self, x, reg_hook=False):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        if reg_hook: # needed for gradcam
            x.register_hook(self.activations_hook)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

    

