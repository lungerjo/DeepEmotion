import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module): #3D
    def __init__(self, cfg, output_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=cfg.CNN.c1, \
            kernel_size=(cfg.CNN.k1, cfg.CNN.k2, cfg.CNN.k3), stride=cfg.CNN.stride, padding=cfg.CNN.padding)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=cfg.CNN.c2, \
            kernel_size=(cfg.CNN.k1, cfg.CNN.k2, cfg.CNN.k3), stride=cfg.CNN.stride, padding=cfg.CNN.padding)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=cfg.CNN.c3, \
            kernel_size=(cfg.CNN.k1, cfg.CNN.k2, cfg.CNN.k3), stride=cfg.CNN.stride, padding=cfg.CNN.padding)
        self.pool = nn.MaxPool3d(kernel_size=cfg.CNN.pk, stride=cfg.CNN.ps)

        self.flattened_dim = 64 * (132 // 4) * (175 // 4) * (48 // 4)
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):

        if x.ndim == 4: # batch dim
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

