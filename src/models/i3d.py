import torch
import torch.nn as nn
import torch.nn.functional as F

class I3D(nn.Module):
    def __init__(self, num_classes=400, dropout_rate=0.5):
        super(I3D, self).__init__()
        
        # Modified first conv to accept single channel instead of 3
        self.conv3d_1a = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.maxpool3d_2a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv3d_2b = nn.Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3d_2c = nn.Conv3d(64, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool3d_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.mixed_3b = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3d_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.mixed_4b = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.mixed_4c = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.mixed_4d = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.mixed_4e = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.mixed_4f = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool3d_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        
        self.mixed_5b = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Adjusted average pooling kernel size based on your input dimensions
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.conv3d_0c_1x1 = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        
    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add channel dimension [B, C, D, H, W]
            
        x = self.conv3d_1a(x)
        x = self.maxpool3d_2a(x)
        x = self.conv3d_2b(x)
        x = self.conv3d_2c(x)
        x = self.maxpool3d_3a(x)
        
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3d_4a(x)
        
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool3d_5a(x)
        
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.conv3d_0c_1x1(x)
        x = x.squeeze(2).squeeze(2).squeeze(2)  # Remove spatial dimensions
        
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()  # Call parent class constructor properly
        
        self.branch1 = nn.Conv3d(in_channels, out_1x1, kernel_size=(1, 1, 1))
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, red_3x3, kernel_size=(1, 1, 1)),
            nn.Conv3d(red_3x3, out_3x3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, red_5x5, kernel_size=(1, 1, 1)),
            nn.Conv3d(red_5x5, out_5x5, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(in_channels, out_pool, kernel_size=(1, 1, 1))
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)