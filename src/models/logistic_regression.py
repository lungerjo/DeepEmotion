import torch.nn as nn
import torch

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply the linear transformation followed by the sigmoid activation
        return self.sigmoid(self.linear(x))
