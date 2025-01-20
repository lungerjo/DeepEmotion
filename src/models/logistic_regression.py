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
