import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftFocalMSELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        probs = torch.softmax(preds, dim=1)
        ce_loss = F.mse_loss(probs, targets, reduction='none')
        focal_term = self.alpha * ((1 - probs) ** self.gamma)
        loss = focal_term * ce_loss
        return loss.mean()

def get_loss(loss_cfg):
    name = loss_cfg.selected.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "soft_focal_mse":
        return SoftFocalMSELoss(**loss_cfg.soft_focal_mse.params)
    else:
        raise ValueError(f"Unsupported loss function: {loss_cfg.name}")
