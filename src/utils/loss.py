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

import torch.nn as nn
import torch

def get_loss(cfg):
    selected = cfg.loss
    device = cfg.device if hasattr(cfg, "device") else "cuda" if torch.cuda.is_available() else "cpu"

    if selected == "mse":
        return nn.MSELoss(reduction="none")

    elif selected == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif selected == "soft_cross_entropy":
        def soft_ce(pred, target):
            log_probs = torch.nn.functional.log_softmax(pred, dim=1)
            return -(target * log_probs).sum(dim=1).mean()
        return soft_ce

    elif selected == "masked_soft_cross_entropy":
        none_index = cfg.data.classification_emotion_idx["NONE"]
        def masked_soft_ce(pred, target):
            log_probs = torch.nn.functional.log_softmax(pred, dim=1)
            mask = target[:, none_index] < 0.3
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            loss = -(target * log_probs).sum(dim=1)
            return loss[mask].mean()
        return masked_soft_ce

    elif selected == "weighted_soft_cross_entropy":
        freqs = torch.tensor(cfg.data.soft_class_frequencies, dtype=torch.float32)
        weights = 1.0 / (freqs + 1e-6)
        weights = weights / weights.sum()
        weights = weights.to(device)

        def weighted_soft_ce(pred, target):
            log_probs = torch.nn.functional.log_softmax(pred, dim=1)  # [B, C]
            ce = -(target * log_probs)  # [B, C]
            weighted_ce = ce * weights.unsqueeze(0)  # [B, C]
            return weighted_ce.sum(dim=1).mean()
        return weighted_soft_ce

    elif selected == "weighted_masked_soft_cross_entropy":
        freqs = torch.tensor(cfg.data.soft_class_frequencies, dtype=torch.float32)
        weights = 1.0 / (freqs + 1e-6)
        weights = weights / weights.sum()
        weights = weights.to(device)
        none_index = cfg.data.classification_emotion_idx["NONE"]
        lam = .95 #TODO tune

        def combined_loss(pred, target):
            log_probs = torch.nn.functional.log_softmax(pred, dim=1)
            mask = target[:, none_index] < 0.99
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)

            target_masked = target[mask]
            log_probs_masked = log_probs[mask]

            # Base soft CE
            loss_masked = -(target_masked * log_probs_masked).sum(dim=1).mean()

            # Weighted target
            weighted_target = target_masked * weights
            weighted_target = weighted_target / (weighted_target.sum(dim=1, keepdim=True) + 1e-8)
            loss_weighted = -(weighted_target * log_probs_masked).sum(dim=1).mean()

            return (1 - lam) * loss_masked + lam * loss_weighted

        return combined_loss


    else:
        raise ValueError(f"Unsupported loss function: {selected}")


