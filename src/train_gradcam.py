import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.dataset import get_data_loaders
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from models.CNN import CNN
from models.resnet import ResNet, BasicBlock
import time
import wandb
import pickle
from collections import Counter
from tqdm import tqdm
from models.GradCam_test import setup_gradcam

# Our CNN model with a modified get_gradcam_target method.
class CNN(nn.Module):  # 3D CNN
    def __init__(self, cfg, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=1, 
            out_channels=cfg.CNN.c1, 
            kernel_size=(cfg.CNN.k1, cfg.CNN.k2, cfg.CNN.k3), 
            stride=cfg.CNN.stride, 
            padding=cfg.CNN.padding
        )
        self.conv2 = nn.Conv3d(
            in_channels=cfg.CNN.c1, 
            out_channels=cfg.CNN.c2, 
            kernel_size=(cfg.CNN.k1, cfg.CNN.k2, cfg.CNN.k3), 
            stride=cfg.CNN.stride, 
            padding=cfg.CNN.padding
        )
        self.conv3 = nn.Conv3d(
            in_channels=cfg.CNN.c2, 
            out_channels=cfg.CNN.c3, 
            kernel_size=(cfg.CNN.k1, cfg.CNN.k2, cfg.CNN.k3), 
            stride=cfg.CNN.stride, 
            padding=cfg.CNN.padding
        )
        self.pool = nn.MaxPool3d(kernel_size=cfg.CNN.pk, stride=cfg.CNN.ps)
        self.flattened_dim = cfg.CNN.c3 * (132 // 4) * (175 // 4) * (48 // 4)
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def get_gradcam_target(self, x):
        # Process input through conv layers up to conv3, then select the middle slice along depth.
        if x.ndim == 4:  # assume (batch, x, y, z)
            x = x.unsqueeze(1)  # now (batch, 1, x, y, z)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))  # shape: (B, C, D, H, W)
        mid = x.shape[2] // 2  # choose middle slice along depth
        return x[:, :, mid, :, :]  # shape: (B, C, H, W)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# A GradCAM wrapper that uses the CNN's get_gradcam_target output.
class GradCamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Return the 4D tensor (B, C, H, W) from the target layer.
        return self.model.get_gradcam_target(x)

def setup_gradcam(model, use_cuda=False):
    target_layers = [GradCamWrapper(model)]
    cam = XGradCAM(
        model=GradCamWrapper(model),
        target_layers=target_layers
        # use_cuda=use_cuda
    )
    return cam

# Inference loop for generating GradCAM heatmaps.
@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load validation data (assuming get_data_loaders is defined).
    _, val_dataloader = get_data_loaders(cfg)
    output_dim = len(cfg.data.emotion_idx)
    
    # Initialize and load the CNN model.
    model = CNN(cfg=cfg, output_dim=output_dim)
    if cfg.data.load_model:
        model_path_torch = cfg.data.load_model_path
        print(f"Loading the model from {model_path_torch}...")
        state_dict_torch = torch.load(model_path_torch, weights_only=True)
        model.load_state_dict(state_dict_torch)
        print("Model loaded.")
    model = model.to(device)
    model.eval()
    
    # Set up GradCAM.
    cam = setup_gradcam(model, use_cuda=torch.cuda.is_available())
    
    # Inference loop: iterate over one batch from validation.
    for batch in val_dataloader:
        data, labels = batch["data_tensor"], batch["label_tensor"]
        if data.ndim == 4:
            data = data.unsqueeze(1)
        data = data.float().to(device)
        
        # Get model predictions.
        with torch.no_grad():
            outputs = model(data)
            _, preds = torch.max(outputs, dim=1)
        
        # Create targets for GradCAM (using predicted classes).
        targets = [ClassifierOutputTarget(pred.item()) for pred in preds]
        
        # Generate GradCAM heatmaps. Now the activations are 4D.
        grayscale_cam = cam(input_tensor=data, targets=targets)
        # grayscale_cam should have shape (B, H, W)
        # Visualize the heatmap for the first sample.
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.imshow(grayscale_cam[0], cmap='jet')
        plt.title(f"GradCAM Heatmap - Predicted Label: {preds[0].item()}")
        plt.colorbar()
        plt.show()
        
        break

if __name__ == "__main__":
    main()
