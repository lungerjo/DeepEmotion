# Get config
import os
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)  # Adjust as needed based on project structure
os.environ['PROJECT_ROOT'] = PROJECT_ROOT
CONFIG_PATH = str(Path(PROJECT_ROOT) / "src" / "configs")

@hydra.main(config_path=CONFIG_PATH, config_name="test", version_base=None)
class EmotionAVDataset(Dataset):
    def __init__(self, cfg: DictConfig, transform=None, target_transform=None):

        annotation_path = cfg.paths.selected_annotation_paths[cfg.paths.selected_annotation]
        self.labels = pd.read_csv(annotation_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label