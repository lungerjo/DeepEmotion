from torch.utils.data import Dataset
import torch
import numpy as np
import os
from torchvision import transforms


class SolidDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data_file_paths = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.labels = np.load(label_dir, allow_pickle=True)

    def __len__(self):
        return len(self.data_file_paths)

    def __getitem__(self, idx):
        data_filename = f"{idx}_s.npy" if f"{idx}_s.npy" in self.data_file_paths else f"{idx}_o.npy"
        data = np.load(os.path.join(self.data_dir, data_filename), allow_pickle=True)
        label = self.labels[idx]
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)
        
        return data, label