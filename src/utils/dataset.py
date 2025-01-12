import zarr
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from io import StringIO
import numpy as np
from omegaconf import DictConfig

class ZarrDataset(Dataset):
    def __init__(self, zarr_path: str):
        # Open the Zarr store in read-only mode.
        print(f"Zarr Dataset: {zarr_path}")
        self.store = zarr.open(zarr_path, mode='r')
        
        # Extract references to the arrays
        self.data = self.store['data']            # shape: (n_files, x, y, z, t_max)
        self.labels = self.store['labels']        # shape: (n_files, t_max)
        self.valid_timepoints = self.store['valid_timepoints']  # shape: (n_files,)
        self.file_paths = self.store['file_paths'][:]  # shape: (n_files,)
        self.subject_ids = self.store['subject_ids'][:]  # shape: (num_subjects,)
        self.session_ids = self.store['session_ids'][:]  # shape: (num_sessions,)
        self.file_to_subject = self.store['file_to_subject'][:]  # shape: (n_files,)
        self.file_to_session = self.store['file_to_session'][:]  # shape: (n_files,)

        # Attributes
        self.emotions = self.store.attrs.get('emotions', [])
        self.aligned_labels_csv = self.store.attrs.get('aligned_labels', None)

        # Assertions to ensure structural integrity
        assert self.data.shape[0] == self.labels.shape[0] == len(self.file_paths), \
            "Mismatch in number of files between data, labels, and file_paths."
        assert len(self.file_to_subject) == self.data.shape[0], \
            "Mismatch between file_to_subject and data."
        assert len(self.file_to_session) == self.data.shape[0], \
            "Mismatch between file_to_session and data."

        # Precompute the valid (volume_idx, row_idx) pairs that have a valid label.
        self.valid_indices = []
        for volume_idx in range(self.data.shape[0]):
            t_max = self.valid_timepoints[volume_idx]
            valid_times = torch.where(torch.tensor(self.labels[volume_idx, :t_max]) != -1)[0]
            for t_idx in valid_times.tolist():
                self.valid_indices.append((volume_idx, t_idx))

        # Print metadata for validation
        print(f"Dataset contains {len(self.file_paths)} files.")
        print(f"Spatial dimensions: {self.data.shape[1:4]}")
        print(f"Maximum timepoints per file: {self.data.shape[4]}")
        print(f"Subjects: {self.subject_ids}")
        print(f"Sessions: {self.session_ids}")
        print(f"Emotion categories: {self.emotions}")
        print(f"Total valid labeled timepoints: {len(self.valid_indices)}")

        # Parse aligned labels if available
        if self.aligned_labels_csv:
            self.aligned_labels = pd.read_csv(StringIO(self.aligned_labels_csv), sep='\t')
        else:
            self.aligned_labels = None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        # Retrieve the (volume_idx, time_idx) for this valid sample
        volume_idx, row_idx = self.valid_indices[idx]

        # Extract the data slice and corresponding label
        data_slice = self.data[volume_idx, :, :, :, row_idx]
        label = self.labels[volume_idx, row_idx]

        # Validate subject/session mapping
        subject = self.file_to_subject[volume_idx]
        session = self.file_to_session[volume_idx]
        assert subject in self.subject_ids, f"Subject {subject} not in subject_ids."
        assert session in self.session_ids, f"Session {session} not in session_ids."

        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data_slice)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Initialize defaults for optional metadata
        time_offset = None
        session_idx = None
        global_idx = idx  # global index with respect to entire dataset

        if self.aligned_labels is not None:
            subset = self.aligned_labels[
                (self.aligned_labels['file_index'] == volume_idx) &
                (self.aligned_labels['row_index'] == row_idx)
            ]
            if not subset.empty:
                time_offset = subset['time_offset'].iloc[0]
                session_idx = subset['session_idx'].iloc[0]
                global_idx = subset['global_idx'].iloc[0]

        return {
            "global_idx": global_idx,  # Global index across all valid timepoints
            "volume_idx": volume_idx,
            "session_idx": session_idx,
            "local_index": row_idx,  # local index within the volume
            "time_offset": time_offset,  
            "data_tensor": data_tensor,
            "label_tensor": label_tensor,
            "file_path": self.file_paths[volume_idx],
            "subject": subject,
            "session": session,
        }

def get_data_loaders(cfg: DictConfig) -> (DataLoader, DataLoader):
    """
    Creates and returns a DataLoader for the ZarrDataset.
    
    Args:
    - cfg: The configuration object loaded by Hydra.
    - zarr_path: Path to the Zarr dataset.
    
    The function assumes:
    - cfg.train.train_ratio: float, ratio of data used for training.
    - cfg.train.batch_size: int, batch size.
    - cfg.train.shuffle: bool, whether to shuffle datasets.
    
    Returns:
    - train_dataloader: DataLoader for training split.
    - val_dataloader: DataLoader for validation split.
    """
    dataset = ZarrDataset(cfg.data.zarr_path)
    print(f"Zarr Path: {cfg.data.zarr_path}")

    # Specify the train-validation split ratio
    train_ratio = cfg.train.train_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    # Example usage:
    zarr_path = "../dataset.zarr"
    # Here cfg would normally come from Hydra, but you can mock it if needed:
    from types import SimpleNamespace
    cfg = SimpleNamespace(train=SimpleNamespace(train_ratio=0.8, batch_size=2, shuffle=True))
    
    # Create the dataset
    zarr_dataset = ZarrDataset(zarr_path)
    
    # Get loaders
    train_loader, val_loader = get_data_loaders(cfg, zarr_path)

    # Inspect some samples from train_loader
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        for i in range(len(batch["data_tensor"])):
            print(f"  Sample {i}:")
            print(f"    File path: {batch['file_path'][i]}")
            print(f"    Subject: {batch['subject'][i]}")
            print(f"    Session: {batch['session'][i]}")
            print(f"    Volume index: {batch['volume_idx'][i]}")
            print(f"    Local Index: {batch['local_index'][i]}")
            print(f"    Global Idx: {batch['global_idx'][i]}")
            print(f"    Session Idx: {batch['session_idx'][i]}")
            print(f"    Time Offset: {batch['time_offset'][i]}")
            print(f"    Data shape: {batch['data_tensor'][i].shape}")
            print(f"    Label: {batch['label_tensor'][i].item()}")
        break
