import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import nibabel as nib
import hydra
from omegaconf import DictConfig
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
from typing import List, Tuple
import pandas as pd
import numpy as np

class CrossSubjectDataset(Dataset):
    def __init__(self, cfg: DictConfig):
        """
        Initialize the dataset, load all subjects, all runs, align labels by observer,
        and exclude 'NONE' labels along with their corresponding data slices.
        """
        self.data_path = Path(cfg.data.data_path).resolve()  # Ensure absolute path
        self.label_path = Path(cfg.data.label_path).resolve()
        self.subjects = cfg.data.subjects
        self.file_pattern_template = cfg.data.file_pattern_template
        self.sessions = cfg.data.sessions
        self.session_offsets = cfg.data.session_offsets  # Cumulative time offsets for alignment
        self.verbose = cfg.verbose
        self.emotion_idx = cfg.data.emotion_idx
        self.normalization = cfg.data.normalization
        self.observer_labels = pd.read_csv(self.label_path, sep='\t')

        # Load data and track number of timepoints per file
        self.data_files, self.data, self.num_timepoints = self._load_data()

        # Align labels and filter out 'NONE' labels
        self.aligned_labels = self._align_labels()
        self.aligned_labels = self.aligned_labels[self.aligned_labels['emotion'] != 'NONE'].reset_index(drop=True)

        # Create index mappings between labels and data slices
        self.index_mappings = self._create_index_mappings()

    def _apply_offset(self, session_idx: int, timestamp: int) -> int:
        """Applies the cumulative time offset based on the session index."""
        return timestamp + self.session_offsets[session_idx]

    def _align_labels(self):
        """Aligns the observer's labels based on time offsets and session-based fMRI offsets."""
        aligned_labels = []
        for session_idx, session_length in enumerate(self.num_timepoints):
            session_start = sum(self.num_timepoints[:session_idx]) * 2  # Assuming 2-second TR
            session_end = session_start + session_length * 2

            # Filter labels within this session's time range
            session_labels = self.observer_labels[
                (self.observer_labels['offset'] >= session_start) &
                (self.observer_labels['offset'] < session_end)
            ].copy()

            # Apply the session time offset
            session_labels['offset'] = session_labels['offset'].apply(
                lambda t: self._apply_offset(session_idx, t)
            )

            aligned_labels.append(session_labels)

        return pd.concat(aligned_labels, ignore_index=True)

    def _find_files(self) -> List[Path]:
        """Searches for files in all subject directories and runs based on the file pattern."""
        files = []
        if self.verbose:
            print(f"Finding files")
        for subject in self.subjects:
            for session in self.sessions:
                subject_dir = self.data_path / subject / "ses-forrestgump/func"
                if self.verbose:
                    print(f"Looking in {subject_dir}")
                file_pattern = self.file_pattern_template.format(session)
                matched_files = list(subject_dir.rglob(file_pattern))
                files.extend(matched_files)
        if self.verbose:
            print(f"{len(files)} files found")
        return files

    def _load_data(self) -> Tuple[List[Path], List[torch.Tensor], List[int]]:
        """Loads all the .nii.gz data into memory and tracks the number of timepoints per file."""
        data_files = self._find_files()
        data = []
        num_timepoints = []

        for file_path in data_files:
            if self.verbose:
                print(f"Loading data from {file_path}")
            nii_data = nib.load(str(file_path)).get_fdata()
            tensor_data = torch.tensor(nii_data)

            if self.normalization:
                tensor_data = (tensor_data - tensor_data.mean()) / (tensor_data.std() + 1e-5)

            data.append(tensor_data)
            num_timepoints.append(tensor_data.shape[-1])  # Number of time points (t)

        return data_files, data, num_timepoints

    def _create_index_mappings(self):
        """
        Creates a mapping from dataset indices to data file indices and time indices,
        excluding 'NONE' labels and their corresponding data slices.
        """
        index_mappings = []
        cumulative_timepoints = np.cumsum([0] + self.num_timepoints)

        for idx, row in self.aligned_labels.iterrows():
            # Get the adjusted offset (assuming TR=2s)
            offset = row['offset']
            global_time_idx = int(offset / 2)  # Convert offset to global time index

            # Find the data file index
            data_file_idx = np.searchsorted(cumulative_timepoints, global_time_idx, side='right') - 1
            time_idx_within_file = global_time_idx - cumulative_timepoints[data_file_idx]

            # Check for out-of-bounds time indices
            if time_idx_within_file >= self.num_timepoints[data_file_idx] or time_idx_within_file < 0:
                continue  # Skip invalid indices

            label = row['emotion']
            label_idx = self.emotion_idx[label]

            index_mappings.append({
                'data_file_idx': data_file_idx,
                'time_idx': time_idx_within_file,
                'label_idx': label_idx
            })

        return index_mappings

    def __len__(self) -> int:
        """Returns the number of valid data points (excluding 'NONE' labels)."""
        return len(self.index_mappings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a data slice and one-hot encoded label for the given index.
        """
        mapping = self.index_mappings[idx]
        data_file_idx = mapping['data_file_idx']
        time_idx = mapping['time_idx']
        label_idx = mapping['label_idx']

        data_slice = self.data[data_file_idx][..., time_idx]

        # Ensure data_slice is a 3D tensor
        if data_slice.ndim == 3:
            data_slice = data_slice.unsqueeze(0)  # Add a channel dimension

        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(label_idx), num_classes=len(self.emotion_idx)
        ).float()

        return data_slice, one_hot_label

    

def get_data_loaders(cfg: DictConfig) -> DataLoader:
    """
    Creates and returns a DataLoader for the fMRI dataset across all subjects and runs.

    Args:
    - cfg: The configuration object loaded by Hydra.
    - batch_size: Batch size for the DataLoader.
    - shuffle: Whether to shuffle the dataset or not.

    Returns:
    - dataloader: The DataLoader for the combined dataset across all subjects and runs.
    """
    
    # Initialize dataset which will load all data into memory
    dataset = CrossSubjectDataset(cfg)

    # Specify the train-validation split ratio
    train_ratio = cfg.train.train_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader from the preloaded dataset
    train_dataloader = DataLoader(train_dataset, batch_size = cfg.train.batch_size, shuffle = cfg.train.shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size = cfg.train.batch_size, shuffle = cfg.train.shuffle)
    return train_dataloader, val_dataloader


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader and processes the dataset.

    Args:
    - cfg: The configuration object loaded by Hydra.
    """

    # Get DataLoader for all subjects and runs
    print(cfg.project_root)
    dataloader = get_data_loaders(cfg)
    if cfg.verbose: print("DataLoader initialized with preloaded data.")
    for data, labels in dataloader:
        # data is a batch of fMRI slices
        # labels is a batch of corresponding emotion labels
        print(data.shape, labels.shape)
    


if __name__ == "__main__":
    main()
