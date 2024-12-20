import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import nibabel as nib
import hydra
from omegaconf import DictConfig
from typing import List, Tuple
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import psutil

import torch
from torch.utils.data import Dataset
from pathlib import Path
import nibabel as nib
from typing import List, Tuple
import pandas as pd
import numpy as np

def log_memory_usage():
    """Logs CPU and GPU memory usage."""

    # CPU Memory
    mem = psutil.virtual_memory()
    print(f"CPU Memory - Free: {mem.free / 1e6:.2f} MB, Used: {mem.used / 1e6:.2f} MB, Total: {mem.total / 1e6:.2f} MB")
    
    # GPU Memory
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # GPU index
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory - Free: {info.free / 1e6:.2f} MB, Used: {info.used / 1e6:.2f} MB, Total: {info.total / 1e6:.2f} MB")

class CrossSubjectDataset(Dataset):
    def __init__(self, cfg):
        """
        Initialize the dataset without loading data slices into memory.
        Data slices will be loaded on-the-fly during batch loading in __getitem__.
        """
        self.data_path = Path(cfg.data.data_path).resolve()
        self.label_path = Path(cfg.data.label_path).resolve()
        self.subjects = cfg.data.subjects
        self.file_pattern_template = cfg.data.file_pattern_template
        self.sessions = cfg.data.sessions
        self.session_offsets = cfg.data.session_offsets  # Cumulative time offsets for alignment
        self.verbose = cfg.verbose
        self.emotion_idx = cfg.data.emotion_idx
        self.normalization = cfg.data.normalization
        self.observer_labels = pd.read_csv(self.label_path, sep='\t')

        if self.verbose:
            log_memory_usage()

        # Find data files and get number of timepoints
        self.data_files_per_session, self.num_timepoints_per_session = self._load_data_info()

        # Flatten the data files and num_timepoints lists
        self.data_files = []
        self.num_timepoints = []
        for session_files, session_num_timepoints in zip(self.data_files_per_session, self.num_timepoints_per_session):
            self.data_files.extend(session_files)
            self.num_timepoints.extend(session_num_timepoints)

        # Align labels and filter out 'NONE' labels
        self.aligned_labels = self._align_labels()
        if self.verbose:
            print(f"aligned_labels: shape {self.aligned_labels.shape[0]}")
        self.aligned_labels = self.aligned_labels[self.aligned_labels['emotion'] != 'NONE'].reset_index(drop=True)
        if self.verbose:
            print(f"filtered_labels: shape {self.aligned_labels.shape[0]}")

        # Create index mappings between labels and data slices
        self.index_mappings = self._create_index_mappings()

    def _align_labels(self):
        """
        Align the observer's labels based on the provided session_offsets and the first subject's timepoints.
        """
        aligned_labels = []
        TR = 2  # seconds per TR

        for session_idx, session_num_timepoints in enumerate(self.num_timepoints_per_session):
            # Use just the first subject’s time for calculating the session length
            # Assumes all subjects have the same number of timepoints per session.
            session_total_timepoints = session_num_timepoints[0]

            session_start = self.session_offsets[session_idx]

            if session_idx < len(self.session_offsets) - 1:
                # If next offset is defined, that gives us session_end directly
                session_end = self.session_offsets[session_idx + 1]
            else:
                # Last session: compute end from timepoints
                session_end = session_start + session_total_timepoints * TR

            if self.verbose:
                print(f"session_{session_idx}_start {session_start}, end {session_end}")

            # Filter labels within this session's time range
            session_labels = self.observer_labels[
                (self.observer_labels['offset'] >= session_start) &
                (self.observer_labels['offset'] < session_end)
            ].copy()

            # If you need to apply offset to labels for alignment (optional)
            # session_labels['offset'] = session_labels['offset'].apply(
            #     lambda t: self._apply_offset(session_idx, t)
            # )

            aligned_labels.append(session_labels)

        return pd.concat(aligned_labels, ignore_index=True)


    def _find_files(self) -> List[List[Path]]:
        """Searches for files in all subject directories and runs based on the file pattern."""
        files_per_session = []
        if self.verbose:
            print(f"Finding files")
        for session in self.sessions:
            session_files = []
            for subject in self.subjects:
                subject_dir = self.data_path / subject / "ses-forrestgump/func"
                if self.verbose:
                    print(f"Looking in {subject_dir}")
                file_pattern = self.file_pattern_template.format(session)
                matched_files = list(subject_dir.rglob(file_pattern))
                session_files.extend(matched_files)
            files_per_session.append(session_files)
        if self.verbose:
            print(f"{sum(len(files) for files in files_per_session)} files found")
        return files_per_session

    def _load_data_info(self):
        """Gets the number of timepoints for each data file without loading data into memory."""
        files_per_session = self._find_files()
        data_files_per_session = []
        num_timepoints_per_session = []

        for session_idx, session_files in enumerate(files_per_session):
            session_data_files = []
            session_num_timepoints = []
            for file_path in session_files:
                nii_img = nib.load(str(file_path))
                shape = nii_img.shape  # (x, y, z, t)
                session_num_timepoints.append(shape[-1])  # Number of timepoints (t)
                session_data_files.append(file_path)
            data_files_per_session.append(session_data_files)
            num_timepoints_per_session.append(session_num_timepoints)

        if self.verbose:
            print(f"num_timepoints_per_session {num_timepoints_per_session}")

        return data_files_per_session, num_timepoints_per_session

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
            if data_file_idx >= len(cumulative_timepoints) - 1:
                data_file_idx -= 1
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
        Loads the data slice from .nii.gz file on-the-fly when the batch is loaded.
        Returns a data slice and one-hot encoded label for the given index.
        """
        mapping = self.index_mappings[idx]
        data_file_idx = mapping['data_file_idx']
        time_idx = mapping['time_idx']
        label_idx = mapping['label_idx']

        data_file_path = self.data_files[data_file_idx]

        # Load data slice using memory-mapped access
        nii_img = nib.load(str(data_file_path), mmap=True)
        data_slice = nii_img.dataobj[..., time_idx]
        # Convert to numpy array (loads only the slice into memory)
        data_slice = np.array(data_slice, dtype=np.float32)

        # Optionally apply normalization
        if self.normalization:
            data_slice = (data_slice - np.mean(data_slice)) / (np.std(data_slice) + 1e-5)

        data_slice = torch.tensor(data_slice, dtype=torch.float32)

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
