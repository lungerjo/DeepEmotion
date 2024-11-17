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

class CrossSubjectDataset(Dataset):
    def __init__(self, cfg: DictConfig):
        """
        Initialize the dataset to load all subjects, all runs, and align labels by observer.

        Args:
        - data_path: The root path to the data directory.
        - label_path: The root path to the label directory.
        - subjects: List of subject directories to include.
        - file_pattern_template: The file pattern template that will be used to find the files.
        - runs: List of runs (e.g., ["01", "02", "03"]) to include.
        - session_offsets: List of session-based time offsets for alignment.
        - verbose: Whether to print verbose output during dataset loading.
        """
        self.data_path = Path(cfg.data.data_path).resolve()  # Ensure absolute path
        self.label_path = Path(cfg.data.label_path).resolve()
        self.subjects = cfg.data.subjects
        self.file_pattern_template = cfg.data.file_pattern_template
        self.sessions = cfg.data.sessions
        self.session_offsets = cfg.data.session_offsets  # Cumulative time offsets for each session
        self.verbose = cfg.verbose
        self.emotion_idx = cfg.data.emotion_idx
        self.normalization = cfg.data.normalization
        self.observer_labels = pd.read_csv(self.label_path, sep='\t')
        self.data_files, self.data, self.num_timepoints = self._load_data()  # Load data and track number of timepoints per file
        self.aligned_labels = self._align_labels()


    def _apply_offset(self, session_idx: int, timestamp: int) -> int:
        """
        Applies the cumulative time offset based on the session index.

        Args:
        - session_idx: Index of the session (0, 1, 2, ...).
        - timestamp: The raw timestamp in the current session.

        Returns:
        - offset_timestamp: The global timestamp adjusted for the session offset.
        """
        return timestamp + self.session_offsets[session_idx]

    def _align_labels(self):
        """
        Aligns the observer's labels based on the time offset column and the session-based fMRI offsets.
        """
        aligned_labels = []
        for session_idx, session_length in enumerate(self.num_timepoints):
            session_start = sum(self.num_timepoints[:session_idx])  # Cumulative session start time
            session_end = session_start + session_length * 2  # Assuming 2-second resolution

            # Filter labels within this session's time range
            session_labels = self.observer_labels[(self.observer_labels['offset'] >= session_start) &
                                                  (self.observer_labels['offset'] < session_end)].copy()

            # Apply the session time offset
            session_labels['offset'] = session_labels['offset'].apply(
                lambda t: self._apply_offset(session_idx, t)
            )

            aligned_labels.append(session_labels)
        
        return pd.concat(aligned_labels)

    def _find_files(self) -> List[Path]:
        """
        Searches for files in all subject directories and runs based on the file pattern.

        Returns:
        - files: A list of file paths that match the file pattern.
        """
        files = []
        if self.verbose:
            print(f"Finding files")
        for subject in self.subjects:
            for session in self.sessions:
                subject_dir = self.data_path / subject / "ses-forrestgump/func"
                print(subject_dir)
                file_pattern = self.file_pattern_template.format(session)
                matched_files = list(subject_dir.rglob(file_pattern))  # Recursively find files matching the pattern
                files.extend(matched_files)
        if self.verbose:
            print(f"{len(files)} files found")
        return files

    def _load_data(self) -> Tuple[List[Path], List[torch.Tensor], List[int]]:
        """
        Loads all the .nii.gz data into memory and tracks the number of timepoints (t) per file.

        Returns:
        - data_files: List of file paths corresponding to the loaded data.
        - data: List of 4D tensors loaded from the .nii.gz files.
        - num_timepoints: List of the number of timepoints (t) for each file.
        """
        data_files = self._find_files()
        data = []
        num_timepoints = []
        
        for file_path in data_files:
            if self.verbose:
                print(f"Loading data from {file_path}")
            nii_data = nib.load(str(file_path)).get_fdata()  # Load the .nii.gz file using nibabel
            tensor_data = torch.tensor(nii_data)  # Convert the loaded data to torch tensor
            
            if self.normalization:
                tensor_data = (tensor_data - tensor_data.mean()) / (tensor_data.std() + 1e-5) # Normalize the data
            
            data.append(tensor_data)
            num_timepoints.append(tensor_data.shape[-1])  # Record the number of time points (t) in the 4D tensor
        
        return data_files, data, num_timepoints

    def __len__(self) -> int:
        """
        Returns the total number of timepoints across all files.
        """
        return sum(self.num_timepoints)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single time slice from the dataset along with one-hot encoded label.

        Args:
        - idx: The index of the data to retrieve.

        Returns:
        - data_slice: A 3D tensor representing a single time slice with shape [1, x_shape, y_shape, z_shape].
        - one_hot_label: A one-hot encoded tensor for the corresponding emotion label.
        """
        # Find the file and corresponding time slice for the given index
        cumulative_timepoints = 0
        for i, timepoints in enumerate(self.num_timepoints):
            if idx < cumulative_timepoints + timepoints:
                time_idx = idx - cumulative_timepoints  # Get the time index within the current file
                data_slice = self.data[i][..., time_idx]  # Extract the 3D slice for the given timepoint

                # Use row index directly to fetch the label (assuming no misalignment)
                if idx >= len(self.aligned_labels):
                    raise IndexError(f"Index {idx} out of bounds for label data with length {len(self.aligned_labels)}")

                # Fetch the label using the row index
                label = str(self.aligned_labels.iloc[idx]['emotion'])  # Get the label using the row index
                label_idx = self.emotion_idx[label]  # Convert the string label to an integer index
                
                # Convert the integer label to a one-hot encoded tensor
                one_hot_label = torch.nn.functional.one_hot(torch.tensor(label_idx), num_classes=len(self.emotion_idx)).float()

                return data_slice, one_hot_label
            
            cumulative_timepoints += timepoints

        raise IndexError(f"Index {idx} out of range")

    

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
