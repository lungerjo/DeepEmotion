import torch
from torch.utils.data import Dataset, DataLoader
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


class CrossSubjectDataset(Dataset):
    def __init__(self, data_root: Path, subjects: List[str], file_pattern_template: str, runs: List[str], verbose: bool = False):
        """
        Initialize the dataset to load all subjects and all runs.

        Args:
        - data_root: The root path to the data directory.
        - subjects: List of subject directories to include.
        - file_pattern_template: The file pattern template that will be used to find the files.
        - runs: List of runs (e.g., ["01", "02", "03"]) to include.
        - verbose: Whether to print verbose output during dataset loading.
        """
        self.data_root = Path(data_root).resolve()  # Ensure absolute path
        self.subjects = subjects
        self.file_pattern_template = file_pattern_template
        self.runs = runs
        self.verbose = verbose
        self.data_files, self.data, self.num_timepoints = self._load_data()  # Load data and track number of timepoints per file

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
            for run in self.runs:
                subject_dir = self.data_root / subject / "ses-forrestgump/func"
                file_pattern = self.file_pattern_template.format(run)
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
            data.append(tensor_data)
            num_timepoints.append(tensor_data.shape[-1])  # Record the number of time points (t) in the 4D tensor
        
        return data_files, data, num_timepoints

    def __len__(self) -> int:
        """
        Returns the total number of timepoints across all files.
        """
        return sum(self.num_timepoints)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns a single time slice from the dataset.

        Args:
        - idx: The index of the data to retrieve.

        Returns:
        - data_slice: A 3D tensor representing a single time slice with shape [1, x_shape, y_shape, z_shape].
        - file_path: The path to the file from which the data was loaded.
        """
        # Find the file and corresponding time slice for the given index
        cumulative_timepoints = 0
        for i, timepoints in enumerate(self.num_timepoints):
            if idx < cumulative_timepoints + timepoints:
                time_idx = idx - cumulative_timepoints  # Get the time index within the current file
                data_slice = self.data[i][..., time_idx]  # Extract the 3D slice for the given timepoint
                data_slice = data_slice.unsqueeze(0)  # Add the singleton dimension to match [1, x_shape, y_shape, z_shape]
                return data_slice, str(self.data_files[i])
            cumulative_timepoints += timepoints

        raise IndexError(f"Index {idx} out of range")


def get_data_loader(cfg: DictConfig, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    """
    Creates and returns a DataLoader for the fMRI dataset across all subjects and runs.

    Args:
    - cfg: The configuration object loaded by Hydra.
    - batch_size: Batch size for the DataLoader.
    - shuffle: Whether to shuffle the dataset or not.

    Returns:
    - dataloader: The DataLoader for the combined dataset across all subjects and runs.
    """
    derivatives_path = Path(cfg.data.derivatives_path).resolve()
    print(derivatives_path)
    
    # Initialize dataset which will load all data into memory
    dataset = CrossSubjectDataset(data_root=derivatives_path, 
                                  subjects=cfg.data.subjects, 
                                  file_pattern_template=cfg.data.file_pattern_template, 
                                  runs=cfg.data.runs,
                                  verbose=cfg.verbose)
    
    # Create DataLoader from the preloaded dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function that initializes the DataLoader and processes the dataset.

    Args:
    - cfg: The configuration object loaded by Hydra.
    """

    # Get DataLoader for all subjects and runs
    dataloader = get_data_loader(cfg)
    if cfg.verbose: print("DataLoader initialized with preloaded data.")


if __name__ == "__main__":
    main()
